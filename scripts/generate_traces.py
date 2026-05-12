"""Generate CoT reasoning traces over BABILong samples with a thinking model.

Runs a thinking-capable model (default Kimi-K2-Thinking) over pulled BABILong samples,
extracts the <think>...</think> trace and the final answer, tags `is_correct` against
the BABILong target, and writes a JSONL whose schema matches what run_teacher.py expects.
"""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Official BABILong prompt pieces (ported from booydar/babilong/babilong/prompts.py).
# Stored in prompts/babilong_prompts.json so prompt edits don't require touching code.
BABILONG_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "babilong_prompts.json"
BABILONG_PROMPTS = json.loads(BABILONG_PROMPTS_PATH.read_text())

# Official BABILong task labels (ported from booydar/babilong/babilong/metrics.py).
TASK_LABELS = {
    "qa1": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa2": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa3": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
}

def format_prompt(sample: dict) -> str:
    """Build the BABILong prompt verbatim (no CoT directive).

    Kimi-K2-Thinking and similar thinking models emit <think>...</think> natively
    without prompting. Adding an explicit CoT directive caused tag-format confusion
    (model interpreted '...' as part of a literal tag name). Trust the model.
    """
    task = sample["qa_task"]
    if task not in BABILONG_PROMPTS:
        raise ValueError(f"No prompt config for task {task!r}. Add it to BABILONG_PROMPTS.")
    p = BABILONG_PROMPTS[task]
    return (
        f"{p['instruction']}\n\n"
        f"{p['examples']}\n\n"
        f"{p['post_prompt']}\n\n"
        f"<context>\n{sample['context'].strip()}\n</context>\n\n"
        f"Question: {sample['problem']}"
    )


def extract_think_trace(text: str) -> str | None:
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def extract_predicted_answer(text: str) -> str | None:
    """Capture the model's final answer line.

    With CoT the answer is always at the END of post-think output (the model may write
    several reasoning lines after `</think>` before stating the answer). So return the
    LAST non-empty line, with an optional `Answer:` prefix and markdown emphasis stripped.
    """
    after_think = THINK_RE.sub("", text).strip()
    lines = [l.strip() for l in after_think.splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1]
    if last.lower().startswith("answer:"):
        last = last[len("answer:"):].strip()
    return last.strip("*_`")


def _preprocess_output(output: str) -> str:
    """Port of BABILong's preprocess_output (metrics.py)."""
    output = output.lower()
    output = output.split(".")[0]
    output = output.split("<context>")[0]
    output = output.split("<example>")[0]
    output = output.split("question")[0]
    return output


def compare_answers(target: str, output: str, question: str, task_labels: list[str]) -> bool:
    """Port of BABILong's compare_answers (metrics.py)."""
    output = _preprocess_output(output)
    target = target.lower()
    label_set = {label.lower() for label in task_labels}

    labels_in_question = {label for label in label_set if label in question.lower()}
    labels_in_output = {label for label in label_set if label in output}
    labels_in_output = labels_in_output - labels_in_question

    if "," in target and len(target) > 3:
        subtargets = target.split(",")
        if all(t in labels_in_output for t in subtargets) and len(labels_in_output) == len(subtargets):
            return True
    else:
        if target in labels_in_output and len(labels_in_output) == 1:
            return True
    return False


def is_answer_correct(post_think_output: str, target: str, qa_task: str, question: str) -> bool:
    if qa_task not in TASK_LABELS:
        raise ValueError(f"No task labels for {qa_task!r}. Add it to TASK_LABELS.")
    return compare_answers(target, post_think_output, question, TASK_LABELS[qa_task])


async def call_with_retry(client, messages, model, temperature, max_tokens, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)


async def process_sample(client, sample, model, temperature, max_tokens, semaphore):
    async with semaphore:
        user_message = format_prompt(sample)
        messages = [
            {"role": "system", "content": "You are careful at tracking facts across long passages."},
            {"role": "user", "content": user_message},
        ]

        try:
            response = await call_with_retry(client, messages, model, temperature, max_tokens)
            raw = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            trace = extract_think_trace(raw)
            predicted = extract_predicted_answer(raw)

            if trace is None or not predicted:
                logger.warning(f"Sample {sample['id']}: unparseable response (finish_reason={finish_reason}); "
                               f"raw[:200]={raw[:200]!r}")
                correct = False
            else:
                correct = is_answer_correct(predicted, sample["target"], sample["qa_task"], sample["problem"])

            return {
                "result": {
                    "id": sample["id"],
                    "problem": sample["problem"],
                    "think_trace": trace,
                    "expected_answer": sample["target"],
                    "predicted_answer": predicted,
                    "raw_response": raw,
                    "is_correct": correct,
                    "finish_reason": finish_reason,
                    "qa_task": sample["qa_task"],
                    "length_config": sample["length_config"],
                },
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Failed on sample {sample['id']}: {e}")
            return {"result": None, "input_tokens": 0, "output_tokens": 0, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Generate CoT traces on BABILong samples")
    parser.add_argument("--model", default="moonshotai/Kimi-K2-Thinking",
                        help="Generator model (default: moonshotai/Kimi-K2-Thinking)")
    parser.add_argument("--input", default="data/raw/babilong_samples.jsonl", help="Input samples path")
    parser.add_argument("--output", default="data/traces/babilong_traces.jsonl", help="Output traces path")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (default: 0.6)")
    parser.add_argument("--max-tokens", type=int, default=12000, help="Max output tokens (default: 12000)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max parallel requests (default: 10)")
    parser.add_argument("--rerun", action="store_true", help="Ignore existing output and re-run from scratch")
    parser.add_argument("--retry-incorrect", action="store_true",
                        help="Re-attempt previously-generated samples where is_correct is false")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples (for smoke runs)")
    args = parser.parse_args()

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        logger.error("TINKER_API_KEY not set. Add it to .env or export it.")
        return

    samples = []
    with open(args.input) as f:
        for line in f:
            samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples from {args.input}")

    if args.limit is not None:
        samples = samples[: args.limit]
        logger.info(f"Limiting to first {len(samples)} samples")

    in_scope = list(samples)

    existing_results = {}
    output_path = Path(args.output)
    if output_path.exists() and not args.rerun:
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                existing_results[r["id"]] = r

        n_in_scope = len(in_scope)
        n_correct = sum(1 for s in in_scope
                        if existing_results.get(s["id"], {}).get("is_correct"))
        n_incorrect = sum(1 for s in in_scope
                          if s["id"] in existing_results and not existing_results[s["id"]].get("is_correct"))
        n_never = n_in_scope - n_correct - n_incorrect

        samples_by_id = {s["id"]: s for s in in_scope}
        never_attempted = [s for s in in_scope if s["id"] not in existing_results]

        status_line = (f"Resuming from {output_path}: {n_in_scope} input samples in scope — "
                       f"{n_correct} correct, {n_incorrect} incorrect, {n_never} never attempted.")

        if args.retry_incorrect:
            incorrect_in_scope_ids = [s["id"] for s in in_scope
                                      if s["id"] in existing_results and not existing_results[s["id"]].get("is_correct")]
            retries = [samples_by_id[i] for i in incorrect_in_scope_ids]
            samples = never_attempted + retries
            if not samples:
                logger.info(status_line + " Nothing to do — all in-scope samples already correct.")
                return
            logger.info(status_line)
            logger.info(f"Queue: {len(never_attempted)} never-attempted + {len(retries)} retries "
                        f"of incorrect = {len(samples)} samples this run.")
        else:
            samples = never_attempted
            if not samples:
                logger.info(status_line + " Nothing to do — every in-scope sample already has a row "
                            "(pass --retry-incorrect to re-attempt incorrect ones, or --rerun to start over).")
                return
            logger.info(status_line)
            logger.info(f"Queue: {len(samples)} never-attempted samples this run.")

    client = AsyncOpenAI(base_url=TINKER_BASE_URL, api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    logger.info(f"Running with concurrency={args.concurrency} model={args.model}")

    tasks = [
        process_sample(client, s, args.model, args.temperature, args.max_tokens, semaphore)
        for s in samples
    ]
    outputs = await tqdm_asyncio.gather(*tasks, desc=f"Generating with {args.model}")

    results = []
    failures = 0
    attempts_this_run = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for out in outputs:
        total_input_tokens += out["input_tokens"]
        total_output_tokens += out["output_tokens"]
        if out["result"]:
            results.append(out["result"])
            attempts_this_run += 1
        else:
            failures += 1

    if existing_results:
        for r in results:
            existing_results[r["id"]] = r
        results = list(existing_results.values())
    results.sort(key=lambda r: r["id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    final_results = {r["id"]: r for r in results}
    by_cell_correct = {}
    by_cell_incorrect = {}
    by_cell_never = {}
    by_cell_total = {}
    n_correct = n_incorrect = n_never = 0
    for s in in_scope:
        cell = (s["qa_task"], s["length_config"])
        by_cell_total[cell] = by_cell_total.get(cell, 0) + 1
        r = final_results.get(s["id"])
        if r is None:
            by_cell_never[cell] = by_cell_never.get(cell, 0) + 1
            n_never += 1
        elif r.get("is_correct"):
            by_cell_correct[cell] = by_cell_correct.get(cell, 0) + 1
            n_correct += 1
        else:
            by_cell_incorrect[cell] = by_cell_incorrect.get(cell, 0) + 1
            n_incorrect += 1

    logger.info(f"Done. {attempts_this_run} attempts this run wrote rows; "
                f"{failures} hard-failed (no row written).")
    logger.info(f"In-scope totals: {n_correct} correct / {n_incorrect} incorrect / "
                f"{n_never} never-attempted (of {len(in_scope)})")
    for cell in sorted(by_cell_total.keys()):
        c = by_cell_correct.get(cell, 0)
        i = by_cell_incorrect.get(cell, 0)
        nv = by_cell_never.get(cell, 0)
        t = by_cell_total[cell]
        logger.info(f"  {cell[0]} @ {cell[1]}: {c}/{t} correct, {i} incorrect, {nv} never-attempted")
    logger.info(f"Tokens used this run: {total_input_tokens} input, {total_output_tokens} output")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
