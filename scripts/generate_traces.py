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
from collections import Counter
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker import types
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/raw/babilong_samples.jsonl")
OUTPUT_PATH = Path("data/traces/babilong_traces.jsonl")
TEMPERATURE = 0.6
MAX_TOKENS = 12000
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

def format_prompt(sample):
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


def extract_think_trace(text):
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def extract_predicted_answer(text):
    """Capture the model's final answer line. Returns None if no non-empty line follows </think>.

    With CoT the answer is always at the END of post-think output (the model may write
    several reasoning lines after `</think>` before stating the answer). So return the
    LAST non-empty line, with an optional `Answer:` prefix and surrounding markdown
    emphasis chars (`*`, `_`, `` ` ``) stripped.
    """
    after_think = THINK_RE.sub("", text).strip()
    lines = [l.strip() for l in after_think.splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1]
    if last.lower().startswith("answer:"):
        last = last[len("answer:"):].strip()
    return last.strip("*_`")


def _preprocess_output(output):
    """Port of BABILong's preprocess_output (metrics.py)."""
    output = output.lower()
    output = output.split(".")[0]
    output = output.split("<context>")[0]
    output = output.split("<example>")[0]
    output = output.split("question")[0]
    return output


def compare_answers(target, output, question, task_labels):
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


def is_answer_correct(post_think_output, target, qa_task, question):
    if qa_task not in TASK_LABELS:
        raise ValueError(f"No task labels for {qa_task!r}. Add it to TASK_LABELS.")
    return compare_answers(target, post_think_output, question, TASK_LABELS[qa_task])


async def call_with_retry(sampling_client, prompt, sampling_params, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await sampling_client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params,
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)


async def process_sample(sampling_client, tokenizer, sampling_params, sample, semaphore):
    async with semaphore:
        messages = [
            {"role": "system", "content": "You are careful at tracking facts across long passages."},
            {"role": "user", "content": format_prompt(sample)},
        ]

        try:
            input_token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            prompt = types.ModelInput.from_ints(input_token_ids)
            response = await call_with_retry(sampling_client, prompt, sampling_params)
            output_seq = response.sequences[0]
            output_token_ids = list(output_seq.tokens)
            raw = tokenizer.decode(output_token_ids, skip_special_tokens=True)
            finish_reason = str(getattr(output_seq, "stop_reason", "unknown"))
            input_tokens = len(input_token_ids)
            output_tokens = len(output_token_ids)

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
            }
        except Exception as e:
            logger.error(f"Failed on sample {sample['id']}: {e}")
            return {"result": None, "input_tokens": 0, "output_tokens": 0}


async def main():
    parser = argparse.ArgumentParser(description="Generate CoT traces on BABILong samples")
    parser.add_argument("--model", default="moonshotai/Kimi-K2-Thinking",
                        help="Generator model (default: moonshotai/Kimi-K2-Thinking)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max parallel requests (default: 10)")
    parser.add_argument("--rerun", action="store_true",
                        help="Ignore existing output and re-run from scratch (including is_correct==True rows)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples (for smoke runs)")
    args = parser.parse_args()

    if not os.environ.get("TINKER_API_KEY"):
        logger.error("TINKER_API_KEY not set. Add it to .env or export it.")
        return

    samples = []
    with open(INPUT_PATH) as f:
        for line in f:
            samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples from {INPUT_PATH}")

    if args.limit is not None:
        samples = samples[: args.limit]
        logger.info(f"Limiting to first {len(samples)} samples")

    in_scope = list(samples)

    existing_results = {}
    if OUTPUT_PATH.exists() and not args.rerun:
        with open(OUTPUT_PATH) as f:
            for line in f:
                r = json.loads(line)
                existing_results[r["id"]] = r

        never_attempted = [s for s in in_scope if s["id"] not in existing_results]
        incorrect = [s for s in in_scope
                     if s["id"] in existing_results and not existing_results[s["id"]].get("is_correct")]
        n_correct = len(in_scope) - len(never_attempted) - len(incorrect)

        logger.info(f"Resuming from {OUTPUT_PATH}: {len(in_scope)} input samples in scope — "
                    f"{n_correct} correct, {len(incorrect)} incorrect, {len(never_attempted)} never attempted.")

        samples = never_attempted + incorrect
        if not samples:
            logger.info("Nothing to do — all in-scope samples already correct.")
            return
        logger.info(f"Queue: {len(never_attempted)} never-attempted + "
                    f"{len(incorrect)} incorrect = {len(samples)} samples this run.")

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(base_model=args.model)
    tokenizer = sampling_client.get_tokenizer()
    sampling_params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

    semaphore = asyncio.Semaphore(args.concurrency)
    logger.info(f"Running with concurrency={args.concurrency} model={args.model}")

    tasks = [process_sample(sampling_client, tokenizer, sampling_params, s, semaphore) for s in samples]
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    final_results = {r["id"]: r for r in results}

    def status(s):
        r = final_results.get(s["id"])
        if r is None:
            return "never"
        return "correct" if r.get("is_correct") else "incorrect"

    by_status = Counter(status(s) for s in in_scope)
    by_cell = Counter((s["qa_task"], s["length_config"], status(s)) for s in in_scope)
    cells = sorted({(t, l) for t, l, _ in by_cell})

    logger.info(f"Done. {attempts_this_run} attempts this run wrote rows; "
                f"{failures} hard-failed (no row written).")
    logger.info(f"In-scope totals: {by_status['correct']} correct / {by_status['incorrect']} incorrect / "
                f"{by_status['never']} never-attempted (of {len(in_scope)})")
    for task, length in cells:
        c, i, nv = by_cell[task, length, "correct"], by_cell[task, length, "incorrect"], by_cell[task, length, "never"]
        logger.info(f"  {task} @ {length}: {c}/{c + i + nv} correct, {i} incorrect, {nv} never-attempted")
    logger.info(f"Tokens used this run: {total_input_tokens} input, {total_output_tokens} output")
    logger.info(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
