"""Run a teacher model to select [CACHE] insertion points on segmented reasoning traces."""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from segment import segment, reconstruct, format_units

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
DEFAULT_PROMPT_PATH = Path("prompts/cache_selection_prompt.txt")
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_cache_ids(raw: str) -> list[int] | None:
    """Parse the teacher's JSON response. Returns list of ints or None on failure."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = JSON_OBJECT_RE.search(raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    ids = obj.get("cache_unit_ids")
    if not isinstance(ids, list):
        return None
    out: list[int] = []
    for x in ids:
        if isinstance(x, bool) or not isinstance(x, int):
            continue
        out.append(x)
    return out


async def call_with_retry(client, messages, model, temperature, max_tokens, max_retries=3):
    """Call chat completions with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)


async def process_sample(client, sample, prompt_template, model, temperature, max_tokens, semaphore):
    """Segment, query teacher, validate, reconstruct."""
    async with semaphore:
        trace = sample["think_trace"]
        units = segment(trace)
        if not units:
            logger.warning(f"Sample {sample['id']}: segmenter produced zero units")
            return {"result": None, "input_tokens": 0, "output_tokens": 0, "error": "empty segmentation"}

        user_message = prompt_template.replace("{problem}", sample["problem"])
        user_message = user_message.replace("{units}", format_units(trace, units))

        messages = [
            {"role": "system", "content": "You are an expert at analyzing chain-of-thought reasoning."},
            {"role": "user", "content": user_message},
        ]

        try:
            response = await call_with_retry(client, messages, model, temperature, max_tokens)
            raw_output = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            raw_ids = parse_cache_ids(raw_output)
            if raw_ids is None:
                logger.warning(f"Sample {sample['id']}: could not parse JSON response: {raw_output[:200]}")
                return {"result": None, "input_tokens": input_tokens, "output_tokens": output_tokens, "error": "json parse failure"}

            n_units = len(units)
            # Drop the final unit — [CACHE] after the final answer adds no training signal.
            valid_ids = sorted({i for i in raw_ids if 0 <= i < n_units - 1})
            had_invalid = any(i < 0 or i >= n_units for i in raw_ids)

            annotated_trace = reconstruct(trace, units, valid_ids)

            return {
                "result": {
                    "id": sample["id"],
                    "problem": sample["problem"],
                    "original_trace": trace,
                    "annotated_trace": annotated_trace,
                    "num_cache_tokens": len(valid_ids),
                    "model": model,
                    "expected_answer": sample["expected_answer"],
                    "selected_unit_ids": valid_ids,
                    "num_units": n_units,
                    "had_invalid_ids": had_invalid,
                },
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Failed on sample {sample['id']}: {e}")
            return {"result": None, "input_tokens": 0, "output_tokens": 0, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Run teacher model for [CACHE] unit-ID selection")
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen3-235B-A22B-Instruct-2507)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature (default: 0)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens (default: 512)")
    parser.add_argument("--input", default="data/traces/babilong_traces.jsonl", help="Input samples path")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT_PATH), help="Prompt template path")
    parser.add_argument("--concurrency", type=int, default=10, help="Max parallel requests (default: 10)")
    parser.add_argument("--rerun", action="store_true", help="Ignore existing output and re-run from scratch")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples (for smoke runs)")
    parser.add_argument("--require-correct", action=argparse.BooleanOptionalAction, default=True,
                        help="Only annotate samples with is_correct == true (default: true)")
    args = parser.parse_args()

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        logger.error("TINKER_API_KEY not set. Add it to .env or export it.")
        return

    prompt_template = Path(args.prompt).read_text()
    logger.info(f"Loaded prompt template ({len(prompt_template)} chars) from {args.prompt}")

    samples = []
    with open(args.input) as f:
        for line in f:
            samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples from {args.input}")

    if args.require_correct:
        before = len(samples)
        samples = [s for s in samples if s.get("is_correct") is True]
        logger.info(f"Filtered to is_correct==True: {len(samples)}/{before} samples")

    if args.limit is not None:
        samples = samples[: args.limit]
        logger.info(f"Limiting to first {len(samples)} samples")

    existing_results = {}
    output_path = Path(args.output_dir) / "annotated_samples.jsonl"
    if output_path.exists() and not args.rerun:
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                existing_results[r["id"]] = r
        samples = [s for s in samples if s["id"] not in existing_results]
        if not samples:
            logger.info("All samples already completed. Use --rerun to start fresh.")
            return
        logger.info(f"Resuming: {len(existing_results)} done, {len(samples)} remaining")

    client = AsyncOpenAI(base_url=TINKER_BASE_URL, api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    logger.info(f"Running with concurrency={args.concurrency}")

    tasks = [
        process_sample(client, s, prompt_template, args.model, args.temperature, args.max_tokens, semaphore)
        for s in samples
    ]
    outputs = await tqdm_asyncio.gather(*tasks, desc=f"Annotating with {args.model}")

    results = []
    failures = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for out in outputs:
        total_input_tokens += out["input_tokens"]
        total_output_tokens += out["output_tokens"]
        if out["result"]:
            results.append(out["result"])
        else:
            failures += 1

    summary_path = Path(args.output_dir) / "summary.json"
    if summary_path.exists() and not args.rerun:
        prev_summary = json.loads(summary_path.read_text())
        total_input_tokens += prev_summary.get("total_input_tokens", 0)
        total_output_tokens += prev_summary.get("total_output_tokens", 0)

    if existing_results:
        for r in results:
            existing_results[r["id"]] = r
        results = list(existing_results.values())
    results.sort(key=lambda r: r["id"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    cache_counts = [r["num_cache_tokens"] for r in results]
    invalid_samples = sum(1 for r in results if r.get("had_invalid_ids"))
    final_unit_samples = sum(1 for r in results if (r.get("num_units", 0) - 1) in r.get("selected_unit_ids", []))

    summary = {
        "model": args.model,
        "total_samples": len(results) + failures,
        "successful_annotations": len(results),
        "failed_annotations": failures,
        "avg_cache_tokens": sum(cache_counts) / len(cache_counts) if cache_counts else 0,
        "min_cache_tokens": min(cache_counts) if cache_counts else 0,
        "max_cache_tokens": max(cache_counts) if cache_counts else 0,
        "invalid_id_rate": invalid_samples / len(results) if results else 0,
        "selected_final_unit_rate": final_unit_samples / len(results) if results else 0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done. {len(results)} successful, {failures} failed.")
    logger.info(f"Avg [CACHE] tokens: {summary['avg_cache_tokens']:.1f} (range: {summary['min_cache_tokens']}-{summary['max_cache_tokens']})")
    logger.info(f"Invalid-ID rate: {summary['invalid_id_rate']:.1%} | Final-unit selection rate: {summary['selected_final_unit_rate']:.1%}")
    logger.info(f"Tokens used: {total_input_tokens} input, {total_output_tokens} output")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
