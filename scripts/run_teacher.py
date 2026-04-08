"""Run a teacher model to insert [CACHE] tokens into reasoning traces."""

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
DEFAULT_PROMPT_PATH = Path("prompts/cache_insertion_prompt.txt")
THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tokens(text: str) -> str:
    """Remove closed <think>...</think> blocks from model output."""
    return THINK_CLOSED_RE.sub("", text).strip()


async def call_with_retry(client, messages, model, temperature, max_tokens, max_retries=3):
    """Call chat completions with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)


async def process_sample(client, sample, prompt_template, model, temperature, max_tokens, semaphore):
    """Process a single sample with concurrency control."""
    async with semaphore:
        user_message = prompt_template.replace("{problem}", sample["problem"])
        user_message = user_message.replace("{trace}", sample["think_trace"])

        messages = [
            {"role": "system", "content": "You are an expert at analyzing chain-of-thought reasoning."},
            {"role": "user", "content": user_message},
        ]

        try:
            response = await call_with_retry(client, messages, model, temperature, max_tokens)

            raw_output = response.choices[0].message.content or ""

            # Reject unclosed <think> blocks
            if "<think>" in raw_output and "</think>" not in raw_output:
                logger.warning(f"Sample {sample['id']}: unclosed <think> block, marking as failed")
                return {"result": None, "input_tokens": 0, "output_tokens": 0, "error": "unclosed <think> block"}

            annotated_trace = strip_think_tokens(raw_output)
            # Strip preamble the model echoes from the prompt template
            for preamble in [
                "**Annotated trace with [CACHE] tokens:**",
                "Here is the annotated trace with [CACHE] tokens:",
            ]:
                if annotated_trace.startswith(preamble):
                    annotated_trace = annotated_trace[len(preamble):].strip()
                    break
            num_cache = annotated_trace.count("[CACHE]")

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            return {
                "result": {
                    "id": sample["id"],
                    "problem": sample["problem"],
                    "original_trace": sample["think_trace"],
                    "annotated_trace": annotated_trace,
                    "num_cache_tokens": num_cache,
                    "model": model,
                    "expected_answer": sample["expected_answer"],
                },
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Failed on sample {sample['id']}: {e}")
            return {"result": None, "input_tokens": 0, "output_tokens": 0, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Run teacher model for [CACHE] token insertion")
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen3-235B-A22B-Instruct-2507)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature (default: 0)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens (default: 8192)")
    parser.add_argument("--input", default="data/raw/pilot_samples.jsonl", help="Input samples path")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT_PATH), help="Prompt template path (default: prompts/cache_insertion_prompt.txt)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max parallel requests (default: 10)")
    parser.add_argument("--rerun", action="store_true", help="Ignore existing output and re-run all samples from scratch")
    args = parser.parse_args()

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        logger.error("TINKER_API_KEY not set. Add it to .env or export it.")
        return

    # Load prompt template
    prompt_template = Path(args.prompt).read_text()
    logger.info(f"Loaded prompt template ({len(prompt_template)} chars)")

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples from {args.input}")

    # Resume: skip already-completed samples
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

    # Setup async client
    client = AsyncOpenAI(base_url=TINKER_BASE_URL, api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    logger.info(f"Running with concurrency={args.concurrency}")

    # Process all samples concurrently
    tasks = [
        process_sample(client, s, prompt_template, args.model, args.temperature, args.max_tokens, semaphore)
        for s in samples
    ]
    outputs = await tqdm_asyncio.gather(*tasks, desc=f"Annotating with {args.model}")

    # Collect results
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

    # Accumulate token counts from previous runs
    summary_path = Path(args.output_dir) / "summary.json"
    if summary_path.exists() and not args.rerun:
        prev_summary = json.loads(summary_path.read_text())
        total_input_tokens += prev_summary.get("total_input_tokens", 0)
        total_output_tokens += prev_summary.get("total_output_tokens", 0)

    # Merge with existing results if resuming
    if existing_results:
        for r in results:
            existing_results[r["id"]] = r
        results = list(existing_results.values())

    # Sort by id for consistent output
    results.sort(key=lambda r: r["id"])

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotated_samples.jsonl"

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    cache_counts = [r["num_cache_tokens"] for r in results]

    summary = {
        "model": args.model,
        "total_samples": len(results) + failures,
        "successful_annotations": len(results),
        "failed_annotations": failures,
        "avg_cache_tokens": sum(cache_counts) / len(cache_counts) if cache_counts else 0,
        "min_cache_tokens": min(cache_counts) if cache_counts else 0,
        "max_cache_tokens": max(cache_counts) if cache_counts else 0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done. {len(results)} successful, {failures} failed.")
    logger.info(f"Avg [CACHE] tokens: {summary['avg_cache_tokens']:.1f} (range: {summary['min_cache_tokens']}-{summary['max_cache_tokens']})")
    logger.info(f"Tokens used: {total_input_tokens} input, {total_output_tokens} output")
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
