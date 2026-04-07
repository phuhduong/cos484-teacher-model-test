"""Pull and filter samples from nvidia/OpenMathReasoning for [CACHE] token annotation."""

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/raw/pilot_samples.jsonl")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_think_trace(generated_solution: str) -> str | None:
    """Extract text between <think> and </think> tags."""
    match = THINK_RE.search(generated_solution)
    if match:
        return match.group(1).strip()
    return None


def main():
    logger.info("Loading nvidia/OpenMathReasoning (cot split, streaming)...")
    ds = load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True)

    # Collect candidates into difficulty buckets
    # easy: pass_rate > 0.6, medium: 0.3-0.6, hard: < 0.3
    buckets: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    target_per_bucket = 200  # oversample, then pick 33-34 from each

    seen_sources: dict[str, set[str]] = defaultdict(set)
    total_scanned = 0

    for sample in ds:
        total_scanned += 1

        if total_scanned % 10000 == 0:
            counts = {k: len(v) for k, v in buckets.items()}
            logger.info(f"Scanned {total_scanned} samples, bucket sizes: {counts}")

        # Check all buckets full
        if all(len(v) >= target_per_bucket for v in buckets.values()):
            logger.info(f"All buckets full after scanning {total_scanned} samples.")
            break

        generated = sample.get("generated_solution", "")
        if not generated or "<think>" not in generated:
            continue

        trace = extract_think_trace(generated)
        if trace is None:
            continue

        wc = len(trace.split())
        if wc < 200 or wc > 1500:
            continue

        pass_rate_raw = sample.get("pass_rate_72b_tir")
        if pass_rate_raw is None:
            continue
        try:
            pass_rate = float(pass_rate_raw)
        except (ValueError, TypeError):
            continue

        # Determine bucket
        if pass_rate > 0.6:
            bucket = "easy"
        elif pass_rate >= 0.3:
            bucket = "medium"
        else:
            bucket = "hard"

        if len(buckets[bucket]) >= target_per_bucket:
            continue

        source = sample.get("problem_source", "unknown")

        entry = {
            "problem": sample["problem"],
            "think_trace": trace,
            "expected_answer": sample.get("expected_answer", ""),
            "problem_source": source,
            "pass_rate_72b_tir": pass_rate,
            "word_count": wc,
        }

        buckets[bucket].append(entry)
        seen_sources[bucket].add(source)

    # Log what we found
    for bname, entries in buckets.items():
        sources = seen_sources[bname]
        logger.info(f"Bucket '{bname}': {len(entries)} candidates from {len(sources)} sources: {sources}")

    # Stratified sampling: ~33-34 from each bucket for 100 total
    selected = []
    per_bucket = [34, 33, 33]  # easy, medium, hard — totals 100
    for (bname, entries), n in zip(buckets.items(), per_bucket):
        if len(entries) < n:
            logger.warning(f"Bucket '{bname}' has only {len(entries)} samples (wanted {n})")
            n = len(entries)

        # Favor source diversity: group by source, round-robin
        by_source = defaultdict(list)
        for e in entries:
            by_source[e["problem_source"]].append(e)

        # Shuffle within each source
        for src_list in by_source.values():
            random.shuffle(src_list)

        # Round-robin pick
        picked = []
        source_iters = {s: iter(lst) for s, lst in by_source.items()}
        source_keys = list(source_iters.keys())
        random.shuffle(source_keys)
        idx = 0
        while len(picked) < n and source_iters:
            src = source_keys[idx % len(source_keys)]
            try:
                picked.append(next(source_iters[src]))
            except StopIteration:
                source_iters.pop(src)
                source_keys.remove(src)
                if not source_keys:
                    break
                idx = idx % len(source_keys)
                continue
            idx += 1

        selected.extend(picked)

    # Assign sequential IDs and shuffle
    random.shuffle(selected)
    for i, entry in enumerate(selected):
        entry["id"] = i

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for entry in selected:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Saved {len(selected)} samples to {OUTPUT_PATH}")

    # Print summary stats
    wcs = [e["word_count"] for e in selected]
    prs = [e["pass_rate_72b_tir"] for e in selected]
    sources = set(e["problem_source"] for e in selected)
    logger.info(f"Word count: min={min(wcs)}, max={max(wcs)}, mean={sum(wcs)/len(wcs):.0f}")
    logger.info(f"Pass rate: min={min(prs):.3f}, max={max(prs):.3f}, mean={sum(prs)/len(prs):.3f}")
    logger.info(f"Sources ({len(sources)}): {sources}")


if __name__ == "__main__":
    random.seed(42)
    main()
