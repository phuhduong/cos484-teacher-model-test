"""Pull samples from RMT-team/babilong for trace generation and [CACHE] annotation.

Pulls a balanced grid of (qa_task, length_config) cells. Each cell takes the first
`--per-cell` rows from the corresponding BABILong split, so pulls are deterministic.
"""

import argparse
import json
import logging
from collections import Counter
from itertools import product
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/raw/babilong_samples.jsonl")
TASKS = ["qa1", "qa2", "qa3"]
LENGTHS = ["0k", "1k", "2k"]


def main():
    parser = argparse.ArgumentParser(description="Pull samples from RMT-team/babilong")
    parser.add_argument("--per-cell", type=int, default=50,
                        help="Samples per (qa_task, length) cell (default: 50)")
    args = parser.parse_args()

    logger.info(f"Tasks: {TASKS} | Lengths: {LENGTHS} | Per cell: {args.per_cell}")

    selected = []
    next_id = 0
    for length, qa_task in product(LENGTHS, TASKS):
        logger.info(f"Loading RMT-team/babilong name={length} split={qa_task}...")
        ds = load_dataset("RMT-team/babilong", name=length, split=qa_task)
        n = min(args.per_cell, len(ds))
        if n < args.per_cell:
            logger.warning(f"Cell ({qa_task}, {length}) has only {len(ds)} samples (wanted {args.per_cell})")
        for row in ds.select(range(n)):
            selected.append({
                "id": next_id,
                "problem": row["question"],
                "context": row["input"],
                "target": row["target"],
                "qa_task": qa_task,
                "length_config": length,
            })
            next_id += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for entry in selected:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Saved {len(selected)} samples to {OUTPUT_PATH}")
    by_cell = Counter((e["qa_task"], e["length_config"]) for e in selected)
    for (task, length), count in sorted(by_cell.items()):
        logger.info(f"  {task} @ {length}: {count}")


if __name__ == "__main__":
    main()
