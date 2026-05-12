"""Pull samples from RMT-team/babilong for trace generation and [CACHE] annotation.

Pulls a balanced grid of (qa_task, length_config) cells. Each cell takes the first
`--per-cell` rows from the corresponding BABILong split, so pulls are deterministic.
"""

import argparse
import json
import logging
from itertools import product
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/raw/babilong_samples.jsonl")
DEFAULT_TASKS = "qa1,qa2,qa3"
DEFAULT_LENGTHS = "0k,1k,2k"


def main():
    parser = argparse.ArgumentParser(description="Pull samples from RMT-team/babilong")
    parser.add_argument("--per-cell", type=int, default=50,
                        help="Samples per (qa_task, length) cell (default: 50)")
    parser.add_argument("--tasks", default=DEFAULT_TASKS,
                        help="Comma-separated qa splits (default: qa1,qa2,qa3)")
    parser.add_argument("--lengths", default=DEFAULT_LENGTHS,
                        help="Comma-separated length configs (default: 0k,1k,2k)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL path")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    lengths = [l.strip() for l in args.lengths.split(",") if l.strip()]
    logger.info(f"Tasks: {tasks} | Lengths: {lengths} | Per cell: {args.per_cell}")

    selected = []
    next_id = 0
    for length, qa_task in product(lengths, tasks):
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in selected:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Saved {len(selected)} samples to {output_path}")
    by_cell = {}
    for e in selected:
        by_cell[(e["qa_task"], e["length_config"])] = by_cell.get((e["qa_task"], e["length_config"]), 0) + 1
    for cell, count in sorted(by_cell.items()):
        logger.info(f"  {cell[0]} @ {cell[1]}: {count}")


if __name__ == "__main__":
    main()
