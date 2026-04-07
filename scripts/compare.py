"""Compare [CACHE] token annotations from two teacher models."""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ANSI colors
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def find_cache_positions(text: str) -> list[int]:
    """Find character positions of all [CACHE] tokens in text."""
    return [m.start() for m in re.finditer(r"\[CACHE\]", text)]


def positional_agreement(positions_a: list[int], positions_b: list[int], threshold: int = 50) -> float:
    """Compute % of positions in A that have a match within `threshold` chars in B."""
    if not positions_a or not positions_b:
        if not positions_a and not positions_b:
            return 1.0
        return 0.0

    matches = 0
    for pos_a in positions_a:
        if any(abs(pos_a - pos_b) <= threshold for pos_b in positions_b):
            matches += 1
    return matches / len(positions_a)


def cache_density(text: str) -> float:
    """[CACHE] tokens per 100 words."""
    num_cache = text.count("[CACHE]")
    # Count words excluding [CACHE] tokens
    clean = text.replace("[CACHE]", "")
    wc = len(clean.split())
    if wc == 0:
        return 0.0
    return (num_cache / wc) * 100


def highlight_cache(text: str, color: str) -> str:
    """Replace [CACHE] with colored version for terminal display."""
    return text.replace("[CACHE]", f"{color}{BOLD}[CACHE]{RESET}")


def load_annotations(dir_path: Path) -> dict[int, dict]:
    """Load annotated samples keyed by id."""
    path = dir_path / "annotated_samples.jsonl"
    samples = {}
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            samples[s["id"]] = s
    return samples


def main():
    parser = argparse.ArgumentParser(description="Compare [CACHE] annotations from two models")
    parser.add_argument("--dir1", required=True, help="First model output directory")
    parser.add_argument("--dir2", required=True, help="Second model output directory")
    parser.add_argument("--output", required=True, help="Output directory for comparison results")
    parser.add_argument("--interactive", action="store_true", help="Interactive rating mode")
    args = parser.parse_args()

    dir1, dir2 = Path(args.dir1), Path(args.dir2)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    ann1 = load_annotations(dir1)
    ann2 = load_annotations(dir2)

    # Load model names from summary
    model1 = json.loads((dir1 / "summary.json").read_text()).get("model", "Model 1")
    model2 = json.loads((dir2 / "summary.json").read_text()).get("model", "Model 2")

    # Find common IDs
    common_ids = sorted(set(ann1.keys()) & set(ann2.keys()))
    logger.info(f"Comparing {len(common_ids)} common samples between {model1} and {model2}")

    # Compute metrics
    rows = []
    for sid in common_ids:
        s1, s2 = ann1[sid], ann2[sid]
        t1, t2 = s1["annotated_trace"], s2["annotated_trace"]
        pos1, pos2 = find_cache_positions(t1), find_cache_positions(t2)

        agreement_1to2 = positional_agreement(pos1, pos2)
        agreement_2to1 = positional_agreement(pos2, pos1)
        avg_agreement = (agreement_1to2 + agreement_2to1) / 2

        rows.append({
            "id": sid,
            "problem_preview": s1["problem"][:80],
            f"cache_count_{model1.split('/')[-1]}": len(pos1),
            f"cache_count_{model2.split('/')[-1]}": len(pos2),
            f"density_{model1.split('/')[-1]}": round(cache_density(t1), 2),
            f"density_{model2.split('/')[-1]}": round(cache_density(t2), 2),
            "positional_agreement": round(avg_agreement, 3),
        })

    df = pd.DataFrame(rows)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Comparison: {model1} vs {model2}")
    print(f"{'='*70}")
    print(f"Samples compared: {len(common_ids)}")
    print()

    m1_short = model1.split("/")[-1]
    m2_short = model2.split("/")[-1]

    cc1 = df[f"cache_count_{m1_short}"]
    cc2 = df[f"cache_count_{m2_short}"]
    d1 = df[f"density_{m1_short}"]
    d2 = df[f"density_{m2_short}"]

    col = max(len(m1_short), len(m2_short), 12) + 2
    width = 30 + col * 2
    print(f"{'Metric':<30} {m1_short:>{col}} {m2_short:>{col}}")
    print("-" * width)
    print(f"{'Avg [CACHE] tokens':<30} {cc1.mean():>{col}.1f} {cc2.mean():>{col}.1f}")
    print(f"{'Median [CACHE] tokens':<30} {cc1.median():>{col}.1f} {cc2.median():>{col}.1f}")
    print(f"{'Min [CACHE] tokens':<30} {cc1.min():>{col}} {cc2.min():>{col}}")
    print(f"{'Max [CACHE] tokens':<30} {cc1.max():>{col}} {cc2.max():>{col}}")
    print(f"{'Avg density (per 100 words)':<30} {d1.mean():>{col}.2f} {d2.mean():>{col}.2f}")
    print(f"{'Avg positional agreement':<30} {df['positional_agreement'].mean():>{col}.3f}")
    print()

    # Save comparison CSV
    csv_path = output_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison to {csv_path}")

    # Interactive mode
    if args.interactive:
        print(f"\n{BOLD}Pairwise Comparison Mode{RESET}")
        print(f"For each sample, choose which model's annotation is better.")
        print(f"  {YELLOW}A{RESET} = {m1_short}    {GREEN}B{RESET} = {m2_short}    T = tie\n")

        ratings = []
        for i, sid in enumerate(common_ids):
            s1, s2 = ann1[sid], ann2[sid]

            print(f"\n{'='*70}")
            print(f"{BOLD}Sample {i+1}/{len(common_ids)} (id={sid}){RESET}")
            print(f"{BOLD}Problem:{RESET} {s1['problem'][:200]}")
            print(f"\n{BOLD}--- Original Trace ---{RESET}")
            print(s1["original_trace"][:500] + ("..." if len(s1["original_trace"]) > 500 else ""))

            print(f"\n{BOLD}--- {YELLOW}A{RESET} {BOLD}({s1['num_cache_tokens']} [CACHE] tokens) ---{RESET}")
            print(highlight_cache(s1["annotated_trace"], YELLOW))

            print(f"\n{BOLD}--- {GREEN}B{RESET} {BOLD}({s2['num_cache_tokens']} [CACHE] tokens) ---{RESET}")
            print(highlight_cache(s2["annotated_trace"], GREEN))

            while True:
                try:
                    choice = input(f"\n{CYAN}Which is better? (A/B/T): {RESET}").strip().upper()
                    if choice in ("A", "B", "T"):
                        break
                    print("Please enter A, B, or T.")
                except EOFError:
                    choice = "T"
                    break

            ratings.append({"id": sid, "preference": choice})

            # Save incrementally
            ratings_path = output_dir / "ratings.csv"
            pd.DataFrame(ratings).to_csv(ratings_path, index=False)

            # Allow early exit
            if i < len(common_ids) - 1:
                try:
                    cont = input(f"{CYAN}Continue? (y/n, default y): {RESET}").strip().lower()
                    if cont == "n":
                        print("Stopping early. Ratings saved.")
                        break
                except EOFError:
                    break

        ratings_path = output_dir / "ratings.csv"
        pd.DataFrame(ratings).to_csv(ratings_path, index=False)
        logger.info(f"Saved {len(ratings)} ratings to {ratings_path}")

        if ratings:
            a_wins = sum(1 for r in ratings if r["preference"] == "A")
            b_wins = sum(1 for r in ratings if r["preference"] == "B")
            ties = sum(1 for r in ratings if r["preference"] == "T")
            print(f"\n{BOLD}Results{RESET}")
            print(f"  {m1_short} (A): {a_wins} wins")
            print(f"  {m2_short} (B): {b_wins} wins")
            print(f"  Ties: {ties}")


if __name__ == "__main__":
    main()
