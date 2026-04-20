"""Analyze [CACHE] token annotations across multiple teacher models."""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ANNOTATED_DIR = Path("data/annotated")
DEFAULT_MODELS = ["qwen3_235b", "deepseek_v31", "llama_70b"]

HEDGING_RE = re.compile(
    r"(hmm|let me think|wait[,.\s]|not sure|maybe|perhaps|I wonder|is that right)",
    re.IGNORECASE,
)
SENTENCE_END_RE = re.compile(r"[.!?\])]")


def load_annotations(model_dir: Path) -> dict[int, dict]:
    path = model_dir / "annotated_samples.jsonl"
    samples = {}
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            samples[s["id"]] = s
    return samples


def find_cache_positions(text: str) -> list[int]:
    return [m.start() for m in re.finditer(r"\[CACHE\]", text)]


def normalize_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def score_text_preservation(original: str, annotated: str) -> float:
    """Sentinel: must be 1.0 under the selection pipeline. <1.0 signals a reconstructor bug."""
    clean = normalize_whitespace(annotated.replace(" [CACHE]", ""))
    orig = normalize_whitespace(original)
    return 1.0 if clean == orig else 0.0


def score_count_appropriateness(cache_count: int, word_count: int) -> float:
    low = max(3, word_count / 150)
    high = max(8, word_count / 60)
    if low <= cache_count <= high:
        return 1.0
    if cache_count < low:
        return cache_count / low
    return max(0.0, 1.0 - (cache_count - high) / high)


def score_density(cache_count: int, word_count: int) -> float:
    if word_count == 0:
        return 0.0
    density = (cache_count / word_count) * 100
    if 0.5 <= density <= 2.0:
        return 1.0
    if density < 0.5:
        return density / 0.5
    return max(0.0, 1.0 - (density - 2.0) / 2.0)


def score_sentence_boundary(annotated: str) -> float:
    positions = find_cache_positions(annotated)
    if not positions:
        return 0.0
    at_boundary = 0
    for pos in positions:
        before = annotated[max(0, pos - 3):pos].rstrip()
        if before and SENTENCE_END_RE.search(before):
            at_boundary += 1
    return at_boundary / len(positions)


def score_hedging_proximity(annotated: str) -> float:
    positions = find_cache_positions(annotated)
    if not positions:
        return 1.0
    near_hedging = 0
    for pos in positions:
        window = annotated[max(0, pos - 100):pos]
        if HEDGING_RE.search(window):
            near_hedging += 1
    return 1.0 - near_hedging / len(positions)


def score_consensus(positions: list[int], other_models_positions: list[list[int]], threshold: int = 50) -> float:
    if not positions:
        return 0.0
    n_others = len(other_models_positions)
    if n_others == 0:
        return 0.0
    total_agreement = 0.0
    for pos in positions:
        agreeing = 0
        for other_pos in other_models_positions:
            if any(abs(pos - op) <= threshold for op in other_pos):
                agreeing += 1
        total_agreement += agreeing / n_others
    return total_agreement / len(positions)


def main():
    parser = argparse.ArgumentParser(description="Analyze [CACHE] annotations across models")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model directory names to compare")
    parser.add_argument("--output-dir", default="eval", help="Output directory (default: eval)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_annotations = {}
    for model in args.models:
        model_dir = ANNOTATED_DIR / model
        if not model_dir.exists():
            logger.warning(f"Skipping {model}: {model_dir} not found")
            continue
        all_annotations[model] = load_annotations(model_dir)
        logger.info(f"Loaded {len(all_annotations[model])} samples from {model}")

    common_ids = sorted(set.intersection(*[set(a.keys()) for a in all_annotations.values()]))
    logger.info(f"Analyzing {len(common_ids)} common samples across {len(all_annotations)} models")

    rows = []
    for sid in common_ids:
        all_positions = {m: find_cache_positions(all_annotations[m][sid]["annotated_trace"]) for m in all_annotations}

        for model in all_annotations:
            s = all_annotations[model][sid]
            original = s["original_trace"]
            annotated = s["annotated_trace"]
            wc = len(original.split())
            cache_count = s["num_cache_tokens"]

            other_positions = [all_positions[m] for m in all_annotations if m != model]

            preservation = score_text_preservation(original, annotated)
            if preservation < 1.0:
                logger.warning(f"{model} sample {sid}: text_preservation={preservation:.3f} — reconstructor may be broken")

            num_units = s.get("num_units", 0)
            selected = s.get("selected_unit_ids", [])
            final_unit_selected = num_units > 0 and (num_units - 1) in selected
            density = len(selected) / num_units if num_units else 0.0

            rows.append({
                "id": sid,
                "model": model,
                "text_preservation": preservation,
                "count_appropriateness": score_count_appropriateness(cache_count, wc),
                "density_score": score_density(cache_count, wc),
                "sentence_boundary": score_sentence_boundary(annotated),
                "hedging_penalty": score_hedging_proximity(annotated),
                "consensus": score_consensus(all_positions[model], other_positions),
                "invalid_id": 1.0 if s.get("had_invalid_ids") else 0.0,
                "selected_final_unit": 1.0 if final_unit_selected else 0.0,
                "selection_density": density,
                "cache_count": cache_count,
                "word_count": wc,
                "num_units": num_units,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "per_sample_scores.csv", index=False)
    logger.info(f"Saved per-sample scores to {output_dir}/per_sample_scores.csv")

    metrics = ["count_appropriateness", "density_score", "sentence_boundary", "hedging_penalty", "consensus", "invalid_id", "selected_final_unit", "selection_density"]
    summary_rows = []
    for model in all_annotations:
        mdf = df[df["model"] == model]
        row = {"model": model}
        for m in metrics:
            row[f"{m}_mean"] = round(mdf[m].mean(), 3)
            row[f"{m}_median"] = round(mdf[m].median(), 3)
            row[f"{m}_std"] = round(mdf[m].std(), 3)
        row["avg_cache_tokens"] = round(mdf["cache_count"].mean(), 1)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "model_summary.csv", index=False)
    logger.info(f"Saved model summary to {output_dir}/model_summary.csv")

    models = list(all_annotations.keys())
    agreement_data = []
    for m1 in models:
        row = {"model": m1}
        for m2 in models:
            if m1 == m2:
                row[m2] = 1.0
            else:
                agreements = []
                for sid in common_ids:
                    pos1 = find_cache_positions(all_annotations[m1][sid]["annotated_trace"])
                    pos2 = find_cache_positions(all_annotations[m2][sid]["annotated_trace"])
                    if not pos1 and not pos2:
                        agreements.append(1.0)
                    elif not pos1 or not pos2:
                        agreements.append(0.0)
                    else:
                        matches = sum(1 for p in pos1 if any(abs(p - q) <= 50 for q in pos2))
                        agreements.append(matches / len(pos1))
                row[m2] = round(sum(agreements) / len(agreements), 3)
        agreement_data.append(row)

    agreement_df = pd.DataFrame(agreement_data)
    agreement_df.to_csv(output_dir / "pairwise_agreement.csv", index=False)
    logger.info(f"Saved pairwise agreement to {output_dir}/pairwise_agreement.csv")

    print(f"\n{'='*90}")
    print("Model Annotation Analysis")
    print(f"{'='*90}")
    print(f"Samples: {len(common_ids)}  |  Models: {len(all_annotations)}")
    print()

    col = 13
    header = f"{'Model':<16}" + "".join(f"{m[:col]:>{col}}" for m in ["count", "density", "sent_bnd", "hedging", "consensus", "invalid_id", "final_unit", "sel_density", "avg_cache"])
    print(header)
    print("-" * len(header))

    for _, row in summary_df.iterrows():
        line = f"{row['model']:<16}"
        line += f"{row['count_appropriateness_mean']:>{col}.3f}"
        line += f"{row['density_score_mean']:>{col}.3f}"
        line += f"{row['sentence_boundary_mean']:>{col}.3f}"
        line += f"{row['hedging_penalty_mean']:>{col}.3f}"
        line += f"{row['consensus_mean']:>{col}.3f}"
        line += f"{row['invalid_id_mean']:>{col}.3f}"
        line += f"{row['selected_final_unit_mean']:>{col}.3f}"
        line += f"{row['selection_density_mean']:>{col}.3f}"
        line += f"{row['avg_cache_tokens']:>{col}}"
        print(line)

    print()

    print("Pairwise Positional Agreement:")
    col2 = 16
    print(f"{'':>{col2}}" + "".join(f"{m[:col2]:>{col2}}" for m in models))
    for _, row in agreement_df.iterrows():
        line = f"{row['model']:>{col2}}"
        for m in models:
            line += f"{row[m]:>{col2}.3f}"
        print(line)

    print()


if __name__ == "__main__":
    main()
