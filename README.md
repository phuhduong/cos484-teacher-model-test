# cos484-teacher-model-test

Data annotation pipeline for "Memory Is All You Need: Learned State Caching for Chain-of-Thought Reasoning for Mamba" (COS484). Generates CoT traces over BABILong samples (`RMT-team/babilong`) with a thinking model, then uses teacher models to insert `[CACHE]` tokens after reasoning units worth caching for later recall.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Tinker API key:

```
TINKER_API_KEY=your_key_here
```

## Pipeline

### Step 1: Pull BABILong samples

Pulls a balanced grid of (qa_task, length_config) cells from `RMT-team/babilong`. Default grid is `{qa1, qa2, qa3} × {0k, 1k, 2k}` with 50 samples per cell = 450 total.

```bash
python scripts/pull_samples.py                                  # defaults: 450 samples
python scripts/pull_samples.py --per-cell 2                     # 18-sample smoke test
python scripts/pull_samples.py --tasks qa1,qa2 --lengths 0k,1k  # custom grid
```

Output: `data/raw/babilong_samples.jsonl` with `{id, problem, context, target, qa_task, length_config}`.

### Step 2: Generate CoT traces

Runs a thinking model over the pulled samples to produce `<think>…</think>` reasoning traces, extracts the model's final answer (the last non-empty line after `</think>`, with an optional `Answer:` prefix stripped), and tags each trace with `is_correct` using BABILong's label-match check. Resumes from existing output if interrupted.

```bash
python scripts/generate_traces.py --model "moonshotai/Kimi-K2-Thinking"
# Re-run the same command to pick up samples that hard-failed (no <think> block, API error).
# --retry-incorrect to re-attempt previously-generated samples whose final answer was wrong.
# --rerun to start fresh, --concurrency N to adjust parallelism, --limit N for smoke runs.
```

Output: `data/traces/babilong_traces.jsonl` with `{id, problem, think_trace, expected_answer, predicted_answer, raw_response, is_correct, finish_reason, qa_task, length_config}`.

### Step 3: Run teacher models

For each trace, `run_teacher.py` segments the trace into numbered reasoning units, asks the teacher to return a JSON list of unit IDs after which `[CACHE]` should be inserted, then reconstructs the annotated trace deterministically in code. The teacher never re-emits the trace, so preservation is exact by construction. The last segmented unit is filtered out before reconstruction, so `[CACHE]` is never inserted after the final answer. By default, only traces with `is_correct == true` are annotated — pass `--no-require-correct` to annotate all. Runs 10 concurrent requests by default and resumes from where it left off if interrupted.

```bash
python scripts/run_teacher.py --model "Qwen/Qwen3-235B-A22B-Instruct-2507" --output-dir data/annotated/qwen3_235b
python scripts/run_teacher.py --model "deepseek-ai/DeepSeek-V3.1" --output-dir data/annotated/deepseek_v31

# --rerun to start fresh, --no-require-correct to include incorrect traces, --limit N for smoke runs
```

Output: `{output-dir}/annotated_samples.jsonl` and `{output-dir}/summary.json`. Each row includes `{id, problem, original_trace, annotated_trace, num_cache_tokens, model, expected_answer, selected_unit_ids, num_units, had_invalid_ids}`.

### Step 4: Analyze annotations

Evaluates annotation quality across models. Key metrics: cache count appropriateness, density, sentence boundary placement, hedging proximity, cross-model consensus, invalid-ID rate, and selection density. Text preservation and final-unit selection are invariants (always 1.0 and 0.0 respectively) logged as sentinels — non-zero deviations indicate a pipeline bug.

```bash
python scripts/analyze.py
```

Output: `eval/per_sample_scores.csv`, `eval/model_summary.csv`, `eval/pairwise_agreement.csv`
