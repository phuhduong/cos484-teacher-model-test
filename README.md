# cos484-teacher-model-test

Data annotation pipeline for "Memory Is All You Need: Learned State Caching for Chain-of-Thought Reasoning for Mamba" (COS484). Uses teacher models to insert `[CACHE]` tokens into math chain-of-thought reasoning traces from `nvidia/OpenMathReasoning`.

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

### Step 1: Pull samples

Streams the OpenMathReasoning dataset and samples traces stratified by difficulty (easy/medium/hard).

```bash
python scripts/pull_samples.py              # 100 samples (default)
python scripts/pull_samples.py --count 5000  # 5,000 samples
```

Output: `data/raw/pilot_samples.jsonl`

### Step 2: Run teacher models

For each trace, `run_teacher.py` segments the trace into numbered reasoning units, asks the teacher to return a JSON list of unit IDs after which `[CACHE]` should be inserted, then reconstructs the annotated trace deterministically in code. The teacher never re-emits the trace, so preservation is exact by construction. The last segmented unit is filtered out before reconstruction, so `[CACHE]` is never inserted after the final answer. Runs 10 concurrent requests by default and resumes from where it left off if interrupted.

```bash
python scripts/run_teacher.py --model "Qwen/Qwen3-235B-A22B-Instruct-2507" --output-dir data/annotated/qwen3_235b
python scripts/run_teacher.py --model "deepseek-ai/DeepSeek-V3.1" --output-dir data/annotated/deepseek_v31
python scripts/run_teacher.py --model "meta-llama/Llama-3.3-70B-Instruct" --output-dir data/annotated/llama_70b

# Re-run the same command to resume after interruption or failures
# Use --rerun to start fresh, --concurrency N to adjust parallelism, --limit N for smoke runs
```

Output: `{output-dir}/annotated_samples.jsonl` and `{output-dir}/summary.json`. Each row includes `annotated_trace`, `selected_unit_ids`, and `num_units`.

### Step 3: Analyze annotations

Evaluates annotation quality across models. Key metrics: cache count appropriateness, density, sentence boundary placement, hedging proximity, cross-model consensus, invalid-ID rate, and selection density. Text preservation and final-unit selection are invariants (always 1.0 and 0.0 respectively) logged as sentinels — non-zero deviations indicate a pipeline bug.

```bash
python scripts/analyze.py
```

Output: `eval/per_sample_scores.csv`, `eval/model_summary.csv`, `eval/pairwise_agreement.csv`
