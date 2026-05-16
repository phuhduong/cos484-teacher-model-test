# cos484-teacher-model

Data annotation pipeline for the COS 484 final project. Generates CoT traces over BABILong samples (`RMT-team/babilong`) with a thinking model, then uses a teacher model to insert `[CACHE]` tokens after reasoning units worth caching for later recall.

## Setup

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Then create a `.env` with your Tinker API key:

```
TINKER_API_KEY=your_key_here
```

## Pipeline

Run the three steps in order; each consumes the previous step's output. The (qa_task, length_config) grid is hardcoded to `{qa1, qa2, qa3} × {0k, 1k, 2k}`.

### Step 1: Pull BABILong samples

Pulls 50 samples per cell from the 3×3 grid = 450 total. Pulls are deterministic (always the first N rows of each split).

```bash
python3 scripts/pull_samples.py               # defaults: 450 samples
python3 scripts/pull_samples.py --per-cell 2  # 18-sample smoke test
```

Output: `data/raw/babilong_samples.jsonl` with `{id, problem, context, target, qa_task, length_config}`.

### Step 2: Generate CoT traces

Runs a thinking model over the pulled samples, extracts the final answer, and tags each trace with `is_correct` using BABILong's label-match check. On re-run, retries anything missing or wrong.

```bash
python3 scripts/generate_traces.py
# --model NAME for a different generator (default: moonshotai/Kimi-K2-Thinking)
# --rerun to redo even correct rows, --concurrency N to adjust parallelism, --limit N for smoke runs
```

Output: `data/traces/babilong_traces.jsonl` with `{id, problem, think_trace, expected_answer, predicted_answer, raw_response, is_correct, finish_reason, qa_task, length_config}`.

### Step 3: Run teacher model

For each trace, `run_teacher.py` segments it into numbered reasoning units, asks the teacher to return a JSON list of unit IDs after which `[CACHE]` should be inserted, then reconstructs the annotated trace in code — so the original text is preserved exactly. The final unit is dropped so `[CACHE]` never lands after the answer. Only traces with `is_correct == true` are annotated.

The teacher's instructions live in `prompts/cache_selection_prompt.txt` — edit there to tune what gets cached.

```bash
python3 scripts/run_teacher.py --model "Qwen/Qwen3-235B-A22B-Instruct-2507" --output-dir data/annotated/qwen3_235b
python3 scripts/run_teacher.py --model "deepseek-ai/DeepSeek-V3.1" --output-dir data/annotated/deepseek_v31

# --rerun to start fresh, --concurrency N to adjust parallelism, --limit N for smoke runs
```

Output: `{output-dir}/annotated_samples.jsonl` and `{output-dir}/summary.json`. Each row includes `{id, problem, original_trace, annotated_trace, num_cache_tokens, model, expected_answer, selected_unit_ids, num_units, had_invalid_ids}`. The summary reports avg/min/max cache-token counts, invalid-ID rate, final-unit-selection rate, and total token usage.
