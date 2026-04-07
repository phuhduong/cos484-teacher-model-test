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

Streams the OpenMathReasoning dataset and samples 100 traces stratified by difficulty.

```bash
python scripts/pull_samples.py
```

Output: `data/raw/pilot_samples.jsonl`

### Step 2: Run teacher models

Sends each trace to a teacher model to insert `[CACHE]` tokens at reasoning-step boundaries. Runs 5 concurrent requests by default. Automatically resumes from where it left off if interrupted.

```bash
python scripts/run_teacher.py --model "Qwen/Qwen3-235B-A22B-Instruct-2507" --output-dir data/annotated/qwen_235b
python scripts/run_teacher.py --model "Qwen/Qwen3-32B" --output-dir data/annotated/qwen3_32b

# Re-run the same command to resume after interruption or failures
# Use --rerun to start fresh, --concurrency N to adjust parallelism
```

Output: `{output-dir}/annotated_samples.jsonl` and `{output-dir}/summary.json`

### Step 3: Compare annotations

```bash
# Print summary table and save comparison CSV
python scripts/compare.py --dir1 data/annotated/qwen_235b --dir2 data/annotated/qwen3_32b --output eval

# Interactive mode: rate each model's annotations side-by-side
python scripts/compare.py --dir1 data/annotated/qwen_235b --dir2 data/annotated/qwen3_32b --output eval --interactive
```

Output: `eval/comparison.csv` and `eval/ratings.csv` (interactive mode)
