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

Sends each trace to a teacher model to insert `[CACHE]` tokens at reasoning-step boundaries. Runs 10 concurrent requests by default. Automatically resumes from where it left off if interrupted.

```bash
python scripts/run_teacher.py --model "Qwen/Qwen3-235B-A22B-Instruct-2507" --output-dir data/annotated/qwen3_235b
python scripts/run_teacher.py --model "deepseek-ai/DeepSeek-V3.1" --output-dir data/annotated/deepseek_v31
python scripts/run_teacher.py --model "meta-llama/Llama-3.3-70B-Instruct" --output-dir data/annotated/llama_70b

# Re-run the same command to resume after interruption or failures
# Use --rerun to start fresh, --concurrency N to adjust parallelism
```

Output: `{output-dir}/annotated_samples.jsonl` and `{output-dir}/summary.json`

### Step 3: Analyze annotations

Evaluates annotation quality across models using 6 metrics: text preservation, cache count appropriateness, density, sentence boundary placement, hedging proximity, and cross-model consensus.

```bash
python scripts/analyze.py
```

Output: `eval/per_sample_scores.csv`, `eval/model_summary.csv`, `eval/pairwise_agreement.csv`
