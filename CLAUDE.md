# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Data annotation pipeline for a COS484 project. Three scripts (`pull_samples.py` → `run_teacher.py` → `compare.py`) pull math CoT traces, send them to teacher models to insert `[CACHE]` tokens, and compare outputs. See README.md for setup and usage.

## Key Details

- Tinker API uses OpenAI SDK with `base_url` override — model names passed as-is (e.g., `Qwen/Qwen3-235B-A22B-Instruct-2507`)
- Avoid Qwen3.5-* models — they are thinking models that generate large `<think>` blocks, making them very slow for annotation tasks
- The prompt template uses `{problem}` and `{trace}` placeholders (Python `.replace()`, not `.format()`)
- All data files are JSONL format; samples are linked across scripts by `id` field
- `run_teacher.py` rejects unclosed `<think>` blocks and strips preamble text echoed from the prompt template
