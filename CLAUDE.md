# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Data annotation pipeline for a COS484 project. Four scripts (`pull_samples.py` → `generate_traces.py` → `run_teacher.py` → `analyze.py`) pull BABILong samples (`RMT-team/babilong`), generate CoT reasoning traces over them with a thinking model, send those traces to teacher models to insert `[CACHE]` tokens, and evaluate annotation quality. See README.md for setup and usage.

## Key Details

- Tinker API uses OpenAI SDK with `base_url` override — model names passed as-is (e.g., `Qwen/Qwen3-235B-A22B-Instruct-2507`)
- BABILong has no built-in reasoning traces; `generate_traces.py` produces them with a thinking model (default `moonshotai/Kimi-K2-Thinking`) and tags each trace with `is_correct` using BABILong's `compare_answers` (label-match against `TASK_LABELS[qa_task]`, ignoring labels that appear in the question text)
- Avoid Qwen3.5-* models for annotation — they are thinking models that generate large `<think>` blocks, making them very slow per call. They are fine (in fact intended) for trace *generation*.
- `run_teacher.py` filters input to `is_correct == true` by default (override with `--no-require-correct`) — wrong-answer traces likely cached the wrong facts, which is harmful training signal
- The selection prompt template uses `{problem}` and `{units}` placeholders (Python `.replace()`, not `.format()`)
- All data files are JSONL format; samples are linked across scripts by `id` field
- `run_teacher.py` segments traces, gets JSON cache-unit-IDs from the teacher, and reconstructs the annotated trace in code — preservation is an invariant, not a scored metric
- The final segmented unit is filtered out before reconstruction, so `[CACHE]` never lands after the final answer
