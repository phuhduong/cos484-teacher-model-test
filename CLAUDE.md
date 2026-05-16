# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Data annotation pipeline for a COS484 project. Three scripts (`pull_samples.py` → `generate_traces.py` → `run_teacher.py`), plus `segment.py` as a shared helper module, pull BABILong samples (`RMT-team/babilong`), generate CoT reasoning traces over them with a thinking model, and send those traces to a teacher model to insert `[CACHE]` tokens. See README.md for setup and usage.

## Key Details

- Inference goes through Tinker's native Python SDK (`tinker.ServiceClient` → `create_sampling_client_async` → `SamplingClient`). Tinker's OAI-compatible HTTP endpoint only accepts `tinker://...` sampler URIs from your own training runs, not base model names — that's why we use the SDK. Chat messages are converted to tokens with the HuggingFace `tokenizer.apply_chat_template(...)` returned by `sampling_client.get_tokenizer()`, wrapped in `types.ModelInput.from_ints(...)`, then sampled via `sampling_client.sample_async(...)`. The response tokens are decoded with `tokenizer.decode(..., skip_special_tokens=True)`.
- BABILong has no built-in reasoning traces; `generate_traces.py` produces them with a thinking model (default `moonshotai/Kimi-K2-Thinking`) and tags each trace with `is_correct` using BABILong's `compare_answers` (label-match against `TASK_LABELS[qa_task]`, ignoring labels that appear in the question text)
- For the teacher, use a non-thinking model (e.g., `Qwen/Qwen3-235B-A22B-Instruct-2507`) — Qwen3 *Thinking* variants emit large `<think>` blocks that make per-call latency very high. Thinking models are intended for trace *generation* only.
- `run_teacher.py` only annotates traces with `is_correct == true` — wrong-answer traces likely cached the wrong facts, which is harmful training signal
- The selection prompt template uses `{problem}` and `{units}` placeholders (Python `.replace()`, not `.format()`)
- All data files are JSONL format; samples are linked across scripts by `id` field
- `run_teacher.py` segments traces, gets JSON cache-unit-IDs from the teacher, and reconstructs the annotated trace in code — preservation is exact by construction, and the final segmented unit is filtered out so `[CACHE]` never lands after the final answer
