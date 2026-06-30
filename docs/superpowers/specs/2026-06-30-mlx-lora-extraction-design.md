# mlx-lora-finetune — Design & Plan

**Date:** 2026-06-30
**Repo:** new — `openatlaspro-AI/mlx-lora-finetune` (branch `build/initial-implementation`)
**Status:** Approved (autonomous), ready to build

## Goal

Fine-tune a small LLM with **LoRA on a 4-bit quantized base (QLoRA-style) using Apple MLX** to do
**structured field extraction from freeform freight/shipment text → JSON**, and prove it works with
an **honest before/after benchmark** (base model vs fine-tuned adapter) on a held-out set. Every
reported number comes from a committed eval JSON.

## Why (hiring context)

Closes the single biggest gap on Mark's resume: **model fine-tuning / customization**. Demonstrates
QLoRA/LoRA, dataset construction, adapter training, and rigorous before/after evaluation — on-brand
with his logistics domain, runnable on his Mac Mini M4 (Apple Silicon, MLX).

## Task & data

- **Task:** given a freeform shipment sentence (e.g. "Rush LTL load, 12,400 lbs from Toronto to
  Detroit, carrier on-time"), output a strict JSON object with fields:
  `{origin, destination, weight_lbs, mode, on_time}` (mode ∈ LTL/FTL/Parcel/Intermodal; on_time bool).
- **Data:** `src/make_dataset.py` deterministically generates realistic synthetic examples
  (seeded, no network): ~600 train / ~150 valid / ~150 test, written as MLX-LM chat-format JSONL
  (`data/{train,valid,test}.jsonl`). Vary phrasing templates, units, city pairs, modes, noise.

## Approach

- **Base model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (ungated, small, 4-bit → the "Q" in
  QLoRA). If unavailable, fall back to `mlx-community/Qwen2.5-1.5B-Instruct-4bit` and record which
  was used in the eval JSON.
- **Train:** `mlx_lm.lora` (LoRA adapters over the quantized base). Config in `lora_config.yaml`
  (iters ~600, batch 4, lora layers 8, learning-rate 1e-4 — tune down if memory-bound). Adapters
  saved to `adapters/`.
- **Eval:** `src/eval.py` runs both **base** and **fine-tuned** (base + adapter) over `test.jsonl`,
  parsing model output as JSON and scoring:
  - `json_valid_rate` (fraction of outputs that parse as JSON)
  - `field_exact_match` per field + overall (exact value match)
  - `overall_accuracy` (all-fields-correct rate)
  Writes `results/base.json`, `results/finetuned.json`, `results/comparison.md`.

## Architecture (files)

```
pyproject.toml                     # package + deps (mlx-lm, pydantic, pytest, ruff) + scripts
src/mlx_lora_extraction/
  __init__.py
  schema.py                        # pydantic ShipmentExtraction model + canonicalization
  make_dataset.py                  # deterministic synthetic generator -> data/*.jsonl
  prompt.py                        # system+user prompt builder (shared by train data + eval)
  eval.py                          # score base vs finetuned -> results/*.json + comparison.md
lora_config.yaml                   # mlx_lm.lora training config
data/{train,valid,test}.jsonl      # generated, committed (small)
results/{base,finetuned}.json, comparison.md   # real receipts from the run
adapters/                          # trained LoRA adapter (committed if small; else gitignored w/ note)
tests/test_dataset_and_eval.py     # network-free unit tests
README.md, LICENSE, .github/workflows/ci.yml
```

## Honesty constraints

1. **No fabricated metrics.** `results/*.json` are written by the actual eval run. The README table
   is copied from them. If training or model download fails, STATUS: BLOCKED — do not invent numbers.
2. **Report whatever happens.** Fine-tuned is *expected* to beat base on JSON validity + field
   accuracy, but report the real delta even if small or mixed.
3. **Record the exact base model + iters** used in the eval JSON and README.

## Tests (network-free, no training)

`tests/test_dataset_and_eval.py`:
- dataset generator is deterministic (same seed → identical output) and every record parses to the
  pydantic schema.
- the eval scorer: given hand-built (prediction, gold) pairs, `field_exact_match` / `overall_accuracy`
  / `json_valid_rate` return the expected values (including a malformed-JSON prediction → counted
  invalid, not crashing).
- prompt builder includes all five field names.

## Deliverables

- Public repo, MIT, README with before/after table + "how to reproduce" (`make data && make train &&
  make eval`), GitHub Actions CI (ruff + pytest only — NOT training).
- Real `results/` receipts from a completed run on Apple Silicon.

## Resume bullet earned (real numbers post-build)

> Fine-tuned **Qwen2.5 (4-bit) with LoRA via Apple MLX** for structured freight-text → JSON
> extraction; built the synthetic dataset, training, and a before/after eval harness. Fine-tuning
> lifted overall field accuracy from **[base]% to [ft]%** and JSON-validity to **[ft]%** on a
> held-out set — every number backed by a committed results JSON. Open source.
