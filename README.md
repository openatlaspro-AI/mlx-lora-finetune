# mlx-lora-finetune

Fine-tune a small **4-bit quantized LLM with LoRA (QLoRA-style) on Apple MLX** to do
**structured field extraction** from freeform freight/shipment text into strict JSON — with an
**honest before/after benchmark** (base model vs base + LoRA adapter) on a held-out test set.

> **Task:** given a sentence like *"Rush LTL load, 12,400 lbs from Toronto to Detroit, carrier
> on-time"*, output `{"origin": "Toronto", "destination": "Detroit", "weight_lbs": 12400,
> "mode": "LTL", "on_time": true}` (mode ∈ LTL/FTL/Parcel/Intermodal).

## Results (real run, committed receipts)

Numbers below are copied verbatim from [`results/comparison.md`](results/comparison.md), which is
written by the actual eval over 150 held-out test examples — not hand-entered.

- **Base model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (4-bit quantized — the "Q" in QLoRA)
- **Fine-tune:** LoRA, 8 layers, rank 8, **600 iters**, batch 4, lr 1e-4 — trained in **~6m14s**
  wall-clock on an Apple Silicon Mac (peak ~1.85 GB memory)

| Metric | Base | Fine-tuned | Delta |
| --- | --- | --- | --- |
| JSON valid rate | 100.0% | 100.0% | +0.0% |
| Overall accuracy (all fields correct) | **71.3%** | **100.0%** | **+28.7%** |
| Field exact match (micro-avg) | 93.7% | 100.0% | +6.3% |

Per-field exact match:

| Field | Base | Fine-tuned |
| --- | --- | --- |
| origin | 100.0% | 100.0% |
| destination | 100.0% | 100.0% |
| weight_lbs | 100.0% | 100.0% |
| mode | 80.0% | 100.0% |
| on_time | 88.7% | 100.0% |

The base 0.5B model already emits valid JSON and nails the easy fields, but it mislabels the
shipment **mode** and **on-time** status often enough that only 71% of records are *fully* correct.
LoRA fine-tuning closes that gap to 100% on this held-out set, driven almost entirely by fixing
`mode` (80→100%) and `on_time` (88.7→100%).

### Honest notes

- **Synthetic data.** The dataset is generated deterministically by
  [`make_dataset.py`](src/mlx_lora_extraction/make_dataset.py) (seeded, network-free) — realistic
  phrasing but not real shipment records. A 100% fine-tuned score reflects a clean, well-specified
  synthetic task; expect lower numbers on noisy real-world freight text. The point of the project is
  the *method and the honest before/after delta*, not the absolute ceiling.
- **Every number here comes from a committed `results/*.json` produced by an actual run** on Apple
  Silicon. If you re-run, regenerate the receipts (see below) rather than editing the table.

## How to reproduce

Requires **Apple Silicon (arm64) macOS** and Python 3.11+.

```bash
# 1. Environment
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Generate the synthetic dataset -> data/{train,valid,test}.jsonl (committed already)
make data

# 3. Train the LoRA adapter on the 4-bit base -> adapters/  (~6 min)
make train

# 4. Eval base vs fine-tuned over the test set -> results/{base,finetuned}.json + comparison.md
make eval
```

If `mlx-community/Qwen2.5-0.5B-Instruct-4bit` is unavailable, the design fallback is
`mlx-community/Qwen2.5-1.5B-Instruct-4bit`; pass `MODEL=...` to `make train`/`make eval` and the
exact id is recorded in the results JSON either way.

## Layout

```
src/mlx_lora_extraction/
  schema.py        # pydantic ShipmentExtraction model + value canonicalization
  prompt.py        # system+user prompt shared by training data and eval
  make_dataset.py  # deterministic synthetic generator -> MLX-LM chat JSONL
  eval.py          # pure scorer (json_valid_rate / field exact match / overall) + model runner
lora_config.yaml   # mlx_lm.lora training config
data/*.jsonl       # generated dataset (committed)
adapters/          # trained LoRA adapter (final adapters.safetensors committed, ~6 MB)
results/           # base.json, finetuned.json, comparison.md — real receipts
tests/             # network-free unit tests (schema, generator, prompt, scorer)
```

## Tests / CI

```bash
ruff check .
pytest            # network-free; no model download, no training
```

CI (`.github/workflows/ci.yml`) runs **ruff + pytest only** — it never trains (training needs Apple
Silicon + MLX).

## License

MIT © 2026 Mark Teji.
