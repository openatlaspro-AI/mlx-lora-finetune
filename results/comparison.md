# Base vs Fine-Tuned: freight-text -> JSON extraction

- **Base model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- **Adapter:** `adapters` (LoRA, 600 iters)
- **Test examples:** 150

| Metric | Base | Fine-tuned | Delta |
| --- | --- | --- | --- |
| JSON valid rate | 100.0% | 100.0% | 0.0% |
| Overall accuracy (all fields) | 71.3% | 100.0% | 28.7% |
| Field exact match (micro) | 93.7% | 100.0% | 6.3% |

### Per-field exact match

| Field | Base | Fine-tuned |
| --- | --- | --- |
| origin | 100.0% | 100.0% |
| destination | 100.0% | 100.0% |
| weight_lbs | 100.0% | 100.0% |
| mode | 80.0% | 100.0% |
| on_time | 88.7% | 100.0% |
