"""Evaluation harness: score base vs fine-tuned model on freight->JSON extraction.

The scoring functions (parse, field_exact_match, score_predictions) are pure and
network-free so they can be unit-tested. The model-running entrypoint (`main`)
loads MLX models, generates predictions over test.jsonl, scores both base and
base+adapter, and writes results/*.json + comparison.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .schema import canonicalize_city, canonicalize_mode

FIELDS = ["origin", "destination", "weight_lbs", "mode", "on_time"]

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_prediction(text: str) -> dict | None:
    """Parse a model output string into a JSON dict, or None if it doesn't parse.

    Tolerant of leading/trailing prose: extracts the first {...} block if a bare
    json.loads fails. Returns None for anything that isn't a JSON object.
    """
    if text is None:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _canon_value(field: str, value: object) -> object:
    """Canonicalize a single field value for comparison."""
    if value is None:
        return None
    if field in ("origin", "destination"):
        return canonicalize_city(str(value))
    if field == "mode":
        return canonicalize_mode(str(value))
    if field == "weight_lbs":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if field == "on_time":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "yes"):
                return True
            if low in ("false", "no"):
                return False
        return None
    return value


def field_exact_match(pred: dict | None, gold: dict, field: str) -> bool:
    """True if pred's canonicalized value for `field` equals gold's."""
    if pred is None or field not in pred:
        return False
    return _canon_value(field, pred[field]) == _canon_value(field, gold[field])


def score_predictions(predictions: list[str], golds: list[dict]) -> dict:
    """Score a list of raw prediction strings against gold dicts.

    Returns json_valid_rate, per-field exact-match rates, field_exact_match
    (micro-average over all fields), and overall_accuracy (all-fields-correct rate).
    """
    if len(predictions) != len(golds):
        raise ValueError("predictions and golds must be the same length")
    n = len(golds)
    if n == 0:
        raise ValueError("need at least one example to score")

    valid = 0
    per_field_correct = dict.fromkeys(FIELDS, 0)
    all_correct = 0
    total_field_hits = 0

    for raw, gold in zip(predictions, golds, strict=True):
        parsed = parse_prediction(raw)
        if parsed is not None:
            valid += 1
        row_all = True
        for field in FIELDS:
            ok = field_exact_match(parsed, gold, field)
            if ok:
                per_field_correct[field] += 1
                total_field_hits += 1
            else:
                row_all = False
        if row_all:
            all_correct += 1

    return {
        "n": n,
        "json_valid_rate": valid / n,
        "field_accuracy": {f: per_field_correct[f] / n for f in FIELDS},
        "field_exact_match": total_field_hits / (n * len(FIELDS)),
        "overall_accuracy": all_correct / n,
    }


def load_test(path: Path) -> tuple[list[str], list[dict]]:
    """Load test.jsonl chat records -> (user_texts, gold_dicts)."""
    texts: list[str] = []
    golds: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec["messages"]
            user = next(m["content"] for m in messages if m["role"] == "user")
            assistant = next(m["content"] for m in messages if m["role"] == "assistant")
            texts.append(user)
            golds.append(json.loads(assistant))
    return texts, golds


def _generate_all(model, tokenizer, user_texts: list[str], system_prompt: str) -> list[str]:
    """Generate one completion per user text using mlx_lm.generate."""
    from mlx_lm import generate  # imported lazily so scoring stays network/MLX-free

    outputs: list[str] = []
    for user in user_texts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text = generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False)
        outputs.append(text)
    return outputs


def main() -> None:
    from mlx_lm import load

    from .prompt import SYSTEM_PROMPT

    parser = argparse.ArgumentParser(description="Eval base vs fine-tuned MLX extraction model")
    parser.add_argument("--model", required=True, help="base model id or path")
    parser.add_argument("--adapter", default="adapters", help="adapter dir for fine-tuned run")
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--iters", type=int, default=0, help="training iters (recorded only)")
    parser.add_argument("--limit", type=int, default=0, help="cap test examples (0 = all)")
    args = parser.parse_args()

    test_path = Path(args.test)
    user_texts, golds = load_test(test_path)
    if args.limit:
        user_texts = user_texts[: args.limit]
        golds = golds[: args.limit]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # BASE
    print(f"Loading base model: {args.model}")
    base_model, base_tok = load(args.model)
    print(f"Generating {len(user_texts)} base predictions...")
    base_preds = _generate_all(base_model, base_tok, user_texts, SYSTEM_PROMPT)
    base_scores = score_predictions(base_preds, golds)
    base_scores.update({"model": args.model, "adapter": None, "iters": args.iters})
    (results_dir / "base.json").write_text(json.dumps(base_scores, indent=2) + "\n")
    print("base:", json.dumps(base_scores, indent=2))

    # FINE-TUNED (base + adapter)
    print(f"Loading fine-tuned model: {args.model} + adapter {args.adapter}")
    ft_model, ft_tok = load(args.model, adapter_path=args.adapter)
    print(f"Generating {len(user_texts)} fine-tuned predictions...")
    ft_preds = _generate_all(ft_model, ft_tok, user_texts, SYSTEM_PROMPT)
    ft_scores = score_predictions(ft_preds, golds)
    ft_scores.update({"model": args.model, "adapter": args.adapter, "iters": args.iters})
    (results_dir / "finetuned.json").write_text(json.dumps(ft_scores, indent=2) + "\n")
    print("finetuned:", json.dumps(ft_scores, indent=2))

    write_comparison(base_scores, ft_scores, results_dir / "comparison.md")
    print(f"Wrote {results_dir}/comparison.md")


def write_comparison(base: dict, ft: dict, path: Path) -> None:
    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    lines = [
        "# Base vs Fine-Tuned: freight-text -> JSON extraction",
        "",
        f"- **Base model:** `{base['model']}`",
        f"- **Adapter:** `{ft['adapter']}` (LoRA, {ft['iters']} iters)",
        f"- **Test examples:** {base['n']}",
        "",
        "| Metric | Base | Fine-tuned | Delta |",
        "| --- | --- | --- | --- |",
        f"| JSON valid rate | {pct(base['json_valid_rate'])} | {pct(ft['json_valid_rate'])} | "
        f"{pct(ft['json_valid_rate'] - base['json_valid_rate'])} |",
        f"| Overall accuracy (all fields) | {pct(base['overall_accuracy'])} | "
        f"{pct(ft['overall_accuracy'])} | "
        f"{pct(ft['overall_accuracy'] - base['overall_accuracy'])} |",
        f"| Field exact match (micro) | {pct(base['field_exact_match'])} | "
        f"{pct(ft['field_exact_match'])} | "
        f"{pct(ft['field_exact_match'] - base['field_exact_match'])} |",
        "",
        "### Per-field exact match",
        "",
        "| Field | Base | Fine-tuned |",
        "| --- | --- | --- |",
    ]
    for field in FIELDS:
        lines.append(
            f"| {field} | {pct(base['field_accuracy'][field])} | "
            f"{pct(ft['field_accuracy'][field])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
