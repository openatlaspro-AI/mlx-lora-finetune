"""Network-free unit tests for schema, dataset generator, prompt, and eval scorer."""

from __future__ import annotations

import json

from mlx_lora_extraction.eval import (
    field_exact_match,
    parse_prediction,
    score_predictions,
)
from mlx_lora_extraction.make_dataset import generate_split
from mlx_lora_extraction.prompt import SYSTEM_PROMPT, build_messages
from mlx_lora_extraction.schema import ShipmentExtraction

# --- dataset generator ---------------------------------------------------------


def test_generator_is_deterministic():
    a = generate_split(50, seed=42)
    b = generate_split(50, seed=42)
    assert a == b


def test_generator_changes_with_seed():
    a = generate_split(50, seed=42)
    b = generate_split(50, seed=43)
    assert a != b


def test_every_record_parses_to_schema():
    records = generate_split(100, seed=7)
    for rec in records:
        messages = rec["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assistant = messages[-1]["content"]
        payload = json.loads(assistant)
        # Round-trips through the pydantic model without error.
        model = ShipmentExtraction(**payload)
        assert model.weight_lbs >= 0
        assert model.origin != model.destination


def test_generator_produces_requested_count_unique():
    records = generate_split(120, seed=11)
    assert len(records) == 120
    texts = {r["messages"][1]["content"] for r in records}
    assert len(texts) == 120  # all user prompts unique


# --- prompt --------------------------------------------------------------------


def test_prompt_includes_all_field_names():
    for field in ("origin", "destination", "weight_lbs", "mode", "on_time"):
        assert field in SYSTEM_PROMPT
    messages = build_messages("Some shipment text")
    assert messages[0]["role"] == "system"
    assert "Some shipment text" in messages[1]["content"]


# --- eval scorer ---------------------------------------------------------------


GOLD = {
    "origin": "Toronto",
    "destination": "Detroit",
    "weight_lbs": 12400,
    "mode": "LTL",
    "on_time": True,
}


def test_parse_prediction_valid_and_invalid():
    assert parse_prediction(json.dumps(GOLD)) == GOLD
    assert parse_prediction("not json at all") is None
    # tolerant extraction of embedded JSON
    assert parse_prediction("Here you go: " + json.dumps(GOLD)) == GOLD
    # malformed brace -> None, no crash
    assert parse_prediction("{origin: Toronto, ") is None


def test_field_exact_match_canonicalization():
    pred = {**GOLD, "origin": "  toronto "}  # whitespace + case
    assert field_exact_match(pred, GOLD, "origin") is True
    pred_wrong = {**GOLD, "destination": "Chicago"}
    assert field_exact_match(pred_wrong, GOLD, "destination") is False
    # missing field counts as miss, not crash
    assert field_exact_match({}, GOLD, "mode") is False


def test_score_all_correct():
    preds = [json.dumps(GOLD)]
    scores = score_predictions(preds, [GOLD])
    assert scores["json_valid_rate"] == 1.0
    assert scores["overall_accuracy"] == 1.0
    assert scores["field_exact_match"] == 1.0


def test_score_malformed_counts_invalid_not_crash():
    preds = ["{broken json", json.dumps(GOLD)]
    golds = [GOLD, GOLD]
    scores = score_predictions(preds, golds)
    assert scores["n"] == 2
    assert scores["json_valid_rate"] == 0.5
    assert scores["overall_accuracy"] == 0.5  # only the second is fully correct


def test_score_partial_field_credit():
    # one field wrong -> not overall-correct, but most fields still match
    wrong_mode = {**GOLD, "mode": "FTL"}
    scores = score_predictions([json.dumps(wrong_mode)], [GOLD])
    assert scores["overall_accuracy"] == 0.0
    assert scores["json_valid_rate"] == 1.0
    # 4 of 5 fields correct
    assert abs(scores["field_exact_match"] - 0.8) < 1e-9
    assert scores["field_accuracy"]["mode"] == 0.0
    assert scores["field_accuracy"]["origin"] == 1.0
