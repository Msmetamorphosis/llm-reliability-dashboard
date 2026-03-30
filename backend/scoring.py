"""
scoring.py  -  All evaluation metrics.
"""

import json
import re
import jsonschema
from typing import Any


def extract_json(raw: str) -> tuple[dict | None, str]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned), ""
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), ""
        except json.JSONDecodeError as e:
            return None, str(e)
    return None, "No JSON object found"


def check_schema(parsed: dict | None, schema: dict) -> tuple[bool, str]:
    if parsed is None:
        return False, "Nothing to validate"
    try:
        jsonschema.validate(instance=parsed, schema=schema)
        return True, ""
    except jsonschema.ValidationError as e:
        return False, e.message


def field_coverage(parsed: dict | None, schema: dict) -> float:
    if parsed is None:
        return 0.0
    required = schema.get("required", [])
    if not required:
        return 1.0
    populated = sum(
        1 for f in required
        if f in parsed and parsed[f] is not None and parsed[f] != [] and parsed[f] != ""
    )
    return round(populated / len(required), 4)


def compare_field(predicted: Any, ground_truth: Any) -> bool:
    if ground_truth is None:
        return predicted is None
    if isinstance(ground_truth, bool):
        return predicted == ground_truth
    if isinstance(ground_truth, (int, float)):
        try:
            return float(predicted) == float(ground_truth)
        except (TypeError, ValueError):
            return False
    if isinstance(ground_truth, str):
        if not isinstance(predicted, str):
            return False
        return ground_truth.strip().lower() in predicted.strip().lower()
    if isinstance(ground_truth, list):
        if not isinstance(predicted, list):
            return False
        for gt_item in ground_truth:
            found = any(
                gt_item.strip().lower() in p.strip().lower()
                for p in predicted if isinstance(p, str)
            )
            if not found:
                return False
        return True
    return predicted == ground_truth


def score_extraction(parsed: dict | None, ground_truth: dict) -> dict:
    if parsed is None:
        return {"overall_accuracy": 0.0, "fields_correct": 0,
                "fields_total": len(ground_truth), "field_results": {}}
    field_results = {}
    correct = 0
    for field, gt_value in ground_truth.items():
        predicted_value = parsed.get(field)
        is_correct = compare_field(predicted_value, gt_value)
        field_results[field] = {"correct": is_correct}
        if is_correct:
            correct += 1
    return {
        "overall_accuracy": round(correct / len(ground_truth), 4),
        "fields_correct": correct,
        "fields_total": len(ground_truth),
        "field_results": field_results
    }


def score_qa(predicted_answer: str, ground_truth_answer: str) -> dict:
    if not predicted_answer:
        return {"overlap_score": 0.0, "likely_correct": False, "numeric_facts_missing": []}
    gt_lower   = ground_truth_answer.lower()
    pred_lower = predicted_answer.lower()
    numbers    = re.findall(r"\$?[\d,]+\.?\d*", gt_lower)
    missing    = [n for n in numbers if n not in pred_lower]
    gt_words   = set(re.findall(r"\b\w{4,}\b", gt_lower))
    pred_words = set(re.findall(r"\b\w{4,}\b", pred_lower))
    overlap    = len(gt_words & pred_words) / max(len(gt_words), 1)
    return {
        "overlap_score": round(overlap, 4),
        "likely_correct": overlap >= 0.5 and len(missing) == 0,
        "numeric_facts_missing": missing
    }
