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
        # null ground truth: predicted should also be null or empty
        return predicted is None or predicted == "" or predicted == []
    if predicted is None:
        # predicted is null but ground truth is not
        return False
    if isinstance(ground_truth, bool):
        if isinstance(predicted, bool):
            return predicted == ground_truth
        if isinstance(predicted, str):
            return predicted.lower() in ("true" if ground_truth else "false")
        return False
    if isinstance(ground_truth, (int, float)):
        try:
            return abs(float(predicted) - float(ground_truth)) < 0.01
        except (TypeError, ValueError):
            # Try extracting number from string
            import re
            nums = re.findall(r"[\d,]+\.?\d*", str(predicted).replace(",", ""))
            for n in nums:
                try:
                    if abs(float(n) - float(ground_truth)) < 0.01:
                        return True
                except ValueError:
                    pass
            return False
    if isinstance(ground_truth, str):
        if not isinstance(predicted, str):
            predicted = str(predicted)
        gt = ground_truth.strip().lower()
        pred = predicted.strip().lower()
        # Direct containment
        if gt in pred or pred in gt:
            return True
        # Key word overlap: if 60%+ of ground truth words appear in prediction
        gt_words = set(w for w in gt.split() if len(w) > 3)
        if not gt_words:
            return gt == pred
        pred_words = set(pred.split())
        overlap = len(gt_words & pred_words) / len(gt_words)
        return overlap >= 0.6
    if isinstance(ground_truth, list):
        if not ground_truth:
            return isinstance(predicted, list)
        if not isinstance(predicted, list):
            # Try treating predicted as a single-item list
            predicted = [str(predicted)]
        # Each ground truth item should have a match somewhere in predicted
        for gt_item in ground_truth:
            if not isinstance(gt_item, str):
                continue
            gt_lower = gt_item.strip().lower()
            found = False
            for p in predicted:
                if not isinstance(p, str):
                    continue
                p_lower = p.strip().lower()
                if gt_lower in p_lower or p_lower in gt_lower:
                    found = True
                    break
                # Word overlap fallback
                gt_words = set(w for w in gt_lower.split() if len(w) > 3)
                p_words  = set(p_lower.split())
                if gt_words and len(gt_words & p_words) / len(gt_words) >= 0.5:
                    found = True
                    break
            if not found:
                return False
        return True
    return str(predicted).strip().lower() == str(ground_truth).strip().lower()


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
