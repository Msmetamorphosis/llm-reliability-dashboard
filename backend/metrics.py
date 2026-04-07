"""
metrics.py - UIBF-aligned metrics for the LLM Reliability Pipeline.

Adds three things:
  1. compute_structure_consistency  - per-sample structural fidelity score
  2. compute_ovs_summary            - Output Variance Score aggregated by system x task
  3. ResultStore                    - lightweight JSONL persistence layer
"""

import json
import math
import time
from pathlib import Path
from typing import Any


# ── 1. Structure Consistency ──────────────────────────────────────────────────

def compute_structure_consistency(
    parsed: dict | None,
    schema: dict,
    parseable: bool
) -> float:
    """
    Score structural completeness of a parsed output against its schema.

    Returns:
        1.0  - parsed is a dict AND all required keys are present
        0.5  - parsed is a dict BUT one or more required keys are missing
        0.25 - output is parseable JSON but not a dict
        0.0  - output is not parseable JSON at all
    """
    if not parseable or parsed is None:
        return 0.0
    if not isinstance(parsed, dict):
        return 0.25
    required_keys = schema.get("required", [])
    if not required_keys:
        return 1.0
    missing = [k for k in required_keys if k not in parsed]
    if not missing:
        return 1.0
    return 0.5


# ── 2. Output Variance Score (OVS) ────────────────────────────────────────────

def _std(values: list[float]) -> float:
    """Population standard deviation. Returns 0 if fewer than 2 values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def compute_ovs_summary(rows: list[dict]) -> list[dict]:
    """
    Aggregate experiment result rows by system x task and compute OVS.

    Each row must have:
        system, task, prompt_style,
        schema_pass, field_accuracy, structure_consistency

    OVS formula (UIBF-aligned, schema weighted most heavily):
        OVS = 0.5 * std(schema_pass per style)
            + 0.25 * std(field_accuracy per style)
            + 0.25 * std(structure_consistency per style)

    Lower OVS = more stable system across interaction styles.
    """
    STYLES = ["structured", "ambiguous", "verbose", "casual"]

    # Group rows by (system, task)
    groups: dict[tuple, dict[str, list]] = {}
    for row in rows:
        key = (row.get("system", "unknown"), row.get("task", "unknown"))
        style = row.get("prompt_style", "unknown")
        if key not in groups:
            groups[key] = {s: {"schema": [], "accuracy": [], "consistency": []}
                           for s in STYLES}
        if style in groups[key]:
            groups[key][style]["schema"].append(float(row.get("schema_pass", 0)))
            groups[key][style]["accuracy"].append(float(row.get("field_accuracy", 0)))
            groups[key][style]["consistency"].append(float(row.get("structure_consistency", 0)))

    summary = []
    for (system, task), style_data in groups.items():
        # Per-style means
        schema_means = []
        accuracy_means = []
        consistency_means = []

        for style in STYLES:
            d = style_data[style]
            if d["schema"]:
                schema_means.append(sum(d["schema"]) / len(d["schema"]))
                accuracy_means.append(sum(d["accuracy"]) / len(d["accuracy"]))
                consistency_means.append(sum(d["consistency"]) / len(d["consistency"]))

        # All-style means and stds
        mean_schema      = sum(schema_means) / len(schema_means) if schema_means else 0.0
        mean_accuracy    = sum(accuracy_means) / len(accuracy_means) if accuracy_means else 0.0
        mean_consistency = sum(consistency_means) / len(consistency_means) if consistency_means else 0.0

        std_schema      = _std(schema_means)
        std_accuracy    = _std(accuracy_means)
        std_consistency = _std(consistency_means)

        ovs = (0.5 * std_schema) + (0.25 * std_accuracy) + (0.25 * std_consistency)

        # Total row count
        total_rows = sum(
            len(style_data[s]["schema"]) for s in STYLES
        )

        summary.append({
            "system":                   system,
            "task":                     task,
            "n_rows":                   total_rows,
            "mean_schema_compliance":   round(mean_schema, 4),
            "std_schema_compliance":    round(std_schema, 4),
            "mean_field_accuracy":      round(mean_accuracy, 4),
            "std_field_accuracy":       round(std_accuracy, 4),
            "mean_structure_consistency": round(mean_consistency, 4),
            "std_structure_consistency":  round(std_consistency, 4),
            "overall_ovs":              round(ovs, 4),
        })

    summary.sort(key=lambda x: (x["system"], x["task"]))
    return summary


# ── 3. Result Store ────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "data" / "results"
RESULTS_FILE = RESULTS_DIR / "experiment_results.jsonl"


def _ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def append_result(record: dict) -> None:
    """Append one result record to the JSONL store. Safe for Render."""
    _ensure_results_dir()
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_results() -> list[dict]:
    """Load all stored result records from JSONL."""
    _ensure_results_dir()
    if not RESULTS_FILE.exists():
        return []
    records = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
