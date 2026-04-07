"""
metrics.py - UIBF-aligned metrics for the LLM Reliability Pipeline.

Provides:
  1. compute_structure_consistency  - per-sample structural fidelity score
  2. compute_ovs_summary            - Output Variance Score by system x task
  3. append_result / load_results   - JSONL persistence layer
"""

import json
import math
from pathlib import Path


# ── 1. Structure Consistency ──────────────────────────────────────────────────

def compute_structure_consistency(
    parsed,
    schema: dict,
    parseable: bool
) -> float:
    """
    1.0  - parsed is a dict AND all required keys present
    0.5  - parsed is a dict BUT missing one or more required keys
    0.25 - parses as JSON but not a dict
    0.0  - not parseable JSON
    """
    if not parseable or parsed is None:
        return 0.0
    if not isinstance(parsed, dict):
        return 0.25
    required = schema.get("required", [])
    if not required:
        return 1.0
    missing = [k for k in required if k not in parsed]
    return 1.0 if not missing else 0.5


# ── 2. Output Variance Score ──────────────────────────────────────────────────

def _std(values: list) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def compute_ovs_summary(rows: list) -> list:
    """
    Aggregate result rows by system x task and compute OVS.

    OVS = 0.5 * std(schema_pass per style)
        + 0.25 * std(field_accuracy per style)
        + 0.25 * std(structure_consistency per style)

    Lower OVS = more stable across interaction styles.
    """
    STYLES = ["structured", "ambiguous", "verbose", "casual"]

    groups = {}
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
        schema_means, accuracy_means, consistency_means = [], [], []

        for style in STYLES:
            d = style_data[style]
            if d["schema"]:
                schema_means.append(sum(d["schema"]) / len(d["schema"]))
                accuracy_means.append(sum(d["accuracy"]) / len(d["accuracy"]))
                consistency_means.append(sum(d["consistency"]) / len(d["consistency"]))

        if not schema_means:
            continue

        mean_schema      = sum(schema_means) / len(schema_means)
        mean_accuracy    = sum(accuracy_means) / len(accuracy_means)
        mean_consistency = sum(consistency_means) / len(consistency_means)
        std_schema       = _std(schema_means)
        std_accuracy     = _std(accuracy_means)
        std_consistency  = _std(consistency_means)
        ovs = 0.5 * std_schema + 0.25 * std_accuracy + 0.25 * std_consistency

        total_rows = sum(len(style_data[s]["schema"]) for s in STYLES)

        summary.append({
            "system":                     system,
            "task":                       task,
            "n_rows":                     total_rows,
            "mean_schema_compliance":     round(mean_schema, 4),
            "std_schema_compliance":      round(std_schema, 4),
            "mean_field_accuracy":        round(mean_accuracy, 4),
            "std_field_accuracy":         round(std_accuracy, 4),
            "mean_structure_consistency": round(mean_consistency, 4),
            "std_structure_consistency":  round(std_consistency, 4),
            "overall_ovs":               round(ovs, 4),
        })

    summary.sort(key=lambda x: (x["system"], x["task"]))
    return summary


# ── 3. Result Storage ─────────────────────────────────────────────────────────

RESULTS_DIR  = Path(__file__).parent / "data" / "results"
RESULTS_FILE = RESULTS_DIR / "experiment_results.jsonl"


def _ensure_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def append_result(record: dict) -> None:
    """Append one result record to the JSONL store."""
    _ensure_dir()
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_results() -> list:
    """Load all stored result records."""
    _ensure_dir()
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
