"""
pipeline.py  -  The reliability pipeline.
Step 1: Generate initial JSON output.
Step 2: Self-critique pass that checks and rewrites if needed.
Step 3: Schema validation. If invalid, one constrained repair attempt.
"""

import json
import anthropic
from config import MAX_TOKENS, EXTRACTION_SCHEMA, QA_SCHEMA
from scoring import extract_json, check_schema
from prompts import task1_prompt, task2_prompt


def _call(client: anthropic.Anthropic, model: str, prompt: str, system: str = "") -> tuple[str, float]:
    import time
    start = time.perf_counter()
    msg = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system or "You are a precise information extraction assistant.",
        messages=[{"role": "user", "content": prompt}]
    )
    latency = (time.perf_counter() - start) * 1000
    return msg.content[0].text, latency


def run_pipeline(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    task: str,
) -> dict:
    schema = EXTRACTION_SCHEMA if task == "task1" else QA_SCHEMA
    schema_str = json.dumps(schema, indent=2)
    total_calls = 0
    total_latency = 0.0

    # Step 1: Generate
    raw1, lat1 = _call(client, model, prompt)
    total_calls += 1
    total_latency += lat1
    parsed1, _ = extract_json(raw1)

    # Step 2: Self-critique
    critique_prompt = f"""You produced this output:
{raw1}

Required JSON schema:
{schema_str}

Review your output carefully. Identify any missing required fields, wrong types,
or schema violations. Rewrite the JSON to fix all issues.
Return ONLY valid JSON that matches the schema exactly. No explanation."""

    raw2, lat2 = _call(client, model, critique_prompt)
    total_calls += 1
    total_latency += lat2
    parsed2, _ = extract_json(raw2)

    # Step 3: Validate, one repair if needed
    valid2, err2 = check_schema(parsed2, schema)
    if valid2:
        return {
            "raw_output": raw2, "parsed_output": parsed2,
            "schema_valid": True, "parseable": True,
            "llm_calls": total_calls, "latency_ms": round(total_latency, 2),
            "repair_attempted": False
        }

    repair_prompt = f"""This JSON does not match the required schema:
{raw2}

Schema error: {err2}

Required schema:
{schema_str}

Fix ONLY the structural and type issues. Return valid JSON only, no explanation."""

    raw3, lat3 = _call(client, model, repair_prompt)
    total_calls += 1
    total_latency += lat3
    parsed3, _ = extract_json(raw3)
    valid3, _ = check_schema(parsed3, schema)

    return {
        "raw_output": raw3, "parsed_output": parsed3,
        "schema_valid": valid3, "parseable": parsed3 is not None,
        "llm_calls": total_calls, "latency_ms": round(total_latency, 2),
        "repair_attempted": True
    }
