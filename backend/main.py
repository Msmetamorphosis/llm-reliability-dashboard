"""
main.py  -  FastAPI backend for the LLM Reliability Pipeline dashboard.

Endpoints:
  GET  /api/health          - health check
  GET  /api/datasets        - return dataset metadata (sample counts)
  POST /api/run             - start experiment, stream results via SSE
"""

import json
import time
import asyncio
from pathlib import Path
from typing import AsyncGenerator

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import (
    TASK1_DATASET, TASK2_DATASET, CORPUS_FILE,
    PROMPT_STYLES, EXTRACTION_SCHEMA, QA_SCHEMA,
    MODEL_DEFAULT, TOP_K
)
from scoring import extract_json, check_schema, field_coverage, score_extraction, score_qa
from prompts import task1_prompt, task2_prompt
from retrieval import retrieve
from pipeline import run_pipeline

app = FastAPI(title="LLM Reliability Pipeline API")

# Allow the GitHub Pages frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from /frontend if running locally
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))


# ── Load datasets once at startup ────────────────────────────────────────────

def _load(path: Path) -> list:
    with open(path, "r") as f:
        return json.load(f)

task1_data: list = []
task2_data: list = []
corpus_data: list = []
corpus_lookup: dict = {}

@app.on_event("startup")
async def startup():
    global task1_data, task2_data, corpus_data, corpus_lookup
    task1_data  = _load(TASK1_DATASET)
    task2_data  = _load(TASK2_DATASET)
    corpus_data = _load(CORPUS_FILE)
    corpus_lookup = {p["id"]: p["text"] for p in corpus_data}
    print(f"Loaded {len(task1_data)} Task1 samples, {len(task2_data)} Task2 samples, {len(corpus_data)} corpus passages")


# ── Request model ─────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    api_key:   str
    system:    str   # baseline_a | baseline_b | pipeline
    task:      str   # task1 | task2 | both
    styles:    list[str] = PROMPT_STYLES
    limit:     int  = 5
    model:     str  = MODEL_DEFAULT


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "task1_samples": len(task1_data),
        "task2_samples": len(task2_data),
        "corpus_passages": len(corpus_data)
    }


@app.get("/api/datasets")
async def datasets():
    return {
        "task1": len(task1_data),
        "task2": len(task2_data),
        "corpus": len(corpus_data),
        "styles": PROMPT_STYLES
    }


# ── Core single-sample runner ─────────────────────────────────────────────────

def run_one_sample(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    task: str,
    sample: dict,
    style: str,
) -> dict:
    schema = EXTRACTION_SCHEMA if task == "task1" else QA_SCHEMA

    # Build prompt
    if task == "task1":
        doc = sample["document"]
        ground_truth = sample["ground_truth"]
        if system == "baseline_b":
            retrieved = retrieve(doc[:200])  # query with start of doc
            context = "\n\n".join(retrieved)
            prompt = task1_prompt(f"Context:\n{context}\n\nDocument:\n{doc}", style)
        else:
            prompt = task1_prompt(doc, style)
    else:
        question = sample["question"]
        ground_truth = sample["ground_truth_answer"]
        if system == "baseline_b":
            passages = retrieve(question)
        else:
            passage_id = sample["supporting_passage_id"]
            passages = [corpus_lookup.get(passage_id, "")]
        prompt = task2_prompt(question, passages, style)

    # Run through correct system
    if system == "pipeline":
        result = run_pipeline(client, model, prompt, task)
    else:
        start = time.perf_counter()
        msg = client.messages.create(
            model=model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = (time.perf_counter() - start) * 1000
        raw = msg.content[0].text
        parsed, _ = extract_json(raw)
        valid, schema_err = check_schema(parsed, schema)
        result = {
            "raw_output": raw, "parsed_output": parsed,
            "schema_valid": valid, "parseable": parsed is not None,
            "llm_calls": 1, "latency_ms": round(latency, 2),
            "repair_attempted": False, "schema_error": schema_err
        }

    # Score
    if task == "task1":
        acc = score_extraction(result["parsed_output"], ground_truth)
        coverage = field_coverage(result["parsed_output"], schema)
        qa_score = {}
    else:
        acc = {"overall_accuracy": 0.0, "fields_correct": 0, "fields_total": 0, "field_results": {}}
        coverage = 0.0
        predicted_ans = (result["parsed_output"] or {}).get("answer", "")
        qa_score = score_qa(predicted_ans, ground_truth)

    return {
        "sample_id":        sample.get("id", "unknown"),
        "system":           system,
        "task":             task,
        "prompt_style":     style,
        "schema_valid":     result["schema_valid"],
        "parseable":        result["parseable"],
        "overall_accuracy": acc["overall_accuracy"],
        "fields_correct":   acc["fields_correct"],
        "fields_total":     acc["fields_total"],
        "field_coverage":   coverage,
        "qa_overlap_score": qa_score.get("overlap_score", 0.0),
        "qa_likely_correct":qa_score.get("likely_correct", False),
        "latency_ms":       result["latency_ms"],
        "llm_calls":        result["llm_calls"],
        "repair_attempted": result.get("repair_attempted", False),
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ── SSE streaming experiment runner ──────────────────────────────────────────

async def experiment_stream(req: RunRequest) -> AsyncGenerator[dict, None]:
    # Validate API key format
    if not req.api_key.startswith("sk-ant-"):
        yield {"event": "error", "data": json.dumps({"message": "Invalid API key format. Must start with sk-ant-"})}
        return

    try:
        client = anthropic.Anthropic(api_key=req.api_key)
    except Exception as e:
        yield {"event": "error", "data": json.dumps({"message": str(e)})}
        return

    tasks_to_run = []
    if req.task in ("task1", "both"):
        tasks_to_run.append(("task1", task1_data[:req.limit]))
    if req.task in ("task2", "both"):
        tasks_to_run.append(("task2", task2_data[:req.limit]))

    styles = req.styles if req.styles else PROMPT_STYLES
    total = sum(len(samples) for _, samples in tasks_to_run) * len(styles)
    completed = 0

    yield {"event": "start", "data": json.dumps({
        "total": total,
        "system": req.system,
        "task": req.task,
        "styles": styles,
        "model": req.model,
        "limit": req.limit
    })}

    for task, samples in tasks_to_run:
        for sample in samples:
            for style in styles:
                try:
                    # Run in thread pool so async loop is not blocked
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        run_one_sample,
                        client, req.model, req.system, task, sample, style
                    )
                    completed += 1
                    yield {
                        "event": "result",
                        "data": json.dumps({**result, "completed": completed, "total": total})
                    }
                except anthropic.AuthenticationError:
                    yield {"event": "error", "data": json.dumps({"message": "Invalid API key. Check your key and try again."})}
                    return
                except anthropic.RateLimitError:
                    yield {"event": "error", "data": json.dumps({"message": "Rate limit hit. Wait a moment and try again."})}
                    return
                except Exception as e:
                    yield {
                        "event": "result",
                        "data": json.dumps({
                            "sample_id": sample.get("id", "unknown"),
                            "system": req.system, "task": task, "prompt_style": style,
                            "schema_valid": False, "parseable": False,
                            "overall_accuracy": 0.0, "latency_ms": 0,
                            "llm_calls": 0, "error": str(e),
                            "completed": completed, "total": total
                        })
                    }

                # Small delay to stay well within rate limits
                await asyncio.sleep(0.3)

    yield {"event": "done", "data": json.dumps({"completed": completed, "total": total})}


@app.post("/api/run")
async def run_experiment(req: RunRequest):
    return EventSourceResponse(experiment_stream(req))
