# LLM Reliability Pipeline — Research Dashboard

**Crystal Tubbs | KSU MSAI | Metamorphic Curations LLC**

A full-stack AI research application testing whether a minimal reliability layer
reduces LLM output failures across behavioral prompt variations. Anyone can visit
the live dashboard, enter their own Anthropic API key, and run the real experiment.

## Architecture

```
GitHub Repo
├── frontend/index.html      →  GitHub Pages (the dashboard UI)
├── backend/                 →  Render.com (Python FastAPI + real experiment logic)
│   ├── main.py              (FastAPI server, SSE streaming)
│   ├── pipeline.py          (self-critique + schema validation + repair)
│   ├── retrieval.py         (ChromaDB RAG for Baseline B)
│   ├── scoring.py           (JSON compliance, accuracy, QA scoring)
│   ├── prompts.py           (4 behavioral prompt wrappers)
│   ├── config.py            (paths, schemas, settings)
│   └── requirements.txt
├── data/
│   ├── labeled/             (15 extraction samples + 25 QA pairs with ground truth)
│   └── corpus/              (20 VA benefits passages for retrieval)
└── render.yaml              (Render deployment config)
```

## Three systems tested

| System | What it does | LLM calls per output |
|---|---|---|
| Baseline A | Direct LLM, no retrieval, no validation | 1 |
| Baseline B | ChromaDB retrieval, no validation | 1 |
| Pipeline | RAG + self-critique + schema validation + repair | 2-3 |

## Four prompt styles (the behavioral variable)

- **Structured** — explicit, well-formed, professional
- **Ambiguous** — underspecified, vague
- **Verbose** — over-explained, buried request
- **Casual** — informal, typo-prone

## Research context

This project tests the conjecture that LLM output failures in enterprise settings
are driven by user prompt behavior rather than model capability. The key finding
from Baseline A is a 100 percentage point compliance gap between structured and
casual prompts despite a 100% JSON parse rate, proving that prompt style is the
root cause of silent output failures in production AI systems.

Domain: VA veteran benefits (public domain, va.gov)
Model: Claude (Anthropic API)
Metrics: JSON schema compliance, field accuracy vs ground truth, output variance, latency
