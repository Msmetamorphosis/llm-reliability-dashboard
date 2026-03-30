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

## Deploy in 15 minutes

### Step 1: Fork or create this repo on GitHub

Upload all files maintaining the folder structure above.

### Step 2: Deploy backend to Render

1. Go to render.com and sign in with GitHub
2. Click **New** and select **Web Service**
3. Connect this GitHub repo
4. Render will auto-detect `render.yaml` and configure everything
5. Click **Deploy**
6. Wait 3-5 minutes. Copy your service URL (format: `https://your-app.onrender.com`)

### Step 3: Deploy frontend to GitHub Pages

1. In your GitHub repo, go to **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: **main**, folder: **/frontend**
4. Save. Wait 2 minutes.
5. Your dashboard is live at `https://yourusername.github.io/repo-name`

### Step 4: Use the dashboard

1. Visit your GitHub Pages URL
2. Enter your Anthropic API key (`sk-ant-...`) in the top banner
3. Enter your Render backend URL in the second banner
4. Click **Test connection** to verify
5. Select system, task, styles, and sample limit
6. Click **Run Experiment** and watch results stream in real time

## Research context

This project tests the conjecture that LLM output failures in enterprise settings
are driven by user prompt behavior rather than model capability. The key finding
from Baseline A is a 100 percentage point compliance gap between structured and
casual prompts despite a 100% JSON parse rate, proving that prompt style is the
root cause of silent output failures in production AI systems.

Domain: VA veteran benefits (public domain, va.gov)
Model: Claude (Anthropic API)
Metrics: JSON schema compliance, field accuracy vs ground truth, output variance, latency
