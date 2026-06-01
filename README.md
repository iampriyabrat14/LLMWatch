# LLMWatch

> LLM Observability & Evaluation Platform for Production Agents

A plug-and-play monitoring layer for any LangChain or LangGraph agent. Tracks **cost per query**, **latency percentiles**, **token usage**, and **hallucination rate** (via RAGAS). Includes a live Streamlit dashboard and a GitHub Actions workflow that auto-runs evaluation on every PR — so you catch quality regressions before they hit production.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&logoColor=white)
![RAGAS](https://img.shields.io/badge/RAGAS-4B0082?style=flat&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-FF6B35?style=flat&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=githubactions&logoColor=white)

---

## Dashboard Preview

```
┌─────────────────────────────────────────────────────────────┐
│  LLMWatch — Production Agent Monitor                        │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  Cost/Query  │  P50 Latency │  P95 Latency │  RAGAS Score  │
│  $0.0023     │  820 ms      │  1,840 ms    │  0.87 / 1.0   │
├──────────────┴──────────────┴──────────────┴───────────────┤
│  Token Usage (last 24h)                                     │
│  Input:  1,240 avg/query    ████████████░░░░░░░░░░         │
│  Output:   380 avg/query    ████░░░░░░░░░░░░░░░░░░         │
├─────────────────────────────────────────────────────────────┤
│  Hallucination Score Trend  [CI gate: fail if < 0.75]       │
│  0.91 ▁▂▃▄▅▅▆▇█ 0.87  ← today                             │
└─────────────────────────────────────────────────────────────┘
```

> Add a screenshot of your Streamlit dashboard here: `assets/dashboard.png`

---

## Architecture

```
LangGraph / LangChain Agent
         │
         ▼
   LLMWatch Wrapper
   (callback handler)
         │
    ┌────┴────┐
    ▼         ▼
LangSmith   SQLite / Postgres
(tracing)   (metrics store)
         │
         ▼
   Streamlit Dashboard
   ┌──────────────────────────────────┐
   │ Cost/query       $0.0023        │
   │ P50 Latency      820 ms         │
   │ P95 Latency      1,840 ms       │
   │ Input Tokens     1,240 / query  │
   │ Output Tokens    380 / query    │
   │ RAGAS Score      0.87           │
   └──────────────────────────────────┘
         │
         ▼
GitHub Actions (on PR)
   → Run RAGAS eval suite
   → Post score as PR comment
   → Fail PR if score < threshold
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph, LangChain |
| LLM Tracing | LangSmith |
| Evaluation | RAGAS |
| Metrics Store | SQLite (dev) / PostgreSQL (prod) |
| Dashboard | Streamlit |
| CI/CD Eval | GitHub Actions |
| LLM | OpenAI GPT-4o, GPT-4o-mini, Groq, Gemini |

---

## Features

- Drop-in callback handler — wrap any existing agent in 2 lines
- **Latency tracking**: P50 / P95 / P99 percentiles, per-model breakdown, configurable alert threshold
- **Cost tracking**: per-call USD cost, cumulative spend, configurable budget limit alert
- **Token tracking**: input vs output token split per call and aggregated over time
- Multi-model pricing table (OpenAI, Anthropic Claude, Groq, Gemini)
- RAGAS evaluation: faithfulness, answer relevancy, context precision
- Streamlit dashboard with live metrics and time-series charts
- GitHub Actions eval workflow — auto-comments RAGAS scores on every PR
- PR quality gate — fails CI if RAGAS score drops below threshold
- LangSmith integration for full trace visualization

---

## Project Structure

```
LLMWatch/
├── llmwatch/
│   ├── __init__.py
│   ├── callback.py             # LangChain callback handler (core)
│   ├── metrics.py              # Cost + token tracking, per-model pricing table
│   ├── evaluator.py            # RAGAS eval runner + PR comment formatter
│   └── db.py                   # SQLite metrics persistence (P50/P95/P99, cost, tokens)
├── dashboard/
│   └── app.py                  # Streamlit dashboard (6 KPI cards + 4 charts)
├── eval/
│   ├── dataset.json            # Q&A eval dataset (5 domain examples)
│   └── run_eval.py             # CLI eval script (used by CI)
├── .github/
│   └── workflows/
│       └── eval.yml            # PR eval + comment + quality gate workflow
├── examples/
│   └── demo_agent.py           # Sample LangGraph agent with LLMWatch
├── tests/
│   └── test_callback.py        # Unit tests: cost calc, token extraction, DB, alerts
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

### Wrap your agent (2 lines)

```python
from llmwatch import LLMWatchCallback

callback = LLMWatchCallback(
    project="my-agent",
    budget_limit_usd=5.00,     # alert when cumulative spend exceeds $5
    latency_alert_ms=3000,     # alert when a single call takes > 3s
)

chain.invoke(
    {"input": "What is the revenue for Q3?"},
    config={"callbacks": [callback]}
)
```

### Print summary after a run

```python
summary = callback.summary()
print(f"Avg latency : {summary['avg_latency_ms']:.0f} ms")
print(f"P95 latency : {summary['p95_ms']:.0f} ms")
print(f"Total cost  : ${summary['total_cost_usd']:.5f}")
print(f"Input tokens: {summary['total_input_tokens']:,}")
```

### Launch dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard live at: `http://localhost:8501`

---

## Latency Tracking

LLMWatch records wall-clock latency (ms) for every LLM call using `time.perf_counter()`. Latencies are stored in SQLite and aggregated on-the-fly.

| Metric | Description |
|--------|-------------|
| P50 (median) | Half of calls complete faster than this |
| P95 | 95% of calls complete faster than this — the SLO target |
| P99 | Worst-case tail latency |
| Avg Latency | Mean response time |

**Alert example** (printed to stdout):
```
[LLMWatch] LATENCY ALERT: 4821.3ms exceeds threshold 3000ms
```

**Per-model breakdown in dashboard:**

| Model | Queries | Avg Latency |
|-------|---------|-------------|
| gpt-4o | 120 | 1,840 ms |
| gpt-4o-mini | 340 | 620 ms |
| llama3-70b | 80 | 510 ms |

---

## Cost & Token Tracking

### How cost is calculated

```
cost = (input_tokens / 1000) × price_per_1k_input
     + (output_tokens / 1000) × price_per_1k_output
```

### Supported model pricing (per 1K tokens, USD)

| Model | Input | Output |
|-------|-------|--------|
| gpt-4o | $0.0050 | $0.0150 |
| gpt-4o-mini | $0.000150 | $0.000600 |
| gpt-4-turbo | $0.0100 | $0.0300 |
| gpt-3.5-turbo | $0.000500 | $0.001500 |
| claude-opus-4-7 | $0.0150 | $0.0750 |
| claude-sonnet-4-6 | $0.0030 | $0.0150 |
| claude-haiku-4-5 | $0.000250 | $0.001250 |
| llama3-70b (Groq) | $0.000590 | $0.000790 |
| gemini-1.5-pro | $0.001250 | $0.005000 |
| gemini-1.5-flash | $0.000075 | $0.000300 |

Unknown models fall back to a generic rate of `$0.002 / $0.006`.

### Token split (prompt vs completion)

The dashboard shows a stacked area chart of input vs output tokens over time — so you can see if your prompts are getting bloated or output length is spiking.

**Budget alert example:**
```
[LLMWatch] BUDGET ALERT: $1.0023 exceeds limit $1.00
```

### Cost breakdown via code

```python
from llmwatch.metrics import cost_breakdown

breakdown = cost_breakdown("gpt-4o", input_tokens=1500, output_tokens=400)
# {
#   "model": "gpt-4o",
#   "input_tokens": 1500,
#   "output_tokens": 400,
#   "total_tokens": 1900,
#   "input_cost_usd": 0.0075,
#   "output_cost_usd": 0.006,
#   "total_cost_usd": 0.0135,
#   "price_per_1k_input": 0.005,
#   "price_per_1k_output": 0.015
# }
```

---

## Dashboard Metrics

| Metric | Description |
|--------|-------------|
| Total Queries | Count of LLM invocations tracked |
| Cost / Query | Average USD cost per invocation |
| Total Cost (USD) | Cumulative spend across all queries |
| P50 Latency | Median response time in ms |
| P95 Latency | 95th percentile response time in ms |
| Error Rate | % of failed invocations |
| Token Usage Chart | Input vs output token split over time |
| Cost by Model | Per-model spend and latency table |

---

## GitHub Actions — Auto Eval on PR

```yaml
# .github/workflows/eval.yml
on: [pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - run: python eval/run_eval.py --output-comment pr_comment.md
      - uses: actions/github-script   # posts score as PR comment
        if: always()
```

Example PR comment auto-posted:

```
## LLMWatch Evaluation Results

| Metric            | Score | Status |
|-------------------|-------|--------|
| Faithfulness      | 0.91  | ✅     |
| Answer Relevancy  | 0.88  | ✅     |
| Context Precision | 0.84  | ✅     |
| Overall           | 0.88  | ✅     |

Threshold: 0.80  |  Result: PASSED ✅
```

---

## Environment Variables

```env
OPENAI_API_KEY=your_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=llmwatch-demo
LANGCHAIN_TRACING_V2=true

LLMWATCH_DB=llmwatch.db
LLMWATCH_PROJECT=default
EVAL_SCORE_THRESHOLD=0.80
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Roadmap

- [ ] Slack/email alert when score drops
- [ ] Multi-model comparison (GPT-4o vs Groq vs Gemini) side-by-side
- [ ] Cost forecasting based on usage trends
- [ ] Export reports as PDF
- [ ] Deploy dashboard to AWS ECS (Fargate)
- [ ] pgvector / PostgreSQL backend for production scale

---

## Connect

**Priyabrat Dalbehera** — AI Engineer | Building production GenAI systems

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/priyabrat-dalbehera-p521)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=flat&logo=vercel&logoColor=white)](https://www.aiwithpriyabrat.com/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/iampriyabrat14)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:ipriyabrat689@gmail.com)