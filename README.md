# LLMWatch

> **LLM Observability & Evaluation Platform for Production Agents** — plug-and-play monitoring layer for any LangChain or LangGraph agent. Tracks cost, latency percentiles, token usage, and hallucination rate. Fails your CI if quality drops.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&logoColor=white)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-4B0082?style=flat&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-FF6B35?style=flat&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=githubactions&logoColor=white)

---

## The Problem It Solves

Most teams build agents and assume they work. No cost visibility. No latency tracking. No quality gates. Then production breaks and no one knows why — too expensive? Too slow? Hallucinating?

LLMWatch wraps your existing agent with **2 lines of code** and gives you full observability: cost per query, P50/P95/P99 latency, token usage trends, and RAGAS-based hallucination scoring — all visible in a live dashboard and enforced in CI.

---

## Add to Any Agent in 2 Lines

```python
from llmwatch import LLMWatchCallback

callback = LLMWatchCallback(
    project="my-rag-agent",
    budget_limit_usd=5.00,      # alert when spend exceeds $5
    latency_alert_ms=3000,      # alert when a call takes >3s
)

# Works with any LangChain/LangGraph chain
chain.invoke(
    {"input": "What is the refund policy?"},
    config={"callbacks": [callback]}
)
```

No code changes to your agent logic. Drop in, monitor immediately.

---

## Live Dashboard

```
┌───────────────────────────────────────────────────────────────────┐
│  LLMWatch — Production Agent Monitor           [last 24h ▼]       │
├─────────────────┬─────────────────┬─────────────────┬────────────┤
│  Total Queries  │   Cost / Query  │   P95 Latency   │ RAGAS Score│
│     1,247       │    $0.0023      │    1,840 ms      │  0.87/1.0  │
│   +12% today    │  -8% vs yest.   │  SLO: 2,000ms ✅ │ gate: 0.80 │
├─────────────────┴─────────────────┴─────────────────┴────────────┤
│  Token Usage — Last 24 Hours                                      │
│                                                                   │
│  Input tokens avg:  1,240 / query                                 │
│  ████████████████░░░░░░░░░░░░ (62%)                               │
│                                                                   │
│  Output tokens avg:   380 / query                                 │
│  █████░░░░░░░░░░░░░░░░░░░░░░░ (19%)                               │
├───────────────────────────────────────────────────────────────────┤
│  Latency Percentiles by Model                                     │
│                                                                   │
│  Model             Queries   P50      P95      P99                │
│  ─────────────── ─────────  ──────  ───────  ───────              │
│  gpt-4o              120    1,240ms  1,840ms  3,120ms             │
│  gpt-4o-mini         340      480ms    820ms  1,240ms             │
│  llama3-70b (Groq)    80      390ms    610ms    940ms             │
├───────────────────────────────────────────────────────────────────┤
│  RAGAS Hallucination Trend                [CI gate: fail if <0.80]│
│                                                                   │
│  Faithfulness     ▁▂▄▅▆▇█  0.91  ✅                              │
│  Answer Relevancy ▁▂▃▄▅▅▆  0.88  ✅                              │
│  Context Precision▁▂▃▃▄▅▅  0.84  ✅                              │
│  Overall                    0.88  ✅  PASS                        │
└───────────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    YOUR EXISTING AGENT                             │
│                                                                    │
│   chain.invoke(input, config={"callbacks": [LLMWatchCallback]})   │
│                          │                                         │
│                          ▼                                         │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │               LLMWatchCallback (core)                    │    │
│   │                                                          │    │
│   │  on_llm_start():                                         │    │
│   │    → start timer (perf_counter)                          │    │
│   │    → count input tokens                                  │    │
│   │                                                          │    │
│   │  on_llm_end():                                           │    │
│   │    → stop timer → compute latency                        │    │
│   │    → count output tokens                                 │    │
│   │    → calculate cost from pricing table                   │    │
│   │    → check vs budget_limit + latency_alert               │    │
│   │    → write to SQLite                                     │    │
│   │    → forward trace to LangSmith                          │    │
│   └──────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
   ┌─────────────────┐ ┌──────────────┐ ┌────────────────────┐
   │    LangSmith    │ │   SQLite /   │ │   RAGAS Evaluator  │
   │                 │ │  PostgreSQL  │ │                    │
   │  Full trace:    │ │              │ │  Runs on eval       │
   │  • agent hops   │ │  Stores:     │ │  dataset via CI     │
   │  • tool calls   │ │  • latency   │ │                    │
   │  • token counts │ │  • cost      │ │  Metrics:           │
   │  • latency      │ │  • tokens    │ │  • Faithfulness     │
   │  • errors       │ │  • errors    │ │  • Answer Relevancy │
   └─────────────────┘ └──────┬───────┘ │  • Context Prec.   │
                              │         └────────────────────┘
                              ▼
                   ┌──────────────────┐
                   │ Streamlit        │
                   │ Dashboard        │
                   │                  │
                   │ KPI cards:       │
                   │ cost, latency,   │
                   │ tokens, RAGAS    │
                   │                  │
                   │ Time-series:     │
                   │ trends over time │
                   └──────────────────┘

                   ┌──────────────────────────────────────────┐
                   │  GitHub Actions — Runs on every PR       │
                   │                                          │
                   │  1. python eval/run_eval.py              │
                   │  2. Post RAGAS scores as PR comment      │
                   │  3. Fail CI if score < threshold (0.80)  │
                   └──────────────────────────────────────────┘
```

---

## Cost Calculation

```
Per LLM call:
  cost = (input_tokens  / 1000) × price_per_1k_input
       + (output_tokens / 1000) × price_per_1k_output

Pricing table (USD per 1K tokens):
  ┌────────────────────────┬──────────┬───────────┐
  │ Model                  │ Input    │ Output    │
  ├────────────────────────┼──────────┼───────────┤
  │ gpt-4o                 │ $0.0050  │ $0.0150   │
  │ gpt-4o-mini            │ $0.00015 │ $0.00060  │
  │ claude-sonnet-4-6      │ $0.0030  │ $0.0150   │
  │ claude-haiku-4-5       │ $0.00025 │ $0.00125  │
  │ llama3-70b (Groq)      │ $0.00059 │ $0.00079  │
  │ gemini-1.5-flash       │ $0.00008 │ $0.00030  │
  └────────────────────────┴──────────┴───────────┘
  Unknown models → fallback rate $0.002 / $0.006
```

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

Auto-posted comment on every PR:

```
## LLMWatch Evaluation Results

| Metric             | Score | Status |
|--------------------|-------|--------|
| Faithfulness       | 0.91  |   ✅   |
| Answer Relevancy   | 0.88  |   ✅   |
| Context Precision  | 0.84  |   ✅   |
| Overall            | 0.88  |   ✅   |

Threshold: 0.80  |  Result: PASSED ✅
```

If Overall < 0.80 → CI fails → PR cannot merge.

---

## Latency Tracking

```
Wall-clock latency per LLM call via time.perf_counter()

Metrics stored and aggregated in SQLite:

  P50 (median):  half of calls complete faster than this
  P95:           95% of calls faster — use this as your SLO
  P99:           worst-case tail latency
  Avg:           mean response time

Alert printed to stdout:
  [LLMWatch] LATENCY ALERT: 4821.3ms exceeds threshold 3000ms
```

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Agent Framework | LangGraph, LangChain | Agents being monitored |
| LLM Tracing | LangSmith | Full trace visualization |
| Evaluation | RAGAS | Hallucination + quality scoring |
| Metrics Store | SQLite (dev) / PostgreSQL (prod) | P50/P95/P99, cost, tokens |
| Dashboard | Streamlit | Live metrics + time-series charts |
| CI/CD Eval | GitHub Actions | Auto RAGAS on every PR |
| LLM Support | OpenAI, Anthropic, Groq, Gemini | Multi-model pricing table |

---

## Project Structure

```
LLMWatch/
├── llmwatch/
│   ├── __init__.py
│   ├── callback.py         # LangChain callback handler (the core)
│   ├── metrics.py          # Cost calculation + multi-model pricing table
│   ├── evaluator.py        # RAGAS eval runner + PR comment formatter
│   └── db.py               # SQLite metrics persistence
├── dashboard/
│   └── app.py              # Streamlit dashboard (KPI cards + 4 charts)
├── eval/
│   ├── dataset.json        # Q&A evaluation dataset
│   └── run_eval.py         # CLI eval runner (used by CI)
├── .github/
│   └── workflows/
│       └── eval.yml        # PR eval + comment + quality gate
├── examples/
│   └── demo_agent.py       # Sample LangGraph agent with LLMWatch
├── tests/
│   └── test_callback.py    # Unit tests: cost calc, tokens, DB, alerts
├── requirements.txt
└── .env.example
```

---

## Quick Start

```bash
git clone https://github.com/iampriyabrat14/LLMWatch
cd LLMWatch
pip install -r requirements.txt
cp .env.example .env        # add your API keys

# Run the demo agent with monitoring
python examples/demo_agent.py

# Launch dashboard
streamlit run dashboard/app.py
```

Dashboard: `http://localhost:8501`

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

## Interview Talking Points

**Q: Why is observability important for LLM agents?**
> Agents fail silently. A RAG agent might retrieve wrong chunks and produce a confident-sounding hallucination — and you'd never know without measuring faithfulness scores. LLMWatch catches quality regressions before they reach users, by failing CI if RAGAS scores drop.

**Q: How do you measure hallucination without ground truth?**
> RAGAS Faithfulness metric doesn't need ground truth answers. It compares the generated answer against the *retrieved context* — if the answer makes claims not supported by the retrieved chunks, it's flagged as hallucination. This works at inference time without labeled data.

**Q: What is the latency alert actually doing?**
> The callback records `time.perf_counter()` at `on_llm_start` and `on_llm_end`. Wall-clock delta is the actual user-perceived latency. Stored in SQLite, aggregated with `numpy.percentile` for P50/P95/P99. If any single call exceeds `latency_alert_ms`, it prints a console alert — production systems would route this to PagerDuty or Slack.

---

## Roadmap

- [ ] Slack/email alerts when score drops
- [ ] Multi-model side-by-side comparison (GPT-4o vs Groq vs Gemini)
- [ ] Cost forecasting based on usage trends
- [ ] Export reports as PDF
- [ ] Deploy dashboard to AWS ECS Fargate
- [ ] PostgreSQL backend for production scale

---

## Connect

**Priyabrat Dalbehera** — AI Engineer | Building production GenAI systems

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/priyabrat-dalbehera-p521)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=flat&logo=vercel&logoColor=white)](https://www.aiwithpriyabrat.com/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/iampriyabrat14)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:ipriyabrat689@gmail.com)
