"""
LLMWatch RAG Demo — Airline Analytics with RAGAS scores
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from rag_interactive import load_file, build_vectorstore, retrieve_context_list
from llmwatch import LLMWatchCallback
from llmwatch.metrics import cost_breakdown
from llmwatch.inline_eval import evaluate_query
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

FILE = r"d:\InterView\All Code Hub\LLMWatch\examples\demo_airline.csv"

QUERIES = [
    "Which route generated the highest revenue overall?",
    "Which month had the best on-time performance on average?",
    "Which route has the worst average delay in minutes?",
    "What is the total revenue across all flights?",
    "Which route has the highest load factor percentage?",
]

print("=" * 62)
print("  LLMWatch RAG Demo — Airline Analytics + RAGAS")
print("=" * 62)

print("\nLoading demo_airline.csv...")
docs = load_file(FILE)
print("Building vector index...")
vs = build_vectorstore(docs)
print()

callback = LLMWatchCallback(project="airline-demo", db_path="airline_demo.db")
llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, callbacks=[callback])
eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

for i, q in enumerate(QUERIES, 1):
    context_list = retrieve_context_list(vs, q, k=6)
    context_str  = "\n\n".join(context_list)

    messages = [
        SystemMessage(content=f"Answer using ONLY this data:\n\n{context_str}"),
        HumanMessage(content=q),
    ]
    response = llm.invoke(messages)
    answer   = response.content

    print(f"Evaluating Q{i} with RAGAS...", end="", flush=True)
    ragas = evaluate_query(q, answer, context_list, llm=eval_llm)
    callback.db.update_last_ragas_score("airline-demo", ragas["overall"])
    print("\r", end="")

    rows = callback.db.get_metrics("airline-demo", limit=1)
    m  = rows[0]
    bd = cost_breakdown(m["model"], m["input_tokens"], m["output_tokens"])

    print(f"Q{i}: {q}")
    print(f"  Answer     : {answer.strip()[:200]}")
    print(f"  Latency    : {m['latency_ms']:.0f} ms  |  Tokens: {bd['total_tokens']}  |  Cost: ${bd['total_cost_usd']:.6f}")
    print(f"  RAGAS      : Faithfulness={ragas.get('faithfulness','N/A')}  ContextRelevance={ragas.get('context_relevance','N/A')}  Overall={ragas['overall']:.4f}")
    print()

s = callback.summary()
print("=" * 62)
print("  Final Summary")
print("=" * 62)
print(f"  Queries        : {s['total_queries']}")
print(f"  Avg latency    : {s['avg_latency_ms']:.0f} ms  |  P95: {s['p95_ms']:.0f} ms")
print(f"  Total cost     : ${s['total_cost_usd']:.6f}  |  Avg: ${s['avg_cost_usd']:.6f}")
print(f"  Input tokens   : {s['total_input_tokens']:,}  |  Output: {s['total_output_tokens']:,}")
print(f"  Avg RAGAS      : {s['avg_ragas_score']:.4f}")
print(f"  Error rate     : {s['error_rate']:.1f}%")
print()
print("Dashboard: streamlit run dashboard/app.py")
print("  Set Project='airline-demo'  DB='airline_demo.db'")
