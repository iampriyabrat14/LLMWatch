"""
Demo: LangGraph agent with LLMWatch monitoring using Groq (llama3).

Shows latency, cost, and token tracking in 2 lines.
"""
import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llmwatch import LLMWatchCallback

load_dotenv()


class AgentState(TypedDict):
    messages: List[dict]
    query: str
    answer: str


def build_agent(callback: LLMWatchCallback):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        callbacks=[callback],
    )

    def analyst_node(state: AgentState) -> AgentState:
        response = llm.invoke([HumanMessage(content=state["query"])])
        return {**state, "answer": response.content}

    graph = StateGraph(AgentState)
    graph.add_node("analyst", analyst_node)
    graph.set_entry_point("analyst")
    graph.add_edge("analyst", END)
    return graph.compile()


def main():
    # ── 2-line integration ──────────────────────────────────────────────────
    callback = LLMWatchCallback(
        project="demo-agent",
        budget_limit_usd=1.00,
        latency_alert_ms=5000,
    )
    # ───────────────────────────────────────────────────────────────────────

    agent = build_agent(callback)

    queries = [
        "What is the total revenue for a company that sold 500 units at $40 each?",
        "Summarize the key risks of investing in emerging markets in 3 bullet points.",
        "What are the top 3 benefits of using LangGraph over plain LangChain?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        result = agent.invoke({"messages": [], "query": query, "answer": ""})
        print(f"Answer: {result['answer'][:300]}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("LLMWatch Summary")
    print("=" * 60)
    summary = callback.summary()
    print(f"  Total queries  : {summary['total_queries']}")
    print(f"  Avg latency    : {summary['avg_latency_ms']:.0f} ms")
    print(f"  P50 latency    : {summary['p50_ms']:.0f} ms")
    print(f"  P95 latency    : {summary['p95_ms']:.0f} ms")
    print(f"  Total cost     : ${summary['total_cost_usd']:.6f}")
    print(f"  Avg cost/query : ${summary['avg_cost_usd']:.6f}")
    print(f"  Input tokens   : {summary['total_input_tokens']:,}")
    print(f"  Output tokens  : {summary['total_output_tokens']:,}")
    print(f"  Error rate     : {summary['error_rate']:.1f}%")
    print("\nDashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
