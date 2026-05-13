"""
Interactive LLMWatch session — type your own queries and see
live latency, cost, and token breakdown per query.
"""
import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llmwatch import LLMWatchCallback
from llmwatch.metrics import cost_breakdown

load_dotenv()


def main():
    callback = LLMWatchCallback(
        project="my-session",
        db_path="my_session.db",
        budget_limit_usd=0.50,
        latency_alert_ms=5000,
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        callbacks=[callback],
    )

    print("=" * 60)
    print("  LLMWatch Interactive Session")
    print("  Model : llama-3.3-70b-versatile (Groq)")
    print("  Type your query and press Enter.")
    print("  Type 'summary' to see totals, 'exit' to quit.")
    print("=" * 60)

    query_num = 0

    while True:
        print()
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            break

        if user_input.lower() == "summary":
            s = callback.summary()
            print("\n--- Session Summary ---")
            print(f"  Queries run    : {s['total_queries']}")
            print(f"  Avg latency    : {s['avg_latency_ms']:.0f} ms")
            print(f"  P50 latency    : {s['p50_ms']:.0f} ms")
            print(f"  P95 latency    : {s['p95_ms']:.0f} ms")
            print(f"  Total cost     : ${s['total_cost_usd']:.6f}")
            print(f"  Avg cost/query : ${s['avg_cost_usd']:.6f}")
            print(f"  Input tokens   : {s['total_input_tokens']:,}")
            print(f"  Output tokens  : {s['total_output_tokens']:,}")
            print(f"  Error rate     : {s['error_rate']:.1f}%")
            continue

        query_num += 1
        before_count = len(callback.db.get_metrics("my-session"))

        try:
            response = llm.invoke([HumanMessage(content=user_input)])
            answer = response.content

            # fetch the metric just recorded
            rows = callback.db.get_metrics("my-session", limit=1)
            if rows:
                m = rows[0]
                bd = cost_breakdown(m["model"], m["input_tokens"], m["output_tokens"])
                print(f"\nAnswer: {answer}\n")
                print(f"  ┌─ Query #{query_num} Metrics ─────────────────────┐")
                print(f"  │  Latency       : {m['latency_ms']:.0f} ms")
                print(f"  │  Input tokens  : {m['input_tokens']:,}")
                print(f"  │  Output tokens : {m['output_tokens']:,}")
                print(f"  │  Total tokens  : {bd['total_tokens']:,}")
                print(f"  │  Input cost    : ${bd['input_cost_usd']:.6f}")
                print(f"  │  Output cost   : ${bd['output_cost_usd']:.6f}")
                print(f"  │  Total cost    : ${bd['total_cost_usd']:.6f}")
                print(f"  └──────────────────────────────────────────────┘")

        except Exception as e:
            print(f"\nError: {e}")

    # Final summary on exit
    s = callback.summary()
    print("\n" + "=" * 60)
    print("  Final Session Summary")
    print("=" * 60)
    print(f"  Total queries  : {s['total_queries']}")
    print(f"  Total cost     : ${s['total_cost_usd']:.6f}")
    print(f"  Avg latency    : {s['avg_latency_ms']:.0f} ms")
    print(f"  P95 latency    : {s['p95_ms']:.0f} ms")
    print(f"  Input tokens   : {s['total_input_tokens']:,}")
    print(f"  Output tokens  : {s['total_output_tokens']:,}")
    print("\nView full dashboard: streamlit run dashboard/app.py --server.port 8502")
    print("  (set Project = 'my-session', DB = 'my_session.db')")


if __name__ == "__main__":
    main()
