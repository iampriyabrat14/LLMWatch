"""
LLMWatch RAG Interactive Session
- Upload your CSV or PDF
- Query it in plain English
- See live latency, cost, token breakdown + RAGAS score per query
"""
import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llmwatch import LLMWatchCallback
from llmwatch.metrics import cost_breakdown
from llmwatch.inline_eval import evaluate_query

load_dotenv()


# ── Document loader ────────────────────────────────────────────────────────────

def load_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv or .pdf")
    return loader.load()


def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"  Indexed {len(chunks)} chunks from {len(docs)} page(s)/row(s).")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def retrieve_context(vectorstore, query: str, k: int = 4) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


def retrieve_context_list(vectorstore, query: str, k: int = 4):
    docs = vectorstore.similarity_search(query, k=k)
    return [d.page_content for d in docs]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  LLMWatch RAG — Query Your CSV / PDF")
    print("=" * 62)

    # ── Step 1: file path ──────────────────────────────────────────
    while True:
        file_path = input("\nEnter path to your CSV or PDF file:\n> ").strip().strip('"')
        if os.path.exists(file_path):
            break
        print(f"  File not found: {file_path}. Try again.")

    print(f"\nLoading '{os.path.basename(file_path)}'...")
    try:
        docs = load_file(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("Building vector index (first time ~10 seconds)...")
    vectorstore = build_vectorstore(docs)
    print("Ready!\n")

    # ── Step 2: LLMWatch + LLM ────────────────────────────────────
    callback = LLMWatchCallback(
        project="rag-session",
        db_path="rag_session.db",
        budget_limit_usd=0.50,
        latency_alert_ms=8000,
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        callbacks=[callback],
    )

    # Separate LLM instance for RAGAS (no callback — avoid double counting)
    eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    print("Commands: type question | 'summary' | 'exit'")
    print("-" * 62)

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
            print(f"  Queries        : {s['total_queries']}")
            print(f"  Avg latency    : {s['avg_latency_ms']:.0f} ms")
            print(f"  P50 latency    : {s['p50_ms']:.0f} ms")
            print(f"  P95 latency    : {s['p95_ms']:.0f} ms")
            print(f"  Total cost     : ${s['total_cost_usd']:.6f}")
            print(f"  Avg cost/query : ${s['avg_cost_usd']:.6f}")
            print(f"  Input tokens   : {s['total_input_tokens']:,}")
            print(f"  Output tokens  : {s['total_output_tokens']:,}")
            print(f"  Avg RAGAS      : {s['avg_ragas_score']:.4f}")
            print(f"  Error rate     : {s['error_rate']:.1f}%")
            continue

        query_num += 1

        try:
            # Retrieve context
            context_list = retrieve_context_list(vectorstore, user_input, k=4)
            context_str  = "\n\n".join(context_list)

            # Generate answer
            messages = [
                SystemMessage(content=(
                    "You are a helpful assistant. Answer using ONLY the context below.\n\n"
                    f"CONTEXT:\n{context_str}"
                )),
                HumanMessage(content=user_input),
            ]
            response = llm.invoke(messages)
            answer = response.content

            # ── Inline RAGAS evaluation ────────────────────────────
            print("  Evaluating with RAGAS...", end="", flush=True)
            ragas_scores = evaluate_query(
                question=user_input,
                answer=answer,
                contexts=context_list,
                llm=eval_llm,
            )
            # Store RAGAS score in DB
            callback.db.update_last_ragas_score("rag-session", ragas_scores["overall"])
            print("\r", end="")  # clear the evaluating line

            # ── Display results ────────────────────────────────────
            rows = callback.db.get_metrics("rag-session", limit=1)
            if rows:
                m  = rows[0]
                bd = cost_breakdown(m["model"], m["input_tokens"], m["output_tokens"])

                ragas_bar = _score_bar(ragas_scores["overall"])

                print(f"\nAnswer: {answer}\n")
                print(f"  ┌─ Query #{query_num} Metrics ──────────────────────────────┐")
                print(f"  │  Latency           : {m['latency_ms']:.0f} ms")
                print(f"  │  Input tokens      : {m['input_tokens']:,}  (prompt + context)")
                print(f"  │  Output tokens     : {m['output_tokens']:,}")
                print(f"  │  Total tokens      : {bd['total_tokens']:,}")
                print(f"  │  Cost this query   : ${bd['total_cost_usd']:.6f}")
                print(f"  ├─ RAGAS Scores ────────────────────────────────────┤")
                print(f"  │  Faithfulness      : {ragas_scores.get('faithfulness', 'N/A')}")
                print(f"  │  Context Relevance : {ragas_scores.get('context_relevance', 'N/A')}")
                print(f"  │  Overall RAGAS     : {ragas_scores['overall']:.4f}  {ragas_bar}")
                print(f"  └───────────────────────────────────────────────────┘")

        except Exception as e:
            print(f"\nError: {e}")

    # Final summary
    s = callback.summary()
    if s["total_queries"] > 0:
        print("\n" + "=" * 62)
        print("  Final Session Summary")
        print("=" * 62)
        print(f"  Total queries  : {s['total_queries']}")
        print(f"  Total cost     : ${s['total_cost_usd']:.6f}")
        print(f"  Avg latency    : {s['avg_latency_ms']:.0f} ms")
        print(f"  P95 latency    : {s['p95_ms']:.0f} ms")
        print(f"  Input tokens   : {s['total_input_tokens']:,}")
        print(f"  Output tokens  : {s['total_output_tokens']:,}")
        print(f"  Avg RAGAS      : {s['avg_ragas_score']:.4f}")
        print()
        print("View full dashboard:")
        print("  streamlit run dashboard/app.py")
        print("  Set Project='rag-session'  DB='rag_session.db'")


def _score_bar(score: float) -> str:
    filled = int(score * 10)
    bar = "█" * filled + "░" * (10 - filled)
    label = "GOOD" if score >= 0.75 else "FAIR" if score >= 0.50 else "LOW"
    return f"[{bar}] {label}"


if __name__ == "__main__":
    main()
