import os
import sys
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(_env_path, override=True)

# Manually parse and inject .env — guarantees Streamlit picks up the vars
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llmwatch.db import MetricsDB
from llmwatch import LLMWatchCallback
from llmwatch.metrics import cost_breakdown
from llmwatch.inline_eval import evaluate_query

st.set_page_config(page_title="LLMWatch Dashboard", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("LLMWatch")
    db_path = st.text_input("DB path", value="rag_ui.db")
    project = st.text_input("Project", value="rag-ui")
    st.markdown("---")
    if st.button("🔄 Refresh Metrics"):
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_query, tab_monitor = st.tabs(["📂 Upload & Query", "📊 Monitor Dashboard"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Query
# ═══════════════════════════════════════════════════════════════════════════════
with tab_query:
    st.header("Upload Your File & Ask Questions")

    uploaded_file = st.file_uploader(
        "Upload CSV or PDF",
        type=["csv", "pdf"],
        help="Upload any CSV or PDF — your data stays local",
    )

    if uploaded_file:
        if st.session_state.file_name != uploaded_file.name:
            with st.spinner(f"Loading '{uploaded_file.name}' and building vector index..."):
                from langchain_community.document_loaders import PyPDFLoader, CSVLoader
                from langchain_community.vectorstores import FAISS
                from langchain_huggingface import HuggingFaceEmbeddings
                from langchain.text_splitter import RecursiveCharacterTextSplitter

                suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".csv"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                if suffix == ".pdf":
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(tmp_path)
                else:
                    from langchain_community.document_loaders import CSVLoader
                    loader = CSVLoader(tmp_path, encoding="utf-8")

                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.file_name = uploaded_file.name
                st.session_state.chat_history = []
                os.unlink(tmp_path)

            st.success(f"✅ Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

        # ── Query input ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader(f"Ask questions about: `{st.session_state.file_name}`")

        with st.form("query_form", clear_on_submit=True):
            user_query = st.text_input("Your question", placeholder="e.g. Which route had the highest revenue?")
            submitted  = st.form_submit_button("Ask →")

        if submitted and user_query.strip():
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage, SystemMessage

            with st.spinner("Generating answer + evaluating with RAGAS..."):
                vs = st.session_state.vectorstore
                docs_retrieved = vs.similarity_search(user_query, k=4)
                context_list = [d.page_content for d in docs_retrieved]
                context_str  = "\n\n".join(context_list)

                groq_api_key = os.getenv("GROQ_API_KEY")
                callback = LLMWatchCallback(project=project, db_path=db_path)
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,
                               groq_api_key=groq_api_key, callbacks=[callback])

                messages = [
                    SystemMessage(content=(
                        "You are a helpful assistant. Answer using ONLY the context below.\n\n"
                        f"CONTEXT:\n{context_str}"
                    )),
                    HumanMessage(content=user_query),
                ]
                response = llm.invoke(messages)
                answer   = response.content

                ragas = evaluate_query(user_query, answer, context_list)
                callback.db.update_last_ragas_score(project, ragas["overall"])

                rows = callback.db.get_metrics(project, limit=1)
                m    = rows[0] if rows else {}
                bd   = cost_breakdown(m.get("model",""), m.get("input_tokens",0), m.get("output_tokens",0))

                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer":   answer,
                    "latency":  m.get("latency_ms", 0),
                    "tokens":   bd.get("total_tokens", 0),
                    "cost":     bd.get("total_cost_usd", 0),
                    "ragas":    ragas["overall"],
                    "faithful": ragas.get("faithfulness", 0),
                    "ctx_rel":  ragas.get("context_relevance", 0),
                })

        # ── Chat history ───────────────────────────────────────────────────
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            with st.container(border=True):
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer']}")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Latency",    f"{item['latency']:.0f} ms")
                m2.metric("Tokens",     f"{item['tokens']:,}")
                m3.metric("Cost",       f"${item['cost']:.6f}")
                m4.metric("RAGAS",      f"{item['ragas']:.2f}")
                m5.metric("Faithful",   f"{item['faithful']:.2f}")

    else:
        st.info("👆 Upload a CSV or PDF file above to get started.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Monitor Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    db = MetricsDB(db_path)
    summary      = db.get_summary(project)
    metrics_raw  = db.get_metrics(project, limit=500)
    model_breakdown = db.get_cost_by_model(project)

    st.header(f"Monitoring — {project}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", summary.get("total_queries", 0))
    c2.metric("P50 Latency",   f"{summary.get('p50_ms', 0):.0f} ms")
    c3.metric("P95 Latency",   f"{summary.get('p95_ms', 0):.0f} ms")
    c4.metric("Error Rate",    f"{summary.get('error_rate', 0):.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Cost",       f"${summary.get('total_cost_usd', 0):.5f}")
    c6.metric("Avg Cost / Query", f"${summary.get('avg_cost_usd', 0):.5f}")
    avg_ragas   = summary.get("avg_ragas_score", 0) or 0
    ragas_label = "GOOD ✅" if avg_ragas >= 0.75 else ("FAIR ⚠️" if avg_ragas >= 0.50 else "LOW ❌")
    c7.metric("Avg RAGAS",     f"{avg_ragas:.4f}  {ragas_label}")
    c8.metric("Total Tokens",  f"{(summary.get('total_input_tokens',0)+summary.get('total_output_tokens',0)):,}")

    st.markdown("---")

    if not metrics_raw:
        st.info("No data yet — ask questions in the Upload & Query tab first.")
        st.stop()

    df = pd.DataFrame(metrics_raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Latency Over Time (ms)")
        st.line_chart(df.set_index("timestamp")["latency_ms"], height=220)
    with col_right:
        st.subheader("RAGAS Score Per Query")
        ragas_df = df[df["ragas_score"].notna()].set_index("timestamp")[["ragas_score"]]
        if not ragas_df.empty:
            st.line_chart(ragas_df, height=220)
        else:
            st.info("No RAGAS scores yet.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Cost Per Query (USD)")
        st.bar_chart(df.set_index("timestamp")["cost_usd"], height=220)
    with col_b:
        st.subheader("Token Usage — Input vs Output")
        st.area_chart(df[["timestamp","input_tokens","output_tokens"]].set_index("timestamp"), height=220)

    if model_breakdown:
        st.subheader("Breakdown by Model")
        mdf = pd.DataFrame(model_breakdown)
        mdf["avg_cost_usd"]   = mdf["avg_cost_usd"].map("${:.5f}".format)
        mdf["total_cost_usd"] = mdf["total_cost_usd"].map("${:.5f}".format)
        mdf["avg_latency_ms"] = mdf["avg_latency_ms"].map("{:.0f} ms".format)
        st.dataframe(mdf[["model","queries","total_cost_usd","avg_cost_usd",
                           "total_input_tokens","total_output_tokens","avg_latency_ms"]],
                     use_container_width=True)

    ragas_rows = df[df["ragas_score"].notna()][["timestamp","ragas_score","latency_ms","cost_usd","input_tokens","output_tokens"]]
    if not ragas_rows.empty:
        st.subheader("RAGAS Score Detail")
        def color_ragas(val):
            if isinstance(val, float):
                color = "#2ecc71" if val >= 0.75 else ("#f39c12" if val >= 0.50 else "#e74c3c")
                return f"color: {color}; font-weight: bold"
            return ""
        st.dataframe(ragas_rows.style.applymap(color_ragas, subset=["ragas_score"]),
                     use_container_width=True)

    with st.expander("Raw Metrics Table"):
        st.dataframe(df, use_container_width=True)
