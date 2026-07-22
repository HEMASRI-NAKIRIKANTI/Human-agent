"""
app.py
------
Streamlit UI for the Humana Agent agentic RAG system.

Pages
─────
  📤 Upload Documents   — drag-and-drop PDF ingestion with real-time progress
  💬 Chat               — conversational interface with inline citations
  📚 Document Registry  — overview of all ingested documents

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Inject Streamlit Cloud secrets into os.environ (no-op on local dev) ──────
try:
    for _key, _val in st.secrets.items():
        if isinstance(_val, str):
            os.environ.setdefault(_key, _val)
except Exception:
    pass  # secrets not available locally — .env is used instead

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Humana Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (after page config) ─────────────────────────────────────────
from agents.graph import run_query  # noqa: E402
from ingestion.pipeline import IngestionRegistry, delete_document_pipeline, run_ingestion_pipeline  # noqa: E402
from ingestion.vector_store import get_vector_store  # noqa: E402
from utils.config_loader import get_config  # noqa: E402

config = get_config()


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers  (defined early so all page blocks can call them)
# ══════════════════════════════════════════════════════════════════════════════

def _render_citations(citations: list) -> None:
    """Render a formatted, collapsed citations panel."""
    if not citations:
        return
    with st.expander(
        f"📎 Sources  ({len(citations)} unique page{'s' if len(citations) != 1 else ''})",
        expanded=True,
    ):
        for cit in citations:
            st.markdown(
                f'<div class="citation-card">'
                f'📄 <strong>{cit["filename"]}</strong> &nbsp;—&nbsp; '
                f'Page {cit["page_number"]} &nbsp; '
                f'<em>(relevance {cit["score"]:.1%})</em>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── General ── */
        [data-testid="stSidebar"] { background: #0d1b2a; }
        [data-testid="stSidebar"] * { color: #e0e6f0 !important; }

        /* ── Citation cards ── */
        .citation-card {
            background: #f0f5ff;
            border-left: 4px solid #2563eb;
            border-radius: 6px;
            padding: 0.55rem 0.9rem;
            margin: 0.35rem 0;
            font-size: 0.85rem;
            color: #1e3a5f;
        }
        .citation-card strong { color: #1d4ed8; }

        /* ── Status badges ── */
        .badge-success { color: #15803d; font-weight: 600; }
        .badge-skip    { color: #92400e; font-weight: 600; }
        .badge-error   { color: #b91c1c; font-weight: 600; }

        /* ── Metric cards ── */
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1rem;
        }

        /* ── Chat input ── */
        [data-testid="stChatInput"] textarea { font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 Humana Agent")
    st.markdown("*Enterprise Agentic RAG*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📤 Upload Documents", "💬 Chat", "📚 Document Registry"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Active Configuration**")
    st.code(
        f"LLM     : {config.llm.provider} / {config.llm.model}\n"
        f"Embed   : {config.embeddings.model}\n"
        f"VectorDB: {config.vector_store.provider}\n"
        f"Top-K   : {config.retriever.top_k}",
        language="yaml",
    )

    st.divider()
    try:
        store = get_vector_store()
        chunk_count = store.get_chunk_count()
        doc_count = len(store.get_all_documents_metadata())
        st.metric("Documents", doc_count)
        st.metric("Chunks indexed", chunk_count)
    except Exception:
        st.caption("Vector store not yet initialised.")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Upload Documents
# ══════════════════════════════════════════════════════════════════════════════
if page == "📤 Upload Documents":
    st.title("📤 Upload Documents")
    st.markdown(
        "Upload one or more PDF documents. "
        "Already-ingested files are detected automatically via SHA-256 hash "
        "and skipped — no duplicates will ever enter the vector store."
    )

    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")

        if st.button("🚀 Ingest Selected Documents", type="primary", use_container_width=True):
            for uploaded_file in uploaded_files:
                st.markdown(f"---\n#### 📄 {uploaded_file.name}")
                progress_bar = st.progress(0.0, text="Initialising…")
                status_box = st.empty()

                def _make_callback(pb=progress_bar, sb=status_box):
                    def _cb(msg: str, pct: float) -> None:
                        pb.progress(pct, text=msg)
                        sb.info(msg)
                    return _cb

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    result = run_ingestion_pipeline(
                        pdf_path=tmp_path,
                        filename=uploaded_file.name,
                        progress_callback=_make_callback(),
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                status = result["status"]

                if status == "success":
                    progress_bar.progress(1.0, text="Done")
                    status_box.empty()
                    elapsed = result.get("elapsed_seconds", 0)
                    st.success(
                        f"✅ **{uploaded_file.name}** ingested in {elapsed}s — "
                        f"{result['pages']} pages · "
                        f"{result['chunks_created']} chunks created · "
                        f"{result['chunks_added']} new chunks added"
                    )
                elif status == "skipped":
                    progress_bar.progress(1.0, text="Skipped")
                    status_box.empty()
                    st.warning(
                        f"⚠️ **{uploaded_file.name}** skipped — {result['reason']}"
                    )
                elif status == "warning":
                    progress_bar.progress(1.0, text="Warning")
                    status_box.empty()
                    st.warning(f"⚠️ **{uploaded_file.name}**: {result['reason']}")
                else:
                    progress_bar.progress(1.0, text="Failed")
                    status_box.empty()
                    reason = result.get("reason", "Unknown error")
                    # Surface a friendly hint for billing quota errors
                    if "insufficient_quota" in reason or "quota exhausted" in reason.lower():
                        st.error(
                            f"❌ **{uploaded_file.name}** — OpenAI quota exhausted.\n\n"
                            "Please add credits at https://platform.openai.com/account/billing "
                            "and then re-upload the document."
                        )
                    else:
                        st.error(f"❌ **{uploaded_file.name}** failed — {reason}")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Chat
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Chat":
    st.title("💬 Chat")

    col_title, col_clear = st.columns([8, 1])
    with col_clear:
        if st.button("🗑️ Clear", help="Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    # ── Render message history ────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("citations"):
                _render_citations(msg["citations"])

            if msg["role"] == "assistant" and msg.get("elapsed"):
                st.caption(f"⏱ {msg['elapsed']}s")

    # ── Chat input ────────────────────────────────────────────────────────────
    if query := st.chat_input("Ask a question about your documents…"):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating response…"):
                try:
                    # Build history from current session (exclude the just-added user msg)
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                        if m["role"] in ("user", "assistant")
                    ]

                    t0 = time.perf_counter()
                    result = run_query(query, chat_history=history)
                    elapsed = round(time.perf_counter() - t0, 2)

                    response_text: str = result.get("final_response", "No response generated.")
                    citations = result.get("citations", [])
                    retry_count: int = result.get("retry_count", 0)
                    validation_reason: str = result.get("validation_reason", "")

                    st.markdown(response_text)
                    _render_citations(citations)

                    # ── Response metadata bar ─────────────────────────────────
                    st.caption(
                        f"⏱ {elapsed}s &nbsp;·&nbsp; "
                        f"📄 {len(result.get('retrieved_docs', []))} chunks retrieved &nbsp;·&nbsp; "
                        f"🔁 {retry_count} generator attempt(s)"
                    )

                    # Agent trace (collapsed by default)
                    with st.expander("🔍 Agent Trace", expanded=False):
                        st.json(
                            {
                                "retriever_chunks": len(result.get("retrieved_docs", [])),
                                "generator_attempts": retry_count,
                                "validation_result": validation_reason,
                                "is_valid": result.get("is_valid"),
                                "response_time_s": elapsed,
                                "session_history_turns": len(history),
                            }
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response_text,
                            "citations": citations,
                            "elapsed": elapsed,
                        }
                    )

                except Exception as exc:
                    st.error(f"An error occurred: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Document Registry
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Document Registry":
    st.title("📚 Document Registry")
    st.markdown("All documents that have been ingested into the knowledge base.")

    try:
        store = get_vector_store()
        total_chunks = store.get_chunk_count()
        docs_meta = store.get_all_documents_metadata()

        registry = IngestionRegistry(config.ingestion.registry_path)
        registry_data = registry.get_all()

        # ── Summary metrics ───────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", len(docs_meta))
        c2.metric("Total Chunks", total_chunks)
        total_pages = sum(d.get("total_pages", 0) for d in docs_meta)
        c3.metric("Total Pages", total_pages)
        avg_chunks = round(total_chunks / max(len(docs_meta), 1))
        c4.metric("Avg Chunks / Doc", avg_chunks)

        st.divider()

        if not docs_meta:
            st.info("No documents have been ingested yet. Use the **Upload Documents** tab to add PDFs.")
        else:
            for doc in docs_meta:
                fname = doc["filename"]
                reg_info = registry_data.get(fname, {})
                confirm_key = f"confirm_delete_{fname}"

                # Initialise confirmation flag
                if confirm_key not in st.session_state:
                    st.session_state[confirm_key] = False

                with st.expander(f"📄 {fname}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Pages:** {doc.get('total_pages', '—')}")
                        st.write(f"**Chunks added:** {reg_info.get('chunks_added', '—')}")
                    with col2:
                        ingested_at = doc.get("ingested_at", "")
                        st.write(f"**Ingested:** {ingested_at[:19].replace('T', ' ') if ingested_at else '—'}")
                        st.write(f"**File hash:** `{doc.get('file_hash', '—')[:16]}…`")
                    with col3:
                        st.write(f"**Source path:** {doc.get('source_path', '—')}")

                    st.divider()

                    # ── Delete button + confirmation ──────────────────────────
                    if not st.session_state[confirm_key]:
                        if st.button(
                            "🗑️ Delete Document",
                            key=f"del_btn_{fname}",
                            help="Remove this document and all its chunks from the system",
                        ):
                            st.session_state[confirm_key] = True
                            st.rerun()
                    else:
                        st.warning(
                            f"⚠️ This will permanently remove **{fname}** and all its indexed chunks. "
                            "This cannot be undone."
                        )
                        c_yes, c_no = st.columns(2)
                        with c_yes:
                            if st.button("✅ Yes, delete", key=f"yes_{fname}", type="primary"):
                                with st.spinner(f"Deleting {fname}…"):
                                    del_result = delete_document_pipeline(fname)
                                st.session_state[confirm_key] = False
                                if del_result["status"] == "success":
                                    st.success(
                                        f"Deleted **{fname}** — "
                                        f"{del_result['chunks_removed']} chunks removed."
                                    )
                                else:
                                    st.error(f"Delete failed: {del_result.get('reason')}")
                                st.rerun()
                        with c_no:
                            if st.button("❌ Cancel", key=f"no_{fname}"):
                                st.session_state[confirm_key] = False
                                st.rerun()

    except Exception as exc:
        st.error(f"Could not load the document registry: {exc}")


