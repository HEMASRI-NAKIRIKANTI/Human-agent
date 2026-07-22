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
        /* ══ Google Font ══════════════════════════════════════════════════════ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ══ Root / body ══════════════════════════════════════════════════════ */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 960px; }

        /* ══ Sidebar ══════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid #334155;
        }
        [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: #f1f5f9 !important; letter-spacing: -0.3px; }
        [data-testid="stSidebar"] hr { border-color: #334155 !important; }
        /* active nav item highlight */
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: rgba(37,99,235,0.25);
            border-radius: 8px;
            padding: 4px 8px;
        }
        /* config code block in sidebar */
        [data-testid="stSidebar"] code {
            background: rgba(255,255,255,0.07) !important;
            border-radius: 6px;
            font-size: 0.78rem !important;
            color: #93c5fd !important;
        }
        /* sidebar metric cards */
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: rgba(255,255,255,0.06) !important;
            border: 1px solid #334155 !important;
            border-radius: 10px;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #60a5fa !important; }

        /* ══ Page titles ══════════════════════════════════════════════════════ */
        h1 { color: #0f172a !important; font-weight: 700 !important; letter-spacing: -0.5px; }
        h2 { color: #1e40af !important; font-weight: 600 !important; }
        h3 { color: #1e3a8a !important; font-weight: 600 !important; }

        /* ══ Metric cards (main area) ═════════════════════════════════════════ */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 1px 4px rgba(37,99,235,0.08);
        }
        [data-testid="stMetricLabel"]  { color: #64748b !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.5px; }
        [data-testid="stMetricValue"]  { color: #1d4ed8 !important; font-weight: 700 !important; }

        /* ══ Citation cards ═══════════════════════════════════════════════════ */
        .citation-card {
            background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            margin: 0.4rem 0;
            font-size: 0.84rem;
            color: #1e3a5f;
            box-shadow: 0 1px 3px rgba(59,130,246,0.1);
            transition: box-shadow 0.2s;
        }
        .citation-card:hover { box-shadow: 0 2px 8px rgba(59,130,246,0.2); }
        .citation-card strong { color: #1d4ed8; }

        /* ══ Chat messages ════════════════════════════════════════════════════ */
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            margin-bottom: 0.5rem;
        }
        [data-testid="stChatInput"] > div {
            border: 2px solid #bfdbfe;
            border-radius: 14px;
            background: #ffffff;
            box-shadow: 0 2px 8px rgba(37,99,235,0.08);
            transition: border-color 0.2s;
        }
        [data-testid="stChatInput"] > div:focus-within { border-color: #3b82f6; }
        [data-testid="stChatInput"] textarea { font-size: 1rem; }

        /* ══ Buttons ══════════════════════════════════════════════════════════ */
        [data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            border: none;
            border-radius: 10px;
            font-weight: 600;
            letter-spacing: 0.2px;
            box-shadow: 0 2px 8px rgba(37,99,235,0.35);
            transition: box-shadow 0.2s, transform 0.1s;
        }
        [data-testid="stButton"] > button[kind="primary"]:hover {
            box-shadow: 0 4px 16px rgba(37,99,235,0.45);
            transform: translateY(-1px);
        }
        [data-testid="stButton"] > button:not([kind="primary"]) {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            font-weight: 500;
            transition: background 0.15s;
        }
        [data-testid="stButton"] > button:not([kind="primary"]):hover { background: #f1f5f9; }

        /* ══ File uploader ════════════════════════════════════════════════════ */
        [data-testid="stFileUploader"] {
            border: 2px dashed #93c5fd;
            border-radius: 14px;
            background: #f0f7ff;
            padding: 1rem;
            transition: border-color 0.2s, background 0.2s;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #3b82f6;
            background: #e0eeff;
        }

        /* ══ Progress bar ════════════════════════════════════════════════════ */
        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            border-radius: 99px;
        }
        [data-testid="stProgress"] { border-radius: 99px; }

        /* ══ Expanders ════════════════════════════════════════════════════════ */
        [data-testid="stExpander"] {
            border: 1px solid #dde8ff !important;
            border-radius: 10px !important;
            background: #fafcff;
            box-shadow: 0 1px 3px rgba(37,99,235,0.05);
        }
        [data-testid="stExpander"] summary {
            font-weight: 600;
            color: #1e3a8a;
        }

        /* ══ Alert / info / success boxes ════════════════════════════════════ */
        [data-testid="stAlert"][data-baseweb="notification"] {
            border-radius: 10px !important;
        }

        /* ══ Divider ══════════════════════════════════════════════════════════ */
        hr { border-color: #e2e8f0 !important; }

        /* ══ Caption / small text ════════════════════════════════════════════ */
        [data-testid="stCaptionContainer"] { color: #94a3b8 !important; font-size: 0.78rem !important; }
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
    st.markdown(
        """
        <div style="padding:1.1rem 0.5rem 0.8rem; text-align:center;">
            <div style="font-size:2.2rem; margin-bottom:0.2rem;">🏥</div>
            <div style="font-size:1.25rem; font-weight:700; color:#f1f5f9;
                        letter-spacing:-0.4px;">Humana Agent</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.15rem;
                        text-transform:uppercase; letter-spacing:0.8px;">
                Enterprise Agentic RAG
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
    st.markdown(
        """<div style="background:linear-gradient(135deg,#1d4ed8 0%,#2563eb 60%,#3b82f6 100%);
            border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.5rem;
            box-shadow:0 4px 20px rgba(37,99,235,0.3);">
            <h1 style="color:#fff!important;margin:0;font-size:1.7rem;">📤 Upload Documents</h1>
            <p style="color:#bfdbfe;margin:0.3rem 0 0;font-size:0.9rem;">
                Drag-and-drop PDFs to ingest them into the knowledge base.
                Duplicate files are detected automatically via SHA-256 hash.
            </p></div>""",
        unsafe_allow_html=True,
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
    st.markdown(
        """<div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 60%,#1d4ed8 100%);
            border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.5rem;
            box-shadow:0 4px 20px rgba(15,23,42,0.3);">
            <h1 style="color:#fff!important;margin:0;font-size:1.7rem;">💬 Chat</h1>
            <p style="color:#93c5fd;margin:0.3rem 0 0;font-size:0.9rem;">
                Ask questions about your uploaded documents.
                Responses include inline citations and relevance scores.
            </p></div>""",
        unsafe_allow_html=True,
    )

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
    st.markdown(
        """<div style="background:linear-gradient(135deg,#064e3b 0%,#065f46 60%,#047857 100%);
            border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.5rem;
            box-shadow:0 4px 20px rgba(6,78,59,0.3);">
            <h1 style="color:#fff!important;margin:0;font-size:1.7rem;">📚 Document Registry</h1>
            <p style="color:#6ee7b7;margin:0.3rem 0 0;font-size:0.9rem;">
                All documents indexed in the knowledge base.
                Expand any row to view details or remove the document.
            </p></div>""",
        unsafe_allow_html=True,
    )

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


