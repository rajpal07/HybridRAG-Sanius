import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Healthcare Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Tighten Streamlit's default top padding (keeps header controls) ── */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 1rem !important;
    }

    /* ── Main header ─────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0f1b4c 0%, #1a3a8f 60%, #2563eb 100%);
        color: white;
        padding: 1.6rem 2.5rem;
        border-radius: 14px;
        margin-top: 0;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 32px rgba(37,99,235,0.18);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #bfcfff;
        margin: 0.4rem 0 0 0;
        font-size: 0.97rem;
        font-weight: 400;
    }
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        color: #e0e8ff;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 0.6rem;
        letter-spacing: 0.3px;
    }

    /* ── Confidence badges ───────────────────────────── */
    .confidence-high {
        background: linear-gradient(135deg,#16a34a,#22c55e);
        color: #fff;
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.3px;
    }
    .confidence-medium {
        background: linear-gradient(135deg,#d97706,#f59e0b);
        color: #fff;
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.3px;
    }
    .confidence-low {
        background: linear-gradient(135deg,#dc2626,#ef4444);
        color: #fff;
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.3px;
    }

    /* ── Source chips ────────────────────────────────── */
    .source-chip {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 2px 9px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 4px;
    }

    /* ── Status dots ─────────────────────────────────── */
    .status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 0;
        font-size: 0.85rem;
    }
    .dot-ok  { width:9px; height:9px; border-radius:50%; background:#22c55e; flex-shrink:0; box-shadow:0 0 6px rgba(34,197,94,.5); }
    .dot-err { width:9px; height:9px; border-radius:50%; background:#ef4444; flex-shrink:0; box-shadow:0 0 6px rgba(239,68,68,.5); }
    .dot-off { width:9px; height:9px; border-radius:50%; background:#94a3b8; flex-shrink:0; }

    /* ── Data source rows ────────────────────────────── */
    .ds-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 3px 0;
        font-size: 0.82rem;
        border-bottom: 1px solid rgba(0,0,0,0.04);
    }
    .ds-count {
        font-weight: 600;
        color: #1d4ed8;
        font-size: 0.8rem;
    }
    .ds-count-zero { color: #94a3b8; font-weight: 400; }

    /* ── Welcome screen ─────────────────────────────── */
    /* suppress Streamlit focus ring on injected HTML */
    .stMarkdown p, .stMarkdown div { outline: none !important; }

    /* Theme-aware: Streamlit CSS vars work in BOTH dark and light mode */
    .welcome-heading {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.3rem;
    }
    .welcome-sub {
        font-size: 0.87rem;
        color: var(--text-color);
        opacity: 0.6;
        margin-bottom: 1.3rem;
    }
    .ds-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.6rem;
        height: 100%;
    }
    .ds-card-title {
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--primary-color);
        margin-bottom: 0.35rem;
    }
    .ds-card-desc {
        font-size: 0.82rem;
        color: var(--text-color);
        opacity: 0.75;
        line-height: 1.5;
    }
    .starters-heading {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text-color);
        margin: 1.4rem 0 0.6rem;
    }
    .starter-wrap {
        display: flex;
        flex-direction: column;
        min-height: 155px;
        border-radius: 12px;
        border: 1.5px solid;
        overflow: hidden;
        background: var(--secondary-background-color);
    }
    .starter-easy   { border-color: #22c55e; }
    .starter-medium { border-color: #f59e0b; }
    .starter-hard   { border-color: #ef4444; }
    .starter-body {
        flex: 1;
        padding: 0.85rem 1rem 0.75rem;
    }
    .starter-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .label-easy   { color: #22c55e; }
    .label-medium { color: #f59e0b; }
    .label-hard   { color: #ef4444; }
    .starter-text {
        font-size: 0.87rem;
        color: var(--text-color);
        font-weight: 500;
        line-height: 1.45;
    }
</style>
""",
    unsafe_allow_html=True,
)



def api_get(path: str, timeout: float = 10.0) -> dict:
    try:
        response = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                return {"error": f"Backend returned non-JSON response (HTTP 200). Body: {response.text[:200]}"}
        try:
            detail = response.json().get("detail", response.text[:300])
        except Exception:
            detail = response.text[:300] or f"HTTP {response.status_code}"
        return {"error": f"Backend error (HTTP {response.status_code}): {detail}"}
    except httpx.ConnectError:
        return {"error": "Cannot connect to backend API. Is the uvicorn server running on port 8000?"}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def api_post(path: str, payload: dict | None = None, timeout: float = 60.0) -> dict:
    try:
        response = httpx.post(f"{API_BASE}{path}", json=payload or {}, timeout=timeout)
        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                return {"error": f"Backend returned non-JSON response (HTTP 200). Body: {response.text[:200]}"}
        try:
            detail = response.json().get("detail", response.text[:300])
        except Exception:
            detail = response.text[:300] or f"HTTP {response.status_code}"
        return {"error": f"Backend error (HTTP {response.status_code}): {detail}"}
    except httpx.TimeoutException:
        return {"error": "Request timed out. Heavy hybrid and analytics queries can take a few minutes — please retry or simplify the question."}
    except httpx.ConnectError:
        return {"error": "Cannot connect to backend API. Is the uvicorn server running on port 8000?"}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def submit_query_with_polling(question: str, session_id: str | None, status_placeholder) -> dict:
    """Submit a query, get a job_id, then poll until done. Updates status_placeholder with live progress."""
    import time

    # Step 1: submit — should return in < 1 second
    submit_resp = api_post("/query", {"question": question, "session_id": session_id}, timeout=15.0)
    if "error" in submit_resp:
        return submit_resp

    job_id = submit_resp.get("job_id")
    if not job_id:
        return {"error": f"Unexpected response from backend: {submit_resp}"}

    # Step 2: poll /query/status/{job_id} until done or error
    start = time.time()
    poll_interval = 3.0  # seconds between polls
    max_wait = 900.0     # 15 minutes hard ceiling

    while True:
        elapsed = int(time.time() - start)
        status_placeholder.markdown(f"Thinking... ({elapsed}s elapsed)")

        poll = api_get(f"/query/status/{job_id}", timeout=10.0)

        if "error" in poll:
            return poll

        job_status = poll.get("status", "running")

        if job_status == "done":
            return poll.get("result", {"error": "Job done but no result returned"})

        if job_status == "error":
            return {"error": poll.get("error", "Query failed on the server")}

        if time.time() - start > max_wait:
            return {"error": f"Query timed out after {int(max_wait / 60)} minutes. Try a simpler question."}

        time.sleep(poll_interval)


def confidence_badge(level: str) -> str:
    level = (level or "low").lower()
    return f'<span class="confidence-{level}">{level.upper()}</span>'


def load_sessions() -> list[dict]:
    result = api_get("/sessions")
    return result.get("sessions", []) if "error" not in result else []


def create_session(title: str | None = None) -> dict | None:
    result = api_post("/sessions", {"title": title} if title else {}, timeout=20.0)
    return result.get("session") if "error" not in result else None


def load_session_messages(session_id: str) -> list[dict]:
    result = api_get(f"/sessions/{session_id}", timeout=20.0)
    if "error" in result:
        return []
    messages = []
    for message in result.get("messages", []):
        if message["role"] == "assistant" and message.get("answer_payload"):
            messages.append({"role": "assistant", "data": message["answer_payload"]})
        else:
            messages.append({"role": message["role"], "content": message["content"]})
    return messages


def ensure_active_session():
    sessions = load_sessions()
    st.session_state.sessions = sessions

    if not sessions:
        session = create_session()
        if session:
            sessions = [session]
            st.session_state.sessions = sessions

    current_session_id = st.session_state.get("current_session_id")
    if not current_session_id and sessions:
        st.session_state.current_session_id = sessions[0]["id"]

    if st.session_state.get("loaded_session_id") != st.session_state.get("current_session_id"):
        active_id = st.session_state.get("current_session_id")
        st.session_state.messages = load_session_messages(active_id) if active_id else []
        st.session_state.loaded_session_id = active_id


def refresh_current_session():
    active_id = st.session_state.get("current_session_id")
    if not active_id:
        return
    st.session_state.sessions = load_sessions()
    st.session_state.messages = load_session_messages(active_id)
    st.session_state.loaded_session_id = active_id


def render_answer(data: dict):
    st.markdown(data.get("answer", "No answer returned."))

    badge_html = confidence_badge(data.get("confidence", "low"))
    source_chips = " ".join(
        f'<span class="source-chip">{source}</span>'
        for source in data.get("sources_used", [])[:6]
    )
    st.markdown(f"**Confidence:** {badge_html} &nbsp; {source_chips}", unsafe_allow_html=True)

    router = data.get("router", {})
    with st.expander("Reasoning Trace"):
        st.markdown(f"**Query type:** `{router.get('type', 'unknown')}`")
        st.markdown(f"**Router reasoning:** {router.get('reasoning', 'N/A')}")
        st.markdown(f"**Agents run:** {', '.join(data.get('agents_run', []))}")
        st.markdown(f"**Execution flow:** {' -> '.join(data.get('execution_flow', []))}")
        st.markdown(
            f"**Session memory used:** `{data.get('session_context_used', False)}` "
            f"({data.get('session_context_reason', 'n/a')})"
        )
        st.markdown(f"**Requires patient data:** {router.get('requires_patient_data', False)}")
        st.markdown(f"**Requires documents:** {router.get('requires_documents', False)}")

    citations = data.get("citations", [])
    rag_chunks = data.get("rag_chunks", [])
    if citations or rag_chunks:
        with st.expander(f"Sources ({len(citations)} citations, {len(rag_chunks)} document chunks)"):
            if citations:
                st.markdown("**Citations extracted from answer:**")
                for citation in citations:
                    st.markdown(f"- {citation}")
            if rag_chunks:
                st.markdown("**Retrieved chunks:**")
                for chunk in rag_chunks:
                    st.markdown(
                        f"**{chunk.get('source', 'unknown')}** ({chunk.get('doc_type', '')}, similarity={chunk.get('similarity', 0):.3f})"
                    )
                    st.markdown(f"> {(chunk.get('content', '')[:300])}...")

    if data.get("sql_used"):
        with st.expander("SQL Query Used"):
            st.code(data["sql_used"], language="sql")
            st.caption(f"{data.get('sql_row_count', 0)} rows returned")

    analytics = data.get("analytics")
    if analytics and not analytics.get("error"):
        with st.expander("Analytics Detail"):
            plan = analytics.get("plan", {})
            review = analytics.get("review", {})
            st.markdown(f"**Execution mode:** `{plan.get('execution_mode', 'unknown')}`")
            st.markdown(f"**Answerability:** `{plan.get('answerability', 'unknown')}`")
            if analytics.get("plan_reasoning"):
                st.markdown(f"**Plan reasoning:** {analytics['plan_reasoning']}")
            if review.get("status"):
                st.markdown(f"**Review status:** `{review.get('status')}`")
            if review.get("reasoning"):
                st.markdown(f"**Review reasoning:** {review.get('reasoning')}")
            if plan.get("steps"):
                st.markdown("**Planned steps:**")
                for step in plan["steps"]:
                    st.markdown(f"- `{step.get('name', 'step')}` - {step.get('question', '')}")
            if analytics.get("step_results"):
                st.markdown("**Executed steps:**")
                for step in analytics["step_results"]:
                    st.markdown(
                        f"- `{step.get('name', 'step')}` - rows={step.get('row_count', 0)}, "
                        f"error={step.get('error') or 'none'}"
                    )
            if analytics.get("summary_stats"):
                st.markdown("**Summary Statistics:**")
                st.json(analytics["summary_stats"])
            if analytics.get("key_findings"):
                st.markdown("**Key Findings:**")
                for finding in analytics["key_findings"]:
                    st.markdown(f"- {finding}")
            if analytics.get("limitations"):
                st.markdown("**Limitations:**")
                for limitation in analytics["limitations"]:
                    st.markdown(f"- {limitation}")
            if analytics.get("missing_requirements"):
                st.markdown("**Missing requirements:**")
                for item in analytics["missing_requirements"]:
                    st.markdown(f"- {item}")
            if analytics.get("sql"):
                st.markdown("**Analytics SQL:**")
                st.code(analytics["sql"], language="sql")
    elif analytics and analytics.get("error"):
        with st.expander("Analytics Detail"):
            st.error(analytics.get("error"))
            if analytics.get("plan_reasoning"):
                st.markdown(f"**Plan reasoning:** {analytics['plan_reasoning']}")
            if analytics.get("step_results"):
                st.markdown("**Executed steps:**")
                for step in analytics["step_results"]:
                    st.markdown(
                        f"- `{step.get('name', 'step')}` - rows={step.get('row_count', 0)}, "
                        f"error={step.get('error') or 'none'}"
                    )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "loaded_session_id" not in st.session_state:
    st.session_state.loaded_session_id = None
if "sessions_expanded" not in st.session_state:
    st.session_state.sessions_expanded = False

_SESSIONS_PREVIEW = 4  # how many to show when collapsed

ensure_active_session()

with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:600;color:#3b82f6;letter-spacing:-0.3px;padding:0.25rem 0;'>Healthcare Intelligence</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        session = create_session()
        if session:
            st.session_state.current_session_id = session["id"]
            st.session_state.loaded_session_id = None
            st.rerun()

    st.markdown("<div style='font-size:0.78rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.6px;margin:0.75rem 0 0.3rem;'>Conversations</div>", unsafe_allow_html=True)

    _all_sessions = st.session_state.sessions
    _expanded = st.session_state.sessions_expanded
    _visible = _all_sessions if _expanded else _all_sessions[:_SESSIONS_PREVIEW]
    _hidden_count = max(0, len(_all_sessions) - _SESSIONS_PREVIEW)

    for session in _visible:
        is_active = session["id"] == st.session_state.current_session_id
        label = (session.get("title") or "New chat")
        if st.button(label, key=f"session_{session['id']}", use_container_width=True):
            st.session_state.current_session_id = session["id"]
            st.session_state.loaded_session_id = None
            st.rerun()

    if _hidden_count > 0 and not _expanded:
        if st.button(f"Show {_hidden_count} more ▾", key="sessions_show_more", use_container_width=True):
            st.session_state.sessions_expanded = True
            st.rerun()
    elif _expanded and len(_all_sessions) > _SESSIONS_PREVIEW:
        if st.button("Show less ▴", key="sessions_show_less", use_container_width=True):
            st.session_state.sessions_expanded = False
            st.rerun()

    st.divider()

    # ── Data Sources ───────────────────────────────────
    stats = api_get("/stats")
    counts = stats.get("counts", {})

    SYNTHEA_TABLES = {
        "patients":            "Patients",
        "encounters":          "Encounters",
        "conditions":          "Conditions",
        "medications":         "Medications",
        "observations":        "Observations",
        "clinical_notes":      "Clinical Notes",
        "operational_metrics": "Operational Metrics",
        "pro_responses":       "PRO Responses",
    }
    MIMIC_TABLES = {
        "mimic_patients":           "Patients",
        "mimic_admissions":         "Admissions",
        "mimic_icustays":           "ICU Stays",
        "mimic_labevents":          "Lab Events",
        "mimic_prescriptions":      "Prescriptions",
        "mimic_admission_features": "Admission Features",
    }
    EHG_TABLES = {
        "ehg_records":  "EHG Records",
        "ehg_features": "EHG Features",
    }
    PDF_TABLES = {
        "documents": "Documents",
    }

    def _render_group(group_label, tables):
        st.markdown(
            f"<div style='font-size:0.78rem;font-weight:600;color:#94a3b8;text-transform:uppercase;"
            f"letter-spacing:0.6px;margin:0.7rem 0 0.3rem;'>{group_label}</div>",
            unsafe_allow_html=True,
        )
        for table, icon_label in tables.items():
            count = counts.get(table, 0)
            count_cls = "ds-count" if count > 0 else "ds-count-zero"
            count_str = f"{count:,}" if count > 0 else "—"
            st.markdown(
                f"<div class='ds-row'><span>{icon_label}</span>"
                f"<span class='{count_cls}'>{count_str}</span></div>",
                unsafe_allow_html=True,
            )

    _render_group("Synthea EHR", SYNTHEA_TABLES)
    _render_group("MIMIC-III", MIMIC_TABLES)
    _render_group("EHG Signals", EHG_TABLES)
    _render_group("Clinical PDFs", PDF_TABLES)

    st.divider()

    # ── Data Ingestion ─────────────────────────────────
    st.markdown("<div style='font-size:0.78rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.5rem;'>Data Ingestion</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("CSVs", use_container_width=True):
            with st.spinner("Loading CSVs..."):
                result = api_post("/ingest/csv", timeout=300.0)
            st.success(result.get("message", "Done")) if "error" not in result else st.error(result["error"])
    with col2:
        if st.button("PDFs", use_container_width=True):
            with st.spinner("Parsing PDFs..."):
                result = api_post("/ingest/pdfs", timeout=300.0)
            st.success(result.get("message", "Done")) if "error" not in result else st.error(result["error"])
    if st.button("Generate Notes", use_container_width=True):
        with st.spinner("Generating clinical notes..."):
            result = api_post("/ingest/notes", timeout=600.0)
        st.success(result.get("message", "Done")) if "error" not in result else st.error(result["error"])

st.markdown(
    """
<div class="main-header">
  <h1>Healthcare Intelligence</h1>
  <p>Ask questions across patient registries, EHR data, clinical notes and NHS guidelines</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='color:#64748b;font-size:0.85rem;margin-top:-0.5rem;margin-bottom:1rem;'>"
    "Each conversation has its own memory — follow-up questions are answered in context."
    "</p>",
    unsafe_allow_html=True,
)

# ── Starter prompt trigger (set by welcome screen buttons) ──────────────────
if "starter_prompt" not in st.session_state:
    st.session_state.starter_prompt = None

# ── Welcome screen (shown only when chat is empty) ──────────────────────────
if not st.session_state.messages:
    st.markdown(
        "<div class='welcome-heading'>What would you like to explore today?</div>"
        "<div class='welcome-sub'>This platform connects to four medical data sources — here is what each one contains:</div>",
        unsafe_allow_html=True,
    )

    ds_cols = st.columns(4)
    data_sources = [
        ("Synthea EHR",
         "Synthetic NHS-style patient records — 10,000+ patients with diagnoses, medications, "
         "lab tests, clinical notes and care pathway data."),
        ("MIMIC-III ICU",
         "Real anonymised ICU data from Beth Israel Hospital — hospital admissions, lab results, "
         "prescriptions and mortality outcomes."),
        ("EHG Signals",
         "Electrohysterography waveform records used to predict preterm labour — signal features "
         "extracted per patient pregnancy."),
        ("Clinical PDFs",
         "Uploaded NHS guidelines and clinical protocols — parsed and indexed so you can ask "
         "natural-language questions about treatment recommendations."),
    ]
    for col, (title, desc) in zip(ds_cols, data_sources):
        with col:
            st.markdown(
                f"<div class='ds-card'>"
                f"<div class='ds-card-title'>{title}</div>"
                f"<div class='ds-card-desc'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='starters-heading'>Try one of these to get started:</div>",
        unsafe_allow_html=True,
    )

    STARTERS = [
        (
            "easy",
            "Simple",
            "How many patients are in the dataset and what are the most common conditions?",
        ),
        (
            "medium",
            "Intermediate",
            "Which patients have uncontrolled diabetes (HbA1c > 8) and are not on any diabetes medication?",
        ),
        (
            "hard",
            "Advanced",
            "Compare ICU outcomes for diabetic vs non-diabetic patients in MIMIC-III and identify the "
            "top three risk factors linked to in-hospital mortality.",
        ),
    ]

    s_cols = st.columns(3)
    for col, (level, label, question) in zip(s_cols, STARTERS):
        with col:
            st.markdown(
                f"<div class='starter-wrap starter-{level}'>"
                f"<div class='starter-body'>"
                f"<div class='starter-label label-{level}'>{label}</div>"
                f"<div class='starter-text'>{question}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Ask this →", key=f"starter_{level}", use_container_width=True):
                st.session_state.starter_prompt = question
                st.rerun()

    st.markdown("<hr style='margin:1.5rem 0 0.5rem;border:none;border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "data" in message:
                render_answer(message["data"])
            else:
                st.markdown(message.get("content", ""))

user_input = st.chat_input("Ask a clinical question...")

# Starter prompt takes priority over manual input
if st.session_state.starter_prompt:
    user_input = st.session_state.starter_prompt
    st.session_state.starter_prompt = None

if user_input and st.session_state.current_session_id:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("Thinking...")
        data = submit_query_with_polling(
            question=user_input,
            session_id=st.session_state.current_session_id,
            status_placeholder=status_placeholder,
        )
        status_placeholder.empty()

        if "error" in data:
            st.error(f"Error: {data['error']}")
        else:
            render_answer(data)
            refresh_current_session()
            st.rerun()


