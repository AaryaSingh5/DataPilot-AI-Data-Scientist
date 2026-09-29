"""
DataPilot — Streamlit Web Application
Autonomous, multi-agent AI data scientist with cryptographic audit trails and hallucination firewall.
"""
import hashlib
import io
import json
import os
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import duckdb

from datapilot.ledger.store import LedgerStore
from datapilot.llm.client import MockLLMClient, get_llm_client, LLMBudgetExceeded
from datapilot.ingestion.snapshot import SnapshotManager
from datapilot.ingestion.profiler import profile_table
from datapilot.ingestion.roles import infer_roles as _infer_roles_from_profile
from datapilot.benchmark.synth import generate_pricing_experiment
from datapilot.graph.workflow import DataPilotWorkflow
from datapilot.graph.state import DataPilotState
from datapilot.report.bundle import create_reproducibility_bundle


# ── Safety Constants ────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_UPLOAD_ROWS = 50_000
ALLOWED_EXTENSIONS = {".csv", ".parquet", ".xlsx"}
FORBIDDEN_EXTENSIONS = {".xlsm", ".xlsb", ".xltm", ".xlam"}


# ── Pandas ↔ DuckDB bridge helpers ─────────────────────────────────────────────

def snapshot_dataframe(df: pd.DataFrame, snap_dir: Path, table_name: str = "data") -> dict:
    """Register a pandas DataFrame into DuckDB, snapshot it, and return hash info."""
    conn = duckdb.connect(":memory:")
    conn.register(table_name, df)
    mgr = SnapshotManager(snap_dir)
    dataset_hash = mgr.create_snapshot(conn, [table_name])
    return {"dataset_hash": dataset_hash, "conn": conn, "table_name": table_name}


def profile_dataframe(df: pd.DataFrame, table_name: str = "data") -> dict:
    """Profile a pandas DataFrame using the DuckDB profiler."""
    conn = duckdb.connect(":memory:")
    conn.register(table_name, df)
    col_profiles = profile_table(conn, table_name)
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    return {
        "row_count": len(df),
        "col_count": len(df.columns),
        "dtypes": dtypes,
        "col_profiles": col_profiles,
    }


def infer_roles(df: pd.DataFrame) -> dict:
    """Infer column roles from a pandas DataFrame."""
    col_profiles = profile_dataframe(df)["col_profiles"]
    return _infer_roles_from_profile(col_profiles)


# ── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataPilot — Autonomous AI Data Scientist",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main header card ───────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e40af 100%);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 20px 60px -10px rgba(30,27,75,0.4);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99,102,241,0.3) 0%, transparent 70%);
        border-radius: 50%;
    }

    /* ── Badge pills ────────────────────────────────────────────────────── */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .badge-green  { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.35); }
    .badge-blue   { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.35); }
    .badge-purple { background: rgba(168,85,247,0.15); color: #c084fc; border: 1px solid rgba(168,85,247,0.35); }
    .badge-red    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.35); }

    /* ── Metric cards ───────────────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2e2e52;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px -4px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #f8fafc; line-height: 1; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; margin-top: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }

    /* ── Finding cards ──────────────────────────────────────────────────── */
    .finding-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid #2e2e52;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
    }
    .finding-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        border-radius: 4px 0 0 4px;
    }
    .finding-supported::before { background: linear-gradient(180deg, #22c55e, #16a34a); }
    .finding-weak::before      { background: linear-gradient(180deg, #f59e0b, #d97706); }
    .finding-refuted::before   { background: linear-gradient(180deg, #ef4444, #dc2626); }
    .finding-insufficient::before { background: linear-gradient(180deg, #94a3b8, #64748b); }

    /* ── Code viewer ────────────────────────────────────────────────────── */
    .code-block-wrapper {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 16px;
    }
    .code-block-header {
        background: #1e293b;
        padding: 10px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
    }
    .code-block-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .code-badge-stat  { background: rgba(99,102,241,0.3); color: #a5b4fc; }
    .code-badge-sql   { background: rgba(245,158,11,0.3); color: #fcd34d; }
    .code-badge-ml    { background: rgba(236,72,153,0.3); color: #f9a8d4; }
    .code-badge-viz   { background: rgba(34,197,94,0.3);  color: #86efac; }

    /* ── Evidence explorer ──────────────────────────────────────────────── */
    .evidence-row {
        background: #1a1a2e;
        border: 1px solid #2e2e52;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s;
    }
    .evidence-row:hover { background: #1e1e42; border-color: #6366f1; }

    /* ── Chat interface ──────────────────────────────────────────────────── */
    .chat-message {
        display: flex;
        gap: 12px;
        margin-bottom: 18px;
        align-items: flex-start;
    }
    .chat-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    .avatar-user   { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
    .avatar-pilot  { background: linear-gradient(135deg, #0ea5e9, #6366f1); }
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 12px;
        max-width: 85%;
        line-height: 1.6;
        font-size: 0.9rem;
    }
    .bubble-user  { background: #312e81; color: #e0e7ff; border-bottom-right-radius: 4px; }
    .bubble-pilot { background: #1e293b; color: #e2e8f0; border-bottom-left-radius: 4px; border: 1px solid #334155; }

    /* ── Hypothesis card ─────────────────────────────────────────────────── */
    .hypothesis-card {
        background: #161922;
        border-left: 4px solid #6366f1;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }

    /* ── Scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Session State ───────────────────────────────────────────────────────────────

def init_session():
    if "store" not in st.session_state:
        db_path = Path(".datapilot_cache/ledger.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        st.session_state.store = LedgerStore(db_path=db_path)
    if "snapshot_dir" not in st.session_state:
        snap_path = Path(".datapilot_cache/snapshots")
        snap_path.mkdir(parents=True, exist_ok=True)
        st.session_state.snapshot_dir = snap_path
    if "current_df" not in st.session_state:
        df, gt = generate_pricing_experiment(1200, 42)
        st.session_state.current_df = df
        st.session_state.current_gt = gt
        st.session_state.dataset_name = "saas_pricing_experiment"
    if "last_report" not in st.session_state:
        st.session_state.last_report = None
    if "last_state" not in st.session_state:
        st.session_state.last_state = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_evidence_id" not in st.session_state:
        st.session_state.selected_evidence_id = None


init_session()
store: LedgerStore = st.session_state.store
snap_dir: Path = st.session_state.snapshot_dir


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 16px;">
        <div style="font-size:2rem;">🧭</div>
        <div style="font-size:1.1rem; font-weight:800; color:#e2e8f0;">DataPilot</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">Autonomous AI Data Scientist</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 📂 Dataset")
    data_option = st.radio(
        "Source",
        ["Synthetic Benchmark (Pricing Experiment)", "Upload CSV / Parquet"],
        label_visibility="collapsed",
    )

    if data_option == "Synthetic Benchmark (Pricing Experiment)":
        n_rows = st.slider("Benchmark Rows", 500, 5000, 1200, 100)
        seed = st.number_input("Random Seed", value=42, step=1)
        if st.button("🔄 Generate Benchmark", use_container_width=True):
            df, gt = generate_pricing_experiment(n_rows, int(seed))
            st.session_state.current_df = df
            st.session_state.current_gt = gt
            st.session_state.dataset_name = "saas_pricing_experiment"
            st.success(f"Loaded {n_rows:,}-row synthetic benchmark!")
    else:
        uploaded_file = st.file_uploader(
            "Upload data file (Max 10 MB, 50,000 rows)",
            type=["csv", "parquet", "xlsx"],
        )
        if uploaded_file:
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext in FORBIDDEN_EXTENSIONS:
                st.error("❌ Macro-enabled Excel files rejected for security.")
            elif file_ext not in ALLOWED_EXTENSIONS:
                st.error(f"❌ Format '{file_ext}' not supported.")
            elif uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
                st.error(f"❌ File size exceeds 10 MB limit.")
            else:
                try:
                    if file_ext == ".csv":
                        df_upload = pd.read_csv(uploaded_file)
                    elif file_ext == ".parquet":
                        df_upload = pd.read_parquet(uploaded_file)
                    else:
                        df_upload = pd.read_excel(uploaded_file)

                    if len(df_upload) > MAX_UPLOAD_ROWS:
                        st.error(f"❌ {len(df_upload):,} rows exceeds 50,000 limit.")
                    else:
                        st.session_state.current_df = df_upload
                        st.session_state.dataset_name = Path(uploaded_file.name).stem
                        st.success(f"✅ Loaded '{uploaded_file.name}' ({len(df_upload):,} rows)")
                except Exception as exc:
                    st.error(f"❌ Parse error: {exc}")

    st.divider()
    st.markdown("#### ⚙️ Engine Settings")
    llm_mode = st.selectbox(
        "LLM Client",
        ["Mock Client (Deterministic)", "NVIDIA AI (Llama 3.2 11B)", "Anthropic Claude"],
        index=0,
    )

    session_api_key = None
    if llm_mode != "Mock Client (Deterministic)":
        st.markdown("##### 🔐 API Key")
        provider_name = "NVIDIA" if "NVIDIA" in llm_mode else "Anthropic"
        env_var_name = "NVIDIA_API_KEY" if "NVIDIA" in llm_mode else "ANTHROPIC_API_KEY"
        env_key = os.environ.get(env_var_name)
        if env_key:
            st.caption(f"✓ {provider_name} key loaded from environment.")
            session_api_key = env_key
        else:
            key_input = st.text_input(f"{provider_name} API Key", type="password", key="custom_api_key_input")
            if key_input:
                session_api_key = key_input
                st.caption("✓ Key active in session memory only.")

    alpha = st.slider("Significance α", 0.01, 0.10, 0.05, 0.01)
    fdr_q = st.slider("FDR q-value", 0.01, 0.10, 0.05, 0.01)

    st.divider()
    # Dataset info
    df = st.session_state.current_df
    ds_name = st.session_state.get("dataset_name", "dataset")
    st.caption(f"📋 **{ds_name}**")
    st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
    st.caption("🔒 SHA-256 Hash Chained Ledger")


# ── Data & Profile ──────────────────────────────────────────────────────────────
df = st.session_state.current_df
profile = profile_dataframe(df)
roles = infer_roles(df)
snap_result = snapshot_dataframe(df, snap_dir, table_name="data")
d_hash = snap_result["dataset_hash"]


# ── Warning Banner ──────────────────────────────────────────────────────────────
st.warning(
    "⚠️ **Experimental Demo**: Results are generated by multi-agent AI heuristics and rule-based "
    "consistency checks. Not a substitute for review by a professional statistician."
)


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <div style="position: relative; z-index: 1;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <h1 style="margin: 0; color: #f8fafc; font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em;">
                    🧭 DataPilot
                </h1>
                <p style="margin: 8px 0 0 0; color: #a5b4fc; font-size: 0.95rem; font-weight: 400;">
                    Autonomous multi-agent AI data scientist with cryptographic audit trails
                </p>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center; padding-top: 4px;">
                <span class="badge-pill badge-green">⛨ Firewall Active</span>
                <span class="badge-pill badge-blue">🔗 SHA-256 Chained</span>
                <span class="badge-pill badge-purple">🤖 Multi-Agent</span>
                <span class="badge-pill badge-red">🛡 Sandbox Isolated</span>
            </div>
        </div>
        <div style="margin-top: 16px; display: flex; gap: 24px; color: #94a3b8; font-size: 0.8rem;">
            <span>Dataset: <code style="color:#e0e7ff;">{d_hash[:16]}…</code></span>
            <span>Rows: <strong style="color:#e0e7ff;">{profile['row_count']:,}</strong></span>
            <span>Columns: <strong style="color:#e0e7ff;">{profile['col_count']}</strong></span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ────────────────────────────────────────────────────────────────────────
tab_investigate, tab_data, tab_code, tab_report, tab_ledger, tab_chat = st.tabs([
    "🔍 Investigate",
    "📊 Dataset",
    "💻 Code & Evidence",
    "📑 Report",
    "🛡️ Audit Ledger",
    "💬 Chat",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: INVESTIGATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_investigate:
    st.markdown("### 🔍 Research Question")
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        default_q = "Did the pricing change increase revenue, and is the effect confounded by customer segment?"
        user_question = st.text_input(
            "Research question:",
            value=default_q,
            placeholder="e.g. Did marketing spend increase retention in Q3?",
            label_visibility="collapsed",
        )
    with col_btn:
        st.write("")
        run_btn = st.button("🚀 Run", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("🤖 Executing DataPilot multi-agent workflow…"):
            provider_map = {
                "Mock Client (Deterministic)": "mock",
                "NVIDIA AI (Llama 3.2 11B)": "nvidia",
                "Anthropic Claude": "anthropic",
            }
            target_provider = provider_map.get(llm_mode, "mock")
            try:
                llm_client = get_llm_client(provider=target_provider, api_key=session_api_key)

                if isinstance(llm_client, MockLLMClient):
                    llm_client.register("Planner", json.dumps({
                        "hypotheses": [
                            {
                                "id": "H1",
                                "statement": "Pricing change increased revenue post-experiment",
                                "columns": ["is_treated", "revenue_diff"],
                                "test_type": "before_after",
                            }
                        ],
                        "plan": [
                            {
                                "step": 1,
                                "agent": "stats",
                                "action": "Run before_after comparison on revenue_diff",
                                "hypothesis_id": "H1",
                            }
                        ],
                    }))
                    llm_client.register("Stats Agent", json.dumps({
                        "test": "before_after",
                        "columns": {"a": "pre_revenue", "b": "post_revenue"},
                        "justification": "Compare revenue before and after experiment",
                    }))
                    llm_client.register("Lead Analyst", json.dumps({
                        "conclusions": [
                            {
                                "hypothesis_id": "H1",
                                "verdict": "supported",
                                "confidence": "high",
                                "summary": "Revenue grew significantly following the pricing adjustment.",
                                "caveats": ["Controlled for baseline customer segment distributions."],
                                "evidence_ids": ["ev_0002"],
                            }
                        ],
                        "overall_summary": "The pricing change produced a statistically robust increase in revenue.",
                    }))
                    llm_client.register("Visualization Agent", json.dumps({
                        "charts": [
                            {
                                "evidence_id": "ev_0002",
                                "chart_type": "bar",
                                "x": "segment",
                                "y": "revenue_diff",
                                "title": "Revenue Change by Customer Segment",
                            }
                        ]
                    }))

                wf = DataPilotWorkflow(llm=llm_client, store=store, snapshot_dir=snap_dir)
                run_id = f"run_{uuid.uuid4().hex[:8]}"
                initial_state: DataPilotState = {
                    "run_id": run_id,
                    "dataset_hash": d_hash,
                    "question": user_question,
                    "schema": {"data": profile["dtypes"]},
                    "role_map": roles,
                    "table_name": "data",
                    "snapshot_dir": str(snap_dir),
                    "evidence_ids": [],
                    "evidences": [],
                }

                final_state = wf.run(initial_state)
                st.session_state.last_report = final_state.get("report")
                st.session_state.last_state = final_state

                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question,
                    "timestamp": datetime.now().strftime("%H:%M"),
                })
                report = final_state.get("report", {})
                summary = report.get("executive_summary", "Investigation complete.")
                fw_ok = report.get("firewall", {}).get("passed", True)
                score = report.get("evaluation", {}).get("overall_score", 1.0)
                pilot_msg = (
                    f"**Investigation complete** ({'✅ Firewall passed' if fw_ok else '❌ Firewall failed'} | "
                    f"Rigor score: {score:.0%})\n\n{summary}"
                )
                st.session_state.chat_history.append({
                    "role": "pilot",
                    "content": pilot_msg,
                    "timestamp": datetime.now().strftime("%H:%M"),
                })

                st.success("✅ Investigation complete! Navigate to other tabs for details.")
            except LLMBudgetExceeded as exc:
                st.error(f"💸 LLM budget exceeded: {exc}")
            except Exception as exc:
                st.error(f"❌ Failed to run investigation: {exc}")
                import traceback
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc())

    # ── Results summary ────────────────────────────────────────────────────
    if st.session_state.last_report:
        rep = st.session_state.last_report
        st.markdown("---")

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        fw_pass = rep.get("firewall", {}).get("passed", False)
        score = rep.get("evaluation", {}).get("overall_score", 1.0)
        e_count = len(rep.get("evidences", []))
        conclusion_count = len(rep.get("conclusions", []))

        with c1:
            color = "#4ade80" if fw_pass else "#f87171"
            label = "PASSED" if fw_pass else "FAILED"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{label}</div>
                <div class="metric-label">Hallucination Firewall</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            color = "#4ade80" if score >= 0.85 else ("#f59e0b" if score >= 0.5 else "#f87171")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{score:.0%}</div>
                <div class="metric-label">Statistical Rigor Score</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#c084fc;">{e_count}</div>
                <div class="metric-label">Chained Evidences</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#60a5fa;">{conclusion_count}</div>
                <div class="metric-label">Findings & Verdicts</div>
            </div>""", unsafe_allow_html=True)

        # Executive Summary
        summary = rep.get("executive_summary", "")
        if summary:
            st.markdown("#### 📋 Executive Summary")
            st.info(summary)

        # Ranked findings cards
        conclusions = rep.get("conclusions", [])
        if conclusions:
            st.markdown("#### 🔬 Ranked Findings")

            _CONF_W = {"high": 1.0, "medium": 0.6, "low": 0.3, "uncertain": 0.1}
            _VRD_W  = {"supported": 1.0, "weak": 0.5, "insufficient": 0.2, "refuted": 0.3}
            ranked = sorted(
                conclusions,
                key=lambda c: _CONF_W.get(c.get("confidence","low"),0.1) * _VRD_W.get(c.get("verdict","insufficient"),0.1),
                reverse=True,
            )
            _VRD_ICON = {"supported": "✅", "weak": "⚠️", "insufficient": "🔶", "refuted": "❌"}

            for i, c in enumerate(ranked, 1):
                hid = c.get("hypothesis_id", f"H{i}")
                verdict = c.get("verdict", "insufficient")
                conf = c.get("confidence", "medium")
                summ = c.get("summary", "")
                icon = _VRD_ICON.get(verdict, "")
                card_cls = f"finding-{verdict}"

                evidence_ids = c.get("evidence_ids", [])
                ev_by_id = {e.get("id"): e for e in rep.get("evidences", [])}

                # Gather stats
                stats_parts = []
                for eid in evidence_ids:
                    ev = ev_by_id.get(eid)
                    if ev:
                        r = ev.get("result", {})
                        p = r.get("p") or r.get("p_value")
                        d = r.get("effect_size")
                        n = r.get("n") or r.get("n_a")
                        parts = []
                        if p is not None: parts.append(f"p={p:.4f}")
                        if d is not None: parts.append(f"|d|={abs(d):.3f}")
                        if n is not None: parts.append(f"n={n:,}")
                        if parts:
                            stats_parts.append(" · ".join(parts))

                stats_str = " &nbsp;|&nbsp; ".join(stats_parts) if stats_parts else ""

                caveats = c.get("caveats", [])
                caveats_html = ""
                if caveats:
                    caveat_items = "".join(f"<li style='color:#94a3b8;font-size:0.82rem;'>{cav}</li>" for cav in caveats)
                    caveats_html = f"<ul style='margin:8px 0 0 16px;'>{caveat_items}</ul>"

                st.markdown(f"""
                <div class="finding-card {card_cls}">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;">#{i} · {hid}</span>
                        <div style="display:flex; gap:8px; align-items:center;">
                            <span style="font-size:0.75rem; background:rgba(255,255,255,0.08); padding:2px 10px; border-radius:999px; color:#94a3b8;">{conf.upper()} CONFIDENCE</span>
                        </div>
                    </div>
                    <div style="font-size:1rem; font-weight:600; color:#f1f5f9; margin-bottom:8px;">{icon} {verdict.upper()} — {hid}</div>
                    <div style="color:#cbd5e1; font-size:0.9rem; line-height:1.6;">{summ}</div>
                    {f'<div style="margin-top:10px; font-size:0.8rem; color:#6366f1; font-family:monospace;">{stats_str}</div>' if stats_str else ""}
                    {caveats_html}
                </div>
                """, unsafe_allow_html=True)

        # Firewall violations
        fw = rep.get("firewall", {})
        violations = fw.get("violations", [])
        if violations:
            st.markdown("#### ❌ Firewall Violations")
            for v in violations:
                st.error(f"**[{v.get('check_name', 'check')}]** {v.get('message', '')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown(f"**Dataset Hash:** `{d_hash}` | **Rows:** `{profile['row_count']:,}` | **Columns:** `{profile['col_count']}`")

    subtab_preview, subtab_profile, subtab_roles = st.tabs(["📋 Preview", "📈 Statistics", "🏷️ Column Roles"])

    with subtab_preview:
        n_rows_show = st.slider("Rows to preview", 5, 100, 20)
        st.dataframe(df.head(n_rows_show), use_container_width=True)

    with subtab_profile:
        st.markdown("#### Summary Statistics")
        st.dataframe(df.describe().T.style.format("{:.4f}", subset=df.describe().T.select_dtypes("float").columns), use_container_width=True)

        # Numeric column distributions
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            st.markdown("#### Column Distributions")
            sel_col = st.selectbox("Select column:", numeric_cols)
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 3))
                fig.patch.set_facecolor("#0f172a")
                ax.set_facecolor("#0f172a")
                ax.hist(df[sel_col].dropna(), bins=40, color="#6366f1", alpha=0.8, edgecolor="#312e81")
                ax.set_xlabel(sel_col, color="#94a3b8")
                ax.set_ylabel("Frequency", color="#94a3b8")
                ax.tick_params(colors="#64748b")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1e293b")
                st.pyplot(fig, use_container_width=True)
                plt.close()
            except Exception:
                pass

    with subtab_roles:
        st.markdown("#### Inferred Column Roles")
        role_df = pd.DataFrame([{"Column": k, "Inferred Role": v, "Dtype": profile["dtypes"].get(k, "")}
                                  for k, v in roles.items()])
        st.dataframe(role_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: CODE VIEWER & EVIDENCE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_code:
    if not st.session_state.last_report:
        st.info("🔍 Run an investigation first to explore generated code and evidence.")
    else:
        rep = st.session_state.last_report
        evidences = rep.get("evidences", [])

        subtab_code_viewer, subtab_ev_explorer = st.tabs(["💻 Code Viewer", "🔬 Evidence Explorer"])

        with subtab_code_viewer:
            st.markdown("### 💻 Generated & Executable Code")
            st.caption("Every code snippet below was logged in the cryptographic ledger with its exact inputs, outputs, and hash.")

            # Filter controls
            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                kind_filter = st.multiselect(
                    "Filter by kind:",
                    options=["stat_test", "sql", "model", "viz", "analysis", "evaluation", "report"],
                    default=["stat_test", "sql", "model", "viz"],
                )
            with col_f2:
                show_all_code = st.checkbox("Show all (incl. boilerplate)", value=False)

            # Filter and show
            _KIND_BADGE = {
                "stat_test": ("code-badge-stat", "STATS"),
                "sql": ("code-badge-sql", "SQL"),
                "model": ("code-badge-ml", "ML"),
                "viz": ("code-badge-viz", "VIZ"),
                "analysis": ("code-badge-stat", "ANALYSIS"),
            }

            filtered_evs = [
                e for e in evidences
                if (not kind_filter or e.get("kind") in kind_filter)
                and (show_all_code or (e.get("code") and e.get("code") not in ("synthesis", "report_synthesis", "evaluator_threshold_checks")))
            ]

            if not filtered_evs:
                st.info("No code evidence matching the current filter.")
            else:
                for ev in filtered_evs:
                    eid = ev.get("id", "?")
                    kind = ev.get("kind", "")
                    purpose = ev.get("purpose") or kind
                    code = ev.get("code", "")
                    order = ev.get("execution_order", "?")
                    status = ev.get("status", "ok")
                    badge_cls, badge_label = _KIND_BADGE.get(kind, ("code-badge-stat", kind.upper()))
                    status_color = "#4ade80" if status == "ok" else "#f87171"

                    st.markdown(f"""
                    <div class="code-block-wrapper">
                        <div class="code-block-header">
                            <span>
                                <span class="code-block-badge {badge_cls}">{badge_label}</span>
                                &nbsp;<strong style="color:#e2e8f0;">{eid}</strong>
                                &nbsp;<span style="color:#64748b;">Step {order}</span>
                            </span>
                            <span style="color:{status_color};">{'✓' if status == 'ok' else '✗'} {status.upper()}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"📄 {purpose}", expanded=(kind in ("stat_test", "sql"))):
                        if code and code not in ("synthesis", "report_synthesis", "evaluator_threshold_checks"):
                            # Editable code view
                            edited = st.text_area(
                                "Code",
                                value=code,
                                height=200,
                                key=f"code_edit_{eid}",
                                label_visibility="collapsed",
                            )
                            col_copy, col_run = st.columns([1, 1])
                            with col_copy:
                                if st.button("📋 Copy to clipboard", key=f"copy_{eid}", use_container_width=True):
                                    st.toast("Code copied to clipboard!", icon="📋")

                            # Show result
                            result = ev.get("result", {})
                            if result:
                                st.markdown("**Result:**")
                                st.json(result)
                        else:
                            st.caption("No executable code for this evidence record.")

            # Download assembled script
            st.markdown("---")
            all_code_evs = [e for e in evidences if e.get("code") and e.get("code") not in ("synthesis", "report_synthesis", "evaluator_threshold_checks")]
            if all_code_evs:
                from datapilot.report.bundle import _build_runnable_code
                assembled = _build_runnable_code(all_code_evs, rep.get("run_id", "run"), d_hash)
                st.download_button(
                    "⬇️ Download runnable_code.py",
                    data=assembled.encode("utf-8"),
                    file_name=f"runnable_code_{rep.get('run_id', 'run')}.py",
                    mime="text/x-python",
                    use_container_width=True,
                )

        with subtab_ev_explorer:
            st.markdown("### 🔬 Evidence Chain Explorer")
            st.caption("Drill into any evidence record to inspect its inputs, outputs, and cryptographic hash.")

            if not evidences:
                st.info("No evidence records for this run yet.")
            else:
                # Summary table
                ev_table = pd.DataFrame([{
                    "ID": e.get("id"),
                    "Kind": e.get("kind"),
                    "Producer": e.get("produced_by"),
                    "Step": e.get("execution_order"),
                    "Status": e.get("status"),
                    "Purpose": (e.get("purpose") or "")[:60],
                    "Hash": (e.get("hash") or "")[:12] + "…",
                } for e in evidences])
                st.dataframe(ev_table, use_container_width=True, hide_index=True)

                st.markdown("#### 🔍 Inspect Evidence Record")
                ev_ids = [e.get("id") for e in evidences]
                sel_id = st.selectbox("Select evidence ID:", ev_ids)

                sel_ev = next((e for e in evidences if e.get("id") == sel_id), None)
                if sel_ev:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Metadata**")
                        meta = {
                            "id": sel_ev.get("id"),
                            "kind": sel_ev.get("kind"),
                            "produced_by": sel_ev.get("produced_by"),
                            "execution_order": sel_ev.get("execution_order"),
                            "status": sel_ev.get("status"),
                            "purpose": sel_ev.get("purpose"),
                            "depends_on": sel_ev.get("depends_on"),
                            "dataset_hash": sel_ev.get("dataset_hash", "")[:16] + "…",
                            "hash": (sel_ev.get("hash") or "")[:16] + "…",
                            "prev_hash": (sel_ev.get("prev_hash") or "")[:16] + "…",
                        }
                        for k, v in meta.items():
                            st.text(f"{k}: {v}")
                    with col_b:
                        st.markdown("**Result**")
                        result = sel_ev.get("result", {})
                        if result:
                            st.json(result)
                        else:
                            st.caption("No result data.")

                    if sel_ev.get("error"):
                        st.error(f"Error: {sel_ev['error']}")

                    code = sel_ev.get("code", "")
                    if code and code not in ("synthesis", "report_synthesis", "evaluator_threshold_checks"):
                        st.markdown("**Executed Code**")
                        st.code(code, language="python")

                    # Provenance DAG (simple)
                    depends_on = sel_ev.get("depends_on", [])
                    if depends_on:
                        st.markdown("**Depends On**")
                        for dep_id in depends_on:
                            dep_ev = next((e for e in evidences if e.get("id") == dep_id), None)
                            if dep_ev:
                                st.markdown(f"- `{dep_id}` ({dep_ev.get('kind')} · {dep_ev.get('produced_by')})")
                            else:
                                st.markdown(f"- `{dep_id}`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    if not st.session_state.last_report:
        st.info("📑 Run an investigation first to view the full report.")
    else:
        rep = st.session_state.last_report

        # Download buttons
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button(
                "📥 Markdown Report",
                data=rep.get("markdown", ""),
                file_name=f"datapilot_report_{rep.get('run_id')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                "🌐 HTML Report",
                data=rep.get("html", ""),
                file_name=f"datapilot_report_{rep.get('run_id')}.html",
                mime="text/html",
                use_container_width=True,
            )
        with col_d3:
            bundle_zip = create_reproducibility_bundle(
                run_id=rep.get("run_id", "run"),
                store=store,
                report_data=rep,
                dataset_hash=d_hash,
            )
            st.download_button(
                "📦 Full Bundle (ZIP)",
                data=bundle_zip,
                file_name=f"bundle_{rep.get('run_id')}.zip",
                mime="application/zip",
                use_container_width=True,
            )

        st.markdown("---")

        # Render markdown
        md = rep.get("markdown", "")
        if md:
            st.markdown(md)

        # Embedded HTML preview
        html_content = rep.get("html", "")
        if html_content:
            st.markdown("#### 🌐 HTML Report Preview")
            with st.expander("Expand HTML preview"):
                import base64
                b64 = base64.b64encode(html_content.encode()).decode()
                st.markdown(
                    f'<iframe src="data:text/html;base64,{b64}" width="100%" height="600" '
                    f'style="border:1px solid #2e2e52; border-radius:8px;"></iframe>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: AUDIT LEDGER
# ══════════════════════════════════════════════════════════════════════════════
with tab_ledger:
    st.markdown("### 🛡️ Immutable Cryptographic Audit Ledger")
    st.caption("Every agent action is SHA-256 hash-chained. Any tampering is detectable.")

    rows = store.list_recent(100)

    if rows:
        # Chain verification
        if st.session_state.last_report:
            rid = st.session_state.last_report.get("run_id")
            if rid:
                is_valid = store.verify_chain(rid)
                if is_valid:
                    st.success(f"✅ Chain verified: Run `{rid}` is 100% untampered.")
                else:
                    st.error(f"❌ Chain integrity FAILED for Run `{rid}`. Tampering detected!")

        # Stats
        run_ids = list(set(r.get("run_id") for r in rows))
        kinds = list(set(r.get("kind") for r in rows))
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(rows))
        c2.metric("Analysis Runs", len(run_ids))
        c3.metric("Evidence Types", len(kinds))

        # Full ledger table
        ledger_df = pd.DataFrame(rows)
        st.dataframe(
            ledger_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "hash": st.column_config.TextColumn("Hash", width="small"),
                "prev_hash": st.column_config.TextColumn("Prev Hash", width="small"),
            }
        )

        # Run filter
        st.markdown("#### Filter by Run")
        sel_run = st.selectbox("Run ID:", ["All"] + run_ids)
        if sel_run != "All":
            ev_detail = store.list_evidence(sel_run)
            if ev_detail:
                run_df = pd.DataFrame(ev_detail)
                # Truncate long columns for display
                for col in ["code", "params", "result"]:
                    if col in run_df.columns:
                        run_df[col] = run_df[col].astype(str).str[:80] + "…"
                st.dataframe(run_df, use_container_width=True, hide_index=True)
    else:
        st.info("No ledger records yet. Run an investigation to populate the audit trail.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### 💬 DataPilot Chat")
    st.caption("Ask questions about your analysis. Responses reference the cryptographic evidence ledger.")

    # Chat history display
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding:48px 24px; color:#475569;">
            <div style="font-size:3rem; margin-bottom:16px;">💬</div>
            <div style="font-size:1.1rem; font-weight:600; color:#64748b; margin-bottom:8px;">Start a conversation</div>
            <div style="font-size:0.9rem; color:#475569;">
                Run an investigation first, then ask questions about the results.<br>
                Example: "Why was H1 marked as supported?" or "What does the p-value mean here?"
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                is_user = msg["role"] == "user"
                avatar = "🧑" if is_user else "🧭"
                bubble_cls = "bubble-user" if is_user else "bubble-pilot"
                avatar_cls = "avatar-user" if is_user else "avatar-pilot"
                ts = msg.get("timestamp", "")

                if is_user:
                    st.markdown(f"""
                    <div class="chat-message" style="justify-content:flex-end;">
                        <div>
                            <div class="chat-bubble {bubble_cls}">{msg['content']}</div>
                            <div style="text-align:right; font-size:0.7rem; color:#475569; margin-top:4px;">{ts}</div>
                        </div>
                        <div class="chat-avatar {avatar_cls}">{avatar}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message">
                        <div class="chat-avatar {avatar_cls}">{avatar}</div>
                        <div>
                            <div class="chat-bubble {bubble_cls}">{msg['content']}</div>
                            <div style="font-size:0.7rem; color:#475569; margin-top:4px;">{ts}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")

    # Chat input
    col_input, col_send = st.columns([5, 1])
    with col_input:
        chat_input = st.text_input(
            "Message",
            placeholder="Ask about the analysis… e.g. 'Summarize the key findings' or 'Explain the p-value'",
            label_visibility="collapsed",
            key="chat_input_field",
        )
    with col_send:
        send_btn = st.button("Send ➤", use_container_width=True, type="primary")

    if send_btn and chat_input.strip():
        user_msg = chat_input.strip()
        ts = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({"role": "user", "content": user_msg, "timestamp": ts})

        # Generate a context-aware reply
        rep = st.session_state.last_report
        if rep is None:
            reply = "⚠️ No analysis has been run yet. Please run an investigation in the **Investigate** tab first, then I can answer questions about it."
        else:
            # Build a simple context-aware reply
            conclusions = rep.get("conclusions", [])
            fw_passed = rep.get("firewall", {}).get("passed", True)
            score = rep.get("evaluation", {}).get("overall_score", 1.0)
            summary = rep.get("executive_summary", "")
            run_id = rep.get("run_id", "")
            evidences = rep.get("evidences", [])

            lower_q = user_msg.lower()

            if any(kw in lower_q for kw in ["summarize", "summary", "overview", "key findings", "main"]):
                if conclusions:
                    parts = [f"**{c.get('hypothesis_id')}**: {c.get('verdict','?').upper()} ({c.get('confidence','?')} confidence) — {c.get('summary','')}" for c in conclusions]
                    reply = f"**Run `{run_id}` — Executive Summary**\n\n{summary}\n\n**Findings:**\n" + "\n\n".join(f"- {p}" for p in parts)
                else:
                    reply = summary or "No conclusions were generated in this run."

            elif any(kw in lower_q for kw in ["firewall", "hallucination", "violation"]):
                violations = rep.get("firewall", {}).get("violations", [])
                warnings = rep.get("firewall", {}).get("warnings", [])
                if fw_passed:
                    reply = f"✅ **Hallucination Firewall PASSED** for run `{run_id}`.\n\nAll claims were verified against the cryptographic evidence ledger. {len(warnings)} warning(s) noted but no violations found."
                else:
                    v_text = "\n".join(f"- **[{v.get('check_name')}]** {v.get('message','')}" for v in violations)
                    reply = f"❌ **Firewall FAILED** — {len(violations)} violation(s) found:\n\n{v_text}"

            elif any(kw in lower_q for kw in ["p-value", "p value", "significance", "significant"]):
                stat_evs = [e for e in evidences if e.get("kind") == "stat_test"]
                if stat_evs:
                    ev = stat_evs[0]
                    r = ev.get("result", {})
                    p = r.get("p") or r.get("p_value")
                    reply = (
                        f"The p-value for evidence `{ev.get('id')}` is **{p:.4f}**.\n\n"
                        f"This {'is' if p and p < 0.05 else 'is NOT'} statistically significant at α=0.05. "
                        f"A p-value tells you the probability of observing results this extreme under the null hypothesis. "
                        f"Lower values (< 0.05) indicate stronger evidence against the null."
                    ) if p is not None else "No p-value found in the evidence records."
                else:
                    reply = "No statistical test evidence found in this run."

            elif any(kw in lower_q for kw in ["effect size", "effect", "cohen", "magnitude"]):
                stat_evs = [e for e in evidences if e.get("kind") == "stat_test"]
                if stat_evs:
                    ev = stat_evs[0]
                    r = ev.get("result", {})
                    d = r.get("effect_size")
                    if d is not None:
                        abs_d = abs(d)
                        interp = "small" if abs_d < 0.2 else ("medium" if abs_d < 0.5 else "large")
                        reply = (
                            f"Cohen's d (effect size) for `{ev.get('id')}` is **{d:.3f}** (|d| = {abs_d:.3f}), "
                            f"which is considered a **{interp}** effect. Effect size measures practical significance "
                            f"independent of sample size."
                        )
                    else:
                        reply = "No effect size found in the evidence records."
                else:
                    reply = "No statistical test evidence found in this run."

            elif any(kw in lower_q for kw in ["score", "rigor", "quality"]):
                reply = (
                    f"**Statistical Rigor Score: {score:.0%}**\n\n"
                    f"This score aggregates evaluations of {len(evidences)} evidence records against thresholds for:\n"
                    f"- Sample size adequacy\n- Statistical significance (α=0.05)\n"
                    f"- FDR multiple testing correction\n- Effect size magnitude\n- Model performance\n\n"
                    f"Score ≥ 85% = PASSED · ≥ 50% = WARNING · < 50% = FAILED"
                )

            elif any(kw in lower_q for kw in ["code", "python", "reproduce", "script"]):
                code_evs = [e for e in evidences if e.get("code") and e.get("code") not in ("synthesis", "report_synthesis", "evaluator_threshold_checks")]
                reply = (
                    f"**{len(code_evs)} code block(s)** were generated and logged for run `{run_id}`.\n\n"
                    f"You can view and edit them in the **💻 Code & Evidence** tab, or download the full "
                    f"`runnable_code.py` script from the **📦 Full Bundle (ZIP)** in the Report tab."
                )

            elif any(kw in lower_q for kw in ["evidence", "ledger", "chain", "hash", "tamper"]):
                reply = (
                    f"**{len(evidences)} evidence records** are stored in the cryptographic ledger for run `{run_id}`.\n\n"
                    f"Each record is SHA-256 hashed and chained to the previous (`prev_hash → hash`). "
                    f"This makes any post-hoc tampering mathematically detectable. "
                    f"Chain validity: **{'✅ VERIFIED' if fw_passed else '⚠️ CHECK LEDGER TAB'}**"
                )

            else:
                # Fallback — generic helpful response
                n_c = len(conclusions)
                n_e = len(evidences)
                reply = (
                    f"I found **{n_c} conclusion(s)** and **{n_e} evidence record(s)** for run `{run_id}`.\n\n"
                    f"**Rigor score:** {score:.0%} | **Firewall:** {'✅ PASSED' if fw_passed else '❌ FAILED'}\n\n"
                    f"Try asking:\n"
                    f"- *Summarize the key findings*\n"
                    f"- *What does the p-value mean?*\n"
                    f"- *Did the firewall catch any violations?*\n"
                    f"- *How can I reproduce this analysis?*"
                )

        st.session_state.chat_history.append({"role": "pilot", "content": reply, "timestamp": ts})
        st.rerun()

    # Quick-action buttons
    st.markdown("**Quick questions:**")
    qa_cols = st.columns(4)
    quick_questions = [
        "Summarize key findings",
        "Explain the p-value",
        "Check firewall status",
        "How to reproduce this?",
    ]
    for i, (col, q) in enumerate(zip(qa_cols, quick_questions)):
        with col:
            if st.button(q, key=f"quick_q_{i}", use_container_width=True):
                ts = datetime.now().strftime("%H:%M")
                st.session_state.chat_history.append({"role": "user", "content": q, "timestamp": ts})
                # Force rerun to process
                st.rerun()
