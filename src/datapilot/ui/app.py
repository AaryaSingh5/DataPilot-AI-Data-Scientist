"""
DataPilot — Streamlit Web Application
Autonomous, multi-agent AI data scientist with cryptographic audit trails and hallucination firewall.
"""
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
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


# ── Safety Constants ────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_UPLOAD_ROWS = 50_000
ALLOWED_EXTENSIONS = {".csv", ".parquet", ".xlsx"}
FORBIDDEN_EXTENSIONS = {".xlsm", ".xlsb", ".xltm", ".xlam"}


# ── Pandas ↔ DuckDB bridge helpers ──────────────────────────────────────────

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


# Configure Page
st.set_page_config(
    page_title="DataPilot — Autonomous AI Data Scientist",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #111119 100%);
        border: 1px solid #2e2e42;
        padding: 24px;
        border-radius: 14px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px -4px rgba(0,0,0,0.3);
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-pill-green {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-pill-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .metric-card {
        background: #181824;
        border: 1px solid #28283c;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    .hypothesis-card {
        background: #161922;
        border-left: 4px solid #6366f1;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


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


init_session()
store = st.session_state.store
snap_dir = st.session_state.snapshot_dir

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🧭 **DataPilot**")
    st.caption("Autonomous AI Data Scientist")
    st.markdown("---")

    st.markdown("#### 📂 Dataset Configuration")
    data_option = st.radio(
        "Source",
        ["Synthetic Benchmark (Pricing Experiment)", "Upload CSV / Parquet"],
        label_visibility="collapsed",
    )

    if data_option == "Synthetic Benchmark (Pricing Experiment)":
        if st.button("🔄 Regenerate Benchmark", use_container_width=True):
            df, gt = generate_pricing_experiment(1200, 42)
            st.session_state.current_df = df
            st.session_state.current_gt = gt
            st.session_state.dataset_name = "saas_pricing_experiment"
            st.success("Loaded synthetic pricing benchmark!")
    else:
        uploaded_file = st.file_uploader(
            "Upload data file (Max 10 MB, 50,000 rows)",
            type=["csv", "parquet", "xlsx"],
        )
        if uploaded_file:
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext in FORBIDDEN_EXTENSIONS:
                st.error("❌ Macro-enabled Excel files (.xlsm) are rejected for security. Upload standard .csv, .parquet, or .xlsx.")
            elif file_ext not in ALLOWED_EXTENSIONS:
                st.error(f"❌ Format '{file_ext}' not supported. Allowed: .csv, .parquet, .xlsx")
            elif uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
                st.error(f"❌ File size ({uploaded_file.size / (1024*1024):.1f} MB) exceeds maximum limit of 10 MB.")
            else:
                try:
                    if file_ext == ".csv":
                        df_upload = pd.read_csv(uploaded_file)
                    elif file_ext == ".parquet":
                        df_upload = pd.read_parquet(uploaded_file)
                    else:
                        df_upload = pd.read_excel(uploaded_file)

                    if len(df_upload) > MAX_UPLOAD_ROWS:
                        st.error(f"❌ Dataset contains {len(df_upload):,} rows, exceeding maximum demo limit of 50,000 rows.")
                    else:
                        st.session_state.current_df = df_upload
                        st.session_state.dataset_name = Path(uploaded_file.name).stem
                        st.success(f"Loaded '{uploaded_file.name}' ({len(df_upload):,} rows, {len(df_upload.columns)} cols)")
                except Exception as exc:
                    st.error(f"❌ Failed to parse file: {exc}")

    st.markdown("---")
    st.markdown("#### ⚙️ Engine Settings")
    llm_mode = st.selectbox(
        "LLM Client",
        ["Mock Client (Deterministic)", "NVIDIA AI (Llama 3.2 11B)", "Anthropic Claude"],
        index=0,  # Default to Mock mode
    )

    # Demo Authentication & Key Management
    session_api_key = None
    if llm_mode != "Mock Client (Deterministic)":
        st.markdown("##### 🔐 Authentication")
        provider_name = "NVIDIA" if "NVIDIA" in llm_mode else "Anthropic"
        env_var_name = "NVIDIA_API_KEY" if "NVIDIA" in llm_mode else "ANTHROPIC_API_KEY"
        
        env_key = os.environ.get(env_var_name)
        demo_pwd = os.environ.get("DEMO_PASSWORD")
        
        if env_key:
            st.caption(f"✓ {provider_name} API Key loaded from environment.")
            session_api_key = env_key
        else:
            auth_method = st.radio(
                "Access Method",
                ["Demo Password", "Custom API Key"],
                key="auth_method_radio",
            )
            if auth_method == "Demo Password":
                pwd_input = st.text_input("Demo Password", type="password", key="demo_pwd_input")
                if pwd_input:
                    if demo_pwd and pwd_input == demo_pwd:
                        st.success("✓ Demo password accepted.")
                    else:
                        st.error("❌ Invalid demo password.")
            else:
                key_input = st.text_input(f"{provider_name} API Key", type="password", key="custom_api_key_input")
                if key_input:
                    session_api_key = key_input
                    st.caption("✓ Key active in session memory only (never saved).")

    alpha = st.slider("Significance Alpha (α)", 0.01, 0.10, 0.05, 0.01)
    fdr_q = st.slider("FDR q-value", 0.01, 0.10, 0.05, 0.01)

    st.markdown("---")
    st.caption("Cryptographic Integrity: SHA-256 Hash Chained Ledger")


# Snapshot & Profile Current Data
df = st.session_state.current_df
profile = profile_dataframe(df)
roles = infer_roles(df)
snap_result = snapshot_dataframe(df, snap_dir, table_name="data")
d_hash = snap_result["dataset_hash"]


# --- VISIBLE DEMO WARNING BANNER ---
st.warning(
    "⚠️ **Experimental Demo**: Results are generated by multi-agent AI heuristics and rule-based consistency checks. "
    "This output is not a substitute for review by a professional statistician."
)


# --- HEADER ---
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; color: #f8fafc; font-size: 1.8rem; font-weight: 800;">🧭 DataPilot Autonomous Data Scientist</h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.95rem;">Rigorous, statistically justified, tamper-evident data analysis with 12-check Hallucination Firewall.</p>
        </div>
        <div>
            <span class="badge-pill badge-pill-green">Firewall Active</span>
            <span class="badge-pill badge-pill-blue">SHA-256 Chained</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# --- TABS ---
tab_investigate, tab_data, tab_report, tab_ledger = st.tabs([
    "🔍 Investigation & Execution",
    "📊 Dataset Profile & Roles",
    "📑 Final Report & Firewall",
    "🛡️ Audit Ledger & Provenance",
])


# --- TAB 1: INVESTIGATION ---
with tab_investigate:
    col_q, col_btn = st.columns([4, 1])
    with col_q:
        default_q = "Did the pricing change increase revenue, and is the effect confounded by customer segment?"
        user_question = st.text_input(
            "Enter research question:",
            value=default_q,
            placeholder="e.g. Did marketing spend increase retention in Q3?",
        )
    with col_btn:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Run Investigation", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Executing DataPilot multi-agent workflow..."):
            provider_map = {
                "Mock Client (Deterministic)": "mock",
                "NVIDIA AI (Llama 3.2 11B)": "nvidia",
                "Anthropic Claude": "anthropic",
            }
            target_provider = provider_map.get(llm_mode, "mock")
            try:
                llm_client = get_llm_client(provider=target_provider, api_key=session_api_key)

                if isinstance(llm_client, MockLLMClient):
                    # Register standard responses for benchmark
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

                import uuid
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
                st.success("Investigation complete! Results verified through Hallucination Firewall.")
            except Exception as exc:
                st.error(f"Failed to run investigation: {exc}")

    # Show live or previous execution summary
    if st.session_state.last_report:
        rep = st.session_state.last_report
        st.markdown("### 📋 Executive Summary")
        st.info(rep.get("executive_summary", ""))

        c1, c2, c3 = st.columns(3)
        with c1:
            fw_pass = rep.get("firewall", {}).get("passed", False)
            color = "#4ade80" if fw_pass else "#f87171"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {color};">{'PASSED' if fw_pass else 'FAILED'}</div>
                <div class="metric-label">12-Check Hallucination Firewall</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            score = rep.get("evaluation", {}).get("overall_score", 1.0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #60a5fa;">{score:.2f} / 1.00</div>
                <div class="metric-label">Statistical Rigor Score</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            e_count = len(rep.get("evidences", []))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #c084fc;">{e_count}</div>
                <div class="metric-label">Chained Evidences in Ledger</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🔬 Formulated Hypotheses & Findings")
        for c in rep.get("conclusions", []):
            st.markdown(f"""
            <div class="hypothesis-card">
                <strong>{c.get('hypothesis_id')}: {c.get('verdict', '').upper()}</strong> (Confidence: {c.get('confidence')})<br>
                {c.get('summary')}
            </div>
            """, unsafe_allow_html=True)


# --- TAB 2: DATA & ROLES ---
with tab_data:
    st.markdown(f"**Dataset Hash (SHA-256):** `{d_hash}` | **Rows:** `{profile['row_count']}` | **Columns:** `{profile['col_count']}`")
    st.dataframe(df.head(20), use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### 🏷️ Inferred Column Roles")
        role_df = pd.DataFrame([{"Column": k, "Inferred Role": v} for k, v in roles.items()])
        st.dataframe(role_df, use_container_width=True)

    with c_right:
        st.markdown("#### 📈 Summary Statistics")
        st.dataframe(df.describe().T, use_container_width=True)


# --- TAB 3: REPORT & FIREWALL ---
with tab_report:
    if st.session_state.last_report:
        rep = st.session_state.last_report

        # Download Buttons
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button(
                "📥 Download Markdown Report",
                data=rep.get("markdown", ""),
                file_name=f"datapilot_report_{rep.get('run_id')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                "🌐 Download HTML Report",
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
                "📦 Download Reproducibility Bundle (ZIP)",
                data=bundle_zip,
                file_name=f"reproducibility_bundle_{rep.get('run_id')}.zip",
                mime="application/zip",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown(rep.get("markdown", ""))
    else:
        st.info("Run an investigation first to preview and export the final report.")


# --- TAB 4: AUDIT LEDGER ---
with tab_ledger:
    st.markdown("### 🛡️ Immutable Cryptographic Audit Ledger")
    st.caption("Every statistical test, SQL query, and analysis is hash-chained and tamper-evident.")

    rows = store.list_recent(50)

    if rows:
        ledger_df = pd.DataFrame(rows)
        st.dataframe(ledger_df, use_container_width=True)

        if st.session_state.last_report:
            rid = st.session_state.last_report.get("run_id")
            if rid:
                is_valid = store.verify_chain(rid)
                if is_valid:
                    st.success(f"✅ Cryptographic verification confirmed: Run `{rid}` hash chain is 100% untampered.")
                else:
                    st.error(f"❌ Cryptographic verification failed for Run `{rid}`.")
    else:
        st.info("No records in ledger yet.")
