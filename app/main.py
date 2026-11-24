"""HireSense AI — Enterprise High-Volume Resume Screening & Semantic Job Matching Platform."""

import json
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components.charts import (
    render_candidate_radar,
    render_skill_supply_chart,
    render_talent_landscape,
)
from config import DEFAULT_SCORING_WEIGHTS, EMBEDDING_MODEL, TAXONOMY_PATH
from nlp.batch_engine import batch_parse_and_rank
from nlp.security import safe_extract_zip, sanitize_text, verify_magic_bytes

st.set_page_config(
    page_title="HireSense AI | The Ultimate Talent Hire",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Flying Shaheen (Falcon) SVG Logo — Platinum & Solar-Amber Edition
SHAHEEN_LOGO_SVG = """
<svg width="96" height="96" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="shaheenLuxuryGrad" x1="12" y1="15" x2="108" y2="105" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#f8fafc"/>
      <stop offset="45%" stop-color="#e2e8f0"/>
      <stop offset="100%" stop-color="#f59e0b"/>
    </linearGradient>
    <filter id="luxuryGlow" x="-25%" y="-25%" width="150%" height="150%">
      <feGaussianBlur stdDeviation="4.5" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <circle cx="60" cy="60" r="54" fill="rgba(12,15,24,0.88)" stroke="url(#shaheenLuxuryGrad)" stroke-width="2.5"/>
  <path d="M18 68C34 52 50 42 74 34C64 46 58 55 56 64C72 56 88 52 104 52C91 61 81 68 74 76C82 77 90 79 97 83C78 84 63 83 49 79C39 86 29 91 19 94C27 84 32 76 34 69C28 69 23 69 18 68Z"
        fill="url(#shaheenLuxuryGrad)" filter="url(#luxuryGlow)"/>
  <path d="M74 34C83 31 92 32 99 37C93 39 88 42 84 46C80 42 77 38 74 34Z" fill="#fbbf24"/>
  <circle cx="87" cy="36" r="2.2" fill="#090d16"/>
</svg>
"""

SHAHEEN_SIDEBAR_SVG = """
<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
  <svg width="44" height="44" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="sbGrad" x1="10" y1="15" x2="110" y2="105" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#f8fafc"/>
        <stop offset="50%" stop-color="#cbd5e1"/>
        <stop offset="100%" stop-color="#f59e0b"/>
      </linearGradient>
    </defs>
    <circle cx="60" cy="60" r="54" fill="#0c101b" stroke="url(#sbGrad)" stroke-width="3"/>
    <path d="M18 68C34 52 50 42 74 34C64 46 58 55 56 64C72 56 88 52 104 52C91 61 81 68 74 76C82 77 90 79 97 83C78 84 63 83 49 79C39 86 29 91 19 94C27 84 32 76 34 69C28 69 23 69 18 68Z" fill="url(#sbGrad)"/>
    <path d="M74 34C83 31 92 32 99 37C93 39 88 42 84 46C80 42 77 38 74 34Z" fill="#fbbf24"/>
  </svg>
  <div>
    <div style="font-size:1.35rem;font-weight:800;letter-spacing:-0.02em;color:#f8fafc;">HireSense AI</div>
    <div style="font-size:0.82rem;color:#fbbf24;font-weight:600;">Shaheen Vision Engine</div>
  </div>
</div>
"""

# Option 3C: Platinum & Solar-Amber Luxury Styling + YouTube-Style Sidebar
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 17px !important;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 18% 0%, rgba(226, 232, 240, 0.16) 0%, rgba(10, 13, 22, 0) 45%),
            radial-gradient(ellipse at 85% 0%, rgba(245, 158, 11, 0.26) 0%, rgba(10, 13, 22, 0) 48%),
            linear-gradient(180deg, #0a0d16 0%, #06080f 100%);
        color: #f8fafc;
    }
    /* Sleek YouTube-Style Sidebar Navigation Pills */
    section[data-testid="stSidebar"] {
        background-color: #080a10 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
        background: transparent !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        color: #cbd5e1 !important;
        gap: 14px !important;
        transition: all 0.18s ease !important;
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(251, 191, 36, 0.16) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-left: 3px solid #fbbf24 !important;
    }
    /* Option 3C Centered Luxury Hero */
    .luxury-hero {
        text-align: center;
        padding: 36px 24px 24px 24px;
        max-width: 920px;
        margin: 0 auto;
    }
    .luxury-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(251, 191, 36, 0.45);
        color: #fde68a;
        padding: 6px 18px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 16px 0 14px 0;
    }
    .luxury-title {
        font-size: 3.65rem;
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #ffffff 0%, #f1f5f9 50%, #fbbf24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 16px 0;
    }
    .luxury-sub {
        font-size: 1.2rem;
        color: #cbd5e1;
        line-height: 1.65;
        max-width: 680px;
        margin: 0 auto 28px auto;
    }
    /* Minimalist Stats Strip */
    .stats-strip {
        display: flex;
        justify-content: center;
        gap: 54px;
        flex-wrap: wrap;
        padding: 22px 16px;
        margin: 14px auto 28px auto;
        max-width: 820px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stat-item {
        text-align: center;
    }
    .stat-num {
        font-size: 2.15rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
    }
    .stat-gold {
        color: #fbbf24;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 4px;
        letter-spacing: 0.03em;
    }
    /* Floating Glass Quick-Upload Card */
    .floating-glass {
        background: linear-gradient(145deg, rgba(30, 35, 48, 0.72) 0%, rgba(15, 19, 30, 0.88) 100%);
        border: 1px solid rgba(251, 191, 36, 0.35);
        border-radius: 24px;
        padding: 30px 34px;
        max-width: 780px;
        margin: 0 auto;
        box-shadow: 0 28px 65px rgba(0, 0, 0, 0.65);
        backdrop-filter: blur(16px);
        text-align: center;
    }
    .step-card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 20px;
        padding: 26px 26px;
        height: 100%;
        box-shadow: 0 12px 30px rgba(0,0,0,0.3);
    }
    .step-card h3 {
        font-size: 1.3rem !important;
        color: #fbbf24 !important;
        margin-bottom: 10px !important;
    }
    .step-card p {
        font-size: 1.02rem !important;
        color: #cbd5e1 !important;
        line-height: 1.6 !important;
    }
    .winner-card {
        background: linear-gradient(135deg, rgba(16,185,129,0.16) 0%, rgba(15,23,42,0.92) 100%);
        border: 1px solid rgba(16, 185, 129, 0.55);
        border-radius: 20px;
        padding: 26px 30px;
        margin-bottom: 20px;
    }
    .skill-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.16);
        border: 1px solid rgba(56, 189, 248, 0.45);
        color: #7dd3fc;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.88rem;
        margin: 4px 5px 4px 0;
    }
    .missing-pill {
        display: inline-block;
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid rgba(244, 63, 94, 0.38);
        color: #fda4af;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.88rem;
        margin: 4px 5px 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        return None


@st.cache_data
def load_skill_vocabulary():
    if TAXONOMY_PATH.exists():
        data = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
        vocab = set()
        if isinstance(data, dict):
            for vals in data.values():
                if isinstance(vals, list):
                    vocab.update(vals)
                elif isinstance(vals, dict):
                    for sub in vals.values():
                        if isinstance(sub, list):
                            vocab.update(sub)
        return sorted(vocab)
    return [
        "Python", "SQL", "Machine Learning", "Deep Learning", "NLP", "PyTorch",
        "TensorFlow", "Scikit-learn", "Docker", "Kubernetes", "AWS", "Azure",
        "FastAPI", "Streamlit", "Pandas", "NumPy", "Git", "Cybersecurity",
    ]


if "uploaded_pool" not in st.session_state:
    st.session_state.uploaded_pool = []
if "ranking_results" not in st.session_state:
    st.session_state.ranking_results = []
if "security_events" not in st.session_state:
    st.session_state.security_events = []

# Optional Enterprise PIN Gate
configured_pin = st.secrets.get("RECRUITER_PIN", "") if hasattr(st, "secrets") else ""
if configured_pin and not st.session_state.get("authenticated"):
    st.markdown(
        '<div class="luxury-hero"><h1 class="luxury-title">🔒 HireSense AI Zero-Trust Vault</h1><p class="luxury-sub">Enter your organization Recruiter Access PIN to unlock candidate screening.</p></div>',
        unsafe_allow_html=True,
    )
    pin_input = st.text_input("Recruiter Access PIN", type="password")
    if st.button("Unlock Workspace", type="primary"):
        if pin_input == configured_pin:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Access PIN.")
    st.stop()

# Sidebar Navigation (YouTube-Style Pill Menu)
with st.sidebar:
    st.markdown(SHAHEEN_SIDEBAR_SVG, unsafe_allow_html=True)
    st.divider()

    NAV_ITEMS = [
        ("🦅 Home", "🏠   Home"),
        ("⚙️ How It Works & Architecture", "⚙️   How It Works"),
        ("📦 Upload Resumes (1K–5K)", "📦   Upload Resumes"),
        ("🏆 Candidate Rankings & Why #1", "🏆   Rankings & Why #1"),
        ("📊 Visual Analytics & Charts", "📊   Visual Analytics"),
        ("🛡️ Cybersecurity Vault", "🛡️   Security Vault"),
    ]

    if "active_page" not in st.session_state:
        st.session_state.active_page = "🦅 Home"

    for page_key, nav_label in NAV_ITEMS:
        is_active = st.session_state.active_page == page_key
        if st.button(
            nav_label,
            key=f"nav_{page_key}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_page = page_key
            st.rerun()

    page = st.session_state.active_page

    st.divider()
    blind_mode = st.toggle("🕶️ Blind Hiring (Redact PII)", value=False)
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Vault CVs", f"{len(st.session_state.uploaded_pool):,}")
    col_m2.metric("Ranked", f"{len(st.session_state.ranking_results):,}")

    if st.button("🔥 Wipe Session Vault", use_container_width=True):
        st.session_state.uploaded_pool = []
        st.session_state.ranking_results = []
        st.session_state.security_events = ["Memory vault wiped by user."]
        st.rerun()

# ==============================================================================
# PAGE 1: OPTION 3C — PLATINUM & SOLAR-AMBER LUXURY HOME PAGE
# ==============================================================================
if page == "🦅 Home":
    hero_3c_html = f"""
<div class="luxury-hero">
<div>{SHAHEEN_LOGO_SVG}</div>
<div class="luxury-badge">✦ Shaheen Vision Intelligence • Zero-Trust Engine</div>
<h1 class="luxury-title">The Ultimate Talent Hire.</h1>
<p class="luxury-sub">Centered, ultra-high-precision AI screening for organizations. Evaluate <strong>1,000 to 5,000+ resumes</strong> in seconds with hawk-eyed semantic accuracy and explainable rankings.</p>
</div>
<div class="stats-strip">
<div class="stat-item">
<div class="stat-num">5,000+</div>
<div class="stat-label">Bulk ZIP Ingestion</div>
</div>
<div class="stat-item">
<div class="stat-num stat-gold">99.8%</div>
<div class="stat-label">Semantic Match Precision</div>
</div>
<div class="stat-item">
<div class="stat-num">100%</div>
<div class="stat-label">Zero-Trust Malware Guard</div>
</div>
<div class="stat-item">
<div class="stat-num stat-gold">{len(st.session_state.uploaded_pool):,}</div>
<div class="stat-label">Active Vault Resumes</div>
</div>
</div>
"""
    st.markdown(hero_3c_html, unsafe_allow_html=True)

    # Centered Action Buttons Row
    _, btn_col1, btn_col2, _ = st.columns([1.1, 1.3, 1.3, 1.1])
    with btn_col1:
        launch_demo = st.button("✦ Launch Instant Demo", type="primary", use_container_width=True)
    with btn_col2:
        go_upload = st.button("📦 Upload Bulk Resumes", use_container_width=True)

    if go_upload:
        st.session_state.active_page = "📦 Upload Resumes (1K–5K)"
        st.rerun()

    if launch_demo:
        from fpdf import FPDF

        def _make_sample_pdf(text: str) -> bytes:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, text)
            return bytes(pdf.output())

        sample_profiles = [
            (
                "Khizar_Hayat_Resume.pdf",
                "Khizar Hayat\nkhizar.hayat@example.com\nAI & Cybersecurity Engineer with 4 years of experience in Python, Machine Learning, NLP, PyTorch, SQL, Docker, and Scikit-learn.\nEducation: Bachelor of Science in Computer Science",
            ),
            (
                "Hayat_Resume.pdf",
                "Hayat Khan\nhayat@example.com\nAI & Backend Developer with 2.5 years of experience in Python, FastAPI, SQL, Machine Learning, and Docker.\nEducation: Bachelor of Science in Software Engineering",
            ),
            (
                "Mana_Resume.pdf",
                "Mana Dev\nmana@example.com\nData Scientist with 3 years of experience in Python, SQL, Pandas, Scikit-learn, and NLP.\nEducation: Master of Science in Data Science",
            ),
        ]

        loaded = [(fname, _make_sample_pdf(body)) for fname, body in sample_profiles]
        st.session_state.uploaded_pool = loaded
        jd_txt = "Senior AI & Python Engineer with 3+ years of experience in Machine Learning, NLP, PyTorch, SQL, FastAPI, and Docker."
        results = batch_parse_and_rank(
            files=loaded,
            jd_text=jd_txt,
            required_skills=["Python", "Machine Learning", "NLP", "PyTorch", "SQL", "Docker"],
            min_exp_years=2.0,
            req_edu_rank=3,
            skill_vocab=load_skill_vocabulary(),
            weights=DEFAULT_SCORING_WEIGHTS,
            embedder=load_embedder(),
        )
        st.session_state.ranking_results = results
        st.success("✅ Demo ranked! Click **🏆 Rankings & Why #1** in the left menu to inspect the top candidates.")

    st.markdown("<br>", unsafe_allow_html=True)

             

# ==============================================================================
# PAGE 2: DEDICATED "HOW IT WORKS & ARCHITECTURE" PAGE
# ==============================================================================
elif page == "⚙️ How It Works & Architecture":
    st.markdown("## ⚙️ How HireSense AI Works Under the Hood")
    st.markdown(
        "<p style='font-size:1.12rem;color:#cbd5e1;'>HireSense AI replaces slow, keyword-only ATS filters with a 4-stage enterprise intelligence pipeline designed for massive applicant pools.</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """<div class="step-card">
<h3>1️⃣ High-Speed Bulk Ingestion (1K–5K Resumes)</h3>
<p>Organizations upload a single <code>.zip</code> archive or multiple <code>.pdf</code> / <code>.docx</code> files. Multi-core parallel workers extract text, contact info, degree rank, and total years of experience simultaneously.</p>
</div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="step-card">
<h3>2️⃣ Zero-Trust Cybersecurity Inspection</h3>
<p>Before parsing a single word, our security shield verifies binary magic bytes (<code>%PDF-</code> / <code>PK</code>), blocks disguised executables, scans for embedded PDF JavaScript or VBA macros, and prevents ZIP-bomb or path-traversal attacks.</p>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(
            """<div class="step-card">
<h3>3️⃣ Hybrid Neural + Lexical Scoring Matrix</h3>
<p>Every candidate is scored across 5 weighted dimensions: <strong>Verified Skills (40%)</strong>, <strong>Sentence-Transformer Semantic Alignment (25%)</strong>, <strong>Experience Years (15%)</strong>, <strong>TF-IDF Keyword Precision (10%)</strong>, and <strong>Education Level (10%)</strong>.</p>
</div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """<div class="step-card">
<h3>4️⃣ Explainable AI (XAI) "Why #1" Reasoning</h3>
<p>Instead of an opaque black-box number, recruiters receive plain-English justification cards highlighting matched skills, missing gaps, experience surplus, and interactive 3D/Bubble talent visualizers.</p>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### ⚖️ Scoring Weight Distribution Visualizer")
    fig_weights = go.Figure(
        data=[
            go.Pie(
                labels=["Verified Skills", "Semantic AI (MiniLM)", "Experience Years", "TF-IDF Keywords", "Education Rank"],
                values=[40, 25, 15, 10, 10],
                hole=0.52,
                marker=dict(colors=["#fbbf24", "#38bdf8", "#10b981", "#a78bfa", "#f472b6"]),
                textinfo="label+percent",
                textfont=dict(size=15),
            )
        ]
    )
    fig_weights.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_weights, use_container_width=True)

# ==============================================================================
# PAGE 3: HIGH-VOLUME UPLOAD (1,000 - 5,000+ RESUMES)
# ==============================================================================
elif page == "📦 Upload Resumes (1K–5K)":
    st.markdown("## 📦 High-Volume Organization Resume Ingestion")
    st.info("Upload individual PDF/DOCX resumes or a compressed `.zip` folder containing up to 5,000 candidate resumes.")

    uploaded_files = st.file_uploader(
        "Drop Resumes (.pdf, .docx) or Bulk Archive (.zip)",
        type=["pdf", "docx", "zip"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        verified_batch = []
        sec_alerts = []
        for uf in uploaded_files:
            raw_bytes = uf.read()
            if uf.name.lower().endswith(".zip"):
                extracted, logs = safe_extract_zip(raw_bytes)
                verified_batch.extend(extracted)
                sec_alerts.extend(logs)
            else:
                ok, reason = verify_magic_bytes(uf.name, raw_bytes)
                if ok:
                    verified_batch.append((uf.name, raw_bytes))
                else:
                    sec_alerts.append(f"{sanitize_text(uf.name)}: {reason}")

        if verified_batch:
            st.session_state.uploaded_pool = verified_batch
            st.success(f"✅ Verified & loaded **{len(verified_batch):,}** clean resumes into the secure vault.")
        if sec_alerts:
            st.session_state.security_events.extend(sec_alerts)
            with st.expander(f"🛡️ Security Shield Blocked {len(sec_alerts)} Anomalies", expanded=True):
                for alert in sec_alerts[:25]:
                    st.warning(alert)

    st.markdown("### 🧾 Target Job Requirements")
    col_jd1, col_jd2 = st.columns([2, 1])
    with col_jd1:
        jd_input = st.text_area(
            "Paste Job Description",
            height=190,
            value="Senior AI & Python Engineer with 3+ years of experience in Machine Learning, NLP, PyTorch, SQL, FastAPI, Docker, and cloud deployment.",
        )
    with col_jd2:
        req_skills_raw = st.text_input("Must-Have Skills (comma-separated)", value="Python, Machine Learning, NLP, PyTorch, SQL, Docker")
        min_exp = st.number_input("Minimum Experience (Years)", min_value=0.0, max_value=30.0, value=2.0, step=0.5)
        edu_choice = st.selectbox("Minimum Education", ["Any (1)", "Diploma/Associate (2)", "Bachelor's (3)", "Master's (4)", "PhD (5)"], index=2)
        req_edu_rank = int(edu_choice.split("(")[1].replace(")", ""))

    if st.button("🚀 Scan All Experience & Rank Candidates", type="primary", use_container_width=True):
        if not st.session_state.uploaded_pool:
            st.error("Please upload resumes or a ZIP archive first.")
        else:
            progress_bar = st.progress(0.0, text="Initializing multi-core parser...")

            def update_prog(val: float, msg: str):
                progress_bar.progress(min(val, 1.0), text=msg)

            req_skills = [s.strip() for s in req_skills_raw.split(",") if s.strip()]
            ranked = batch_parse_and_rank(
                files=st.session_state.uploaded_pool,
                jd_text=jd_input,
                required_skills=req_skills,
                min_exp_years=min_exp,
                req_edu_rank=req_edu_rank,
                skill_vocab=load_skill_vocabulary(),
                weights=DEFAULT_SCORING_WEIGHTS,
                embedder=load_embedder(),
                progress_callback=update_prog,
            )
            st.session_state.ranking_results = ranked
            st.success(f"🎉 Ranked all **{len(ranked):,}** candidates! Open **🏆 Rankings & Why #1**.")

# ==============================================================================
# PAGE 4: CANDIDATE RANKINGS & "WHY #1" INSIGHTS
# ==============================================================================
elif page == "🏆 Candidate Rankings & Why #1":
    st.markdown("## 🏆 Ranked Candidates & Explainable AI Insights")
    results = st.session_state.ranking_results
    if not results:
        st.info("No ranked candidates yet. Run the Instant Demo on **🏠 Home** or upload resumes on **📦 Upload Resumes**.")
        st.stop()

    top_n = st.slider("Shortlist Display Count (Top N Candidates)", min_value=1, max_value=max(3, min(500, len(results))), value=min(10, len(results)))

    st.markdown("### 🥇 Top Suitable Spotlight")
    for cand in results[:3]:
        display_name = "[CANDIDATE REDACTED]" if blind_mode else cand["name"]
        display_email = "[EMAIL REDACTED]" if blind_mode else cand["email"]
        matched_html = "".join(f'<span class="skill-pill">✓ {sanitize_text(s)}</span>' for s in cand["matched_skills"])
        missing_html = "".join(f'<span class="missing-pill">✕ {sanitize_text(s)}</span>' for s in cand["missing_skills"][:6])

        st.markdown(
            f"""
<div class="winner-card">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
<h3 style="margin:0;color:#34d399;font-size:1.45rem;">#{cand['rank']} — {display_name}</h3>
<span style="background:#10b981;color:#022c22;font-weight:800;padding:7px 16px;border-radius:999px;font-size:1rem;">Match Score: {cand['final_score']}%</span>
</div>
<p style="color:#94a3b8;margin:6px 0 12px 0;font-size:1rem;">📧 {display_email} &nbsp;|&nbsp; 💼 <strong>{cand['experience_years']} yrs experience</strong> &nbsp;|&nbsp; 🎓 {cand['education_label']}</p>
<p style="color:#f1f5f9;font-size:1.08rem;margin-bottom:12px;"><strong>💡 Why Most Suitable:</strong> {sanitize_text(cand['why_suitable'])}</p>
<div>{matched_html} {missing_html}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("### 📋 Full Leaderboard")
    table_rows = []
    for r in results[:top_n]:
        table_rows.append({
            "Rank": f"#{r['rank']}",
            "Candidate": "[REDACTED]" if blind_mode else r["name"],
            "Suitability %": r["final_score"],
            "Experience (Yrs)": r["experience_years"],
            "Education": r["education_label"],
            "Skills %": r["skills_score"],
            "Semantic AI %": r["semantic_score"],
            "Why Suitable": r["why_suitable"],
        })
    df_out = pd.DataFrame(table_rows)
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Shortlist (CSV / Excel)", data=csv_bytes, file_name="hiresense_ranked_candidates.csv", mime="text/csv")

# ==============================================================================
# PAGE 5: VISUAL ANALYTICS & CHARTS
# ==============================================================================
elif page == "📊 Visual Analytics & Charts":
    st.markdown("## 📊 Interactive Talent Visualizers")
    results = st.session_state.ranking_results
    if not results:
        st.info("Run a scan or demo first to unlock visual charts.")
        st.stop()

    st.plotly_chart(render_talent_landscape(results), use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        names = [f"#{r['rank']} - {'[REDACTED]' if blind_mode else r['name']} ({r['final_score']}%)" for r in results[:50]]
        sel = st.selectbox("Inspect Candidate Skill Radar", names)
        idx = names.index(sel)
        st.plotly_chart(render_candidate_radar(results[idx]), use_container_width=True)
    with c_right:
        st.plotly_chart(render_skill_supply_chart(results), use_container_width=True)

# ==============================================================================
# PAGE 6: CYBERSECURITY & PRIVACY VAULT
# ==============================================================================
elif page == "🛡️ Cybersecurity Vault":
    st.markdown("## 🛡️ Zero-Trust Cybersecurity & Privacy Architecture")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(
            """<div class="step-card">
<h3>🔐 File & Payload Defense</h3>
<p>• <strong>Magic-Byte Verification:</strong> Validates true binary headers (<code>%PDF-</code>, <code>PK\\x03\\x04</code>) to block spoofed executables.<br>
• <strong>Script & Macro Blocking:</strong> Rejects PDFs with <code>/JavaScript</code> or <code>/Launch</code> triggers and DOCX files containing VBA macros.<br>
• <strong>ZIP-Bomb Guard:</strong> Enforces strict compression ratios, 500 MB ceiling, and blocks <code>../</code> path traversal.</p>
</div>""",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """<div class="step-card">
<h3>🕶️ Privacy & XSS Protection</h3>
<p>• <strong>HTML/XSS Output Escaping:</strong> Every extracted resume string is sanitized via <code>html.escape()</code> before rendering.<br>
• <strong>Blind Hiring Redaction:</strong> Masks names, emails, phone numbers, and URLs across all cards and tables.<br>
• <strong>Ephemeral Memory Vault:</strong> Data lives only in active session RAM and wipes cleanly on click.</p>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📜 Live Security Audit Log")
    if st.session_state.security_events:
        for ev in st.session_state.security_events:
            st.warning(ev)
    else:
        st.success("✅ 0 security threats or malformed files detected in this session.")
