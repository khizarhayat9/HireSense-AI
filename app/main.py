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
    page_title="HireSense AI | Shaheen Talent Intelligence",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Flying Shaheen (Falcon) SVG Logo
SHAHEEN_LOGO_SVG = """
<svg width="76" height="76" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="shaheenGrad" x1="10" y1="15" x2="110" y2="105" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="55%" stop-color="#818cf8"/>
      <stop offset="100%" stop-color="#fbbf24"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3.5" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <circle cx="60" cy="60" r="54" fill="rgba(15,23,42,0.85)" stroke="url(#shaheenGrad)" stroke-width="2.5"/>
  <!-- Soaring Shaheen Wings & Sharp Gaze Profile -->
  <path d="M18 68C34 52 50 42 74 34C64 46 58 55 56 64C72 56 88 52 104 52C91 61 81 68 74 76C82 77 90 79 97 83C78 84 63 83 49 79C39 86 29 91 19 94C27 84 32 76 34 69C28 69 23 69 18 68Z"
        fill="url(#shaheenGrad)" filter="url(#glow)"/>
  <!-- Sharp Falcon Head & Vision Beam -->
  <path d="M74 34C83 31 92 32 99 37C93 39 88 42 84 46C80 42 77 38 74 34Z" fill="#fbbf24"/>
  <circle cx="87" cy="36" r="2" fill="#0f172a"/>
</svg>
"""

SHAHEEN_SIDEBAR_SVG = """
<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
  <svg width="44" height="44" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="sbGrad" x1="10" y1="15" x2="110" y2="105" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#38bdf8"/>
        <stop offset="60%" stop-color="#818cf8"/>
        <stop offset="100%" stop-color="#fbbf24"/>
      </linearGradient>
    </defs>
    <circle cx="60" cy="60" r="54" fill="#0f172a" stroke="url(#sbGrad)" stroke-width="3"/>
    <path d="M18 68C34 52 50 42 74 34C64 46 58 55 56 64C72 56 88 52 104 52C91 61 81 68 74 76C82 77 90 79 97 83C78 84 63 83 49 79C39 86 29 91 19 94C27 84 32 76 34 69C28 69 23 69 18 68Z" fill="url(#sbGrad)"/>
    <path d="M74 34C83 31 92 32 99 37C93 39 88 42 84 46C80 42 77 38 74 34Z" fill="#fbbf24"/>
  </svg>
  <div>
    <div style="font-size:1.35rem;font-weight:800;letter-spacing:-0.02em;color:#f8fafc;">HireSense AI</div>
    <div style="font-size:0.82rem;color:#38bdf8;font-weight:600;">Shaheen Vision Engine</div>
  </div>
</div>
"""

# Global Typography & Spacious Executive Styling (Larger, Cleaner, Anti-Doc Layout)
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 17px !important;
    }
    .stApp {
        background: radial-gradient(circle at 15% 12%, #131e3a 0%, #080d1a 58%, #050811 100%);
        color: #f8fafc;
    }
    /* Larger, cleaner sidebar radio items */
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1.05rem !important;
        padding: 8px 6px !important;
        font-weight: 500 !important;
    }
    /* Spacious Landing Hero */
    .landing-hero {
        background: linear-gradient(135deg, rgba(30,41,59,0.78) 0%, rgba(15,23,42,0.92) 60%, rgba(30,58,138,0.45) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 26px;
        padding: 48px 48px;
        margin: 10px 0 30px 0;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.55);
    }
    .landing-title {
        font-size: 3.1rem;
        font-weight: 900;
        line-height: 1.15;
        background: linear-gradient(90deg, #ffffff 0%, #7dd3fc 55%, #fbbf24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 14px 0;
    }
    .landing-tagline {
        font-size: 1.25rem;
        color: #cbd5e1;
        line-height: 1.65;
        max-width: 780px;
        margin-bottom: 26px;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 8px;
    }
    .feature-pill {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.38);
        color: #e0f2fe;
        padding: 8px 18px;
        border-radius: 999px;
        font-size: 0.95rem;
        font-weight: 600;
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
        color: #38bdf8 !important;
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
    st.markdown('<div class="landing-hero"><h1 class="landing-title">🔒 HireSense AI Zero-Trust Vault</h1><p class="landing-tagline">Enter your organization Recruiter Access PIN to unlock candidate screening.</p></div>', unsafe_allow_html=True)
    pin_input = st.text_input("Recruiter Access PIN", type="password")
    if st.button("Unlock Workspace", type="primary"):
        if pin_input == configured_pin:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Access PIN.")
    st.stop()

# Sidebar Navigation
with st.sidebar:
    st.markdown(SHAHEEN_SIDEBAR_SVG, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🦅 Home",
            "⚙️ How It Works & Architecture",
            "📦 Upload Resumes (1K–5K)",
            "🏆 Candidate Rankings & Why #1",
            "📊 Visual Analytics & Charts",
            "🛡️ Cybersecurity Vault",
        ],
    )

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
# PAGE 1: CLEAN, HIGH-IMPACT HOME PAGE (NO CLUTTERED DOC TEXT)
# ==============================================================================
if page == "🦅 Home":
    hero_html = f"""
<div class="landing-hero">
<div style="display:flex;align-items:center;gap:22px;margin-bottom:18px;flex-wrap:wrap;">
{SHAHEEN_LOGO_SVG}
<div>
<div style="color:#fbbf24;font-weight:700;letter-spacing:0.08em;font-size:0.9rem;text-transform:uppercase;">Sharp Vision • Precision Talent Intelligence</div>
<h1 class="landing-title">HireSense AI</h1>
</div>
</div>
<p class="landing-tagline">Spot top talent with hawk-eyed precision. Screen <strong>1,000 to 5,000+ resumes</strong> in seconds using hybrid Neural AI, automated experience scoring, and crystal-clear suitability insights.</p>
<div class="pill-row">
<span class="feature-pill">⚡ 5,000+ Bulk ZIP Screening</span>
<span class="feature-pill">🧠 Sentence-Transformer + TF-IDF</span>
<span class="feature-pill">💡 Explainable "Why #1" Cards</span>
<span class="feature-pill">🛡️ Zero-Trust Malware &amp; PII Shield</span>
</div>
</div>
"""
    st.markdown(hero_html, unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Batch Speed", "5,000+ CVs", "Parallel Multi-Core")
    m2.metric("AI Precision", "Hybrid NLP", "Semantic + Lexical")
    m3.metric("Security Status", "Protected", "Zero-Trust Guard")
    m4.metric("Active Pool", f"{len(st.session_state.uploaded_pool):,} Resumes", "Ready")

    st.markdown("<br>", unsafe_allow_html=True)
    col_cta1, col_cta2 = st.columns([1.2, 1])
    with col_cta1:
        if st.button("▶ Launch Instant Demo (3 Sample Candidates)", type="primary", use_container_width=True):
            from fpdf import FPDF

            def _make_sample_pdf(text: str) -> bytes:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", size=11)
                pdf.multi_cell(0, 6, text)
                return bytes(pdf.output())

            sample_profiles = [
                (
                    "Zainab_Ali_Resume.pdf",
                    "Zainab Ali\nzainab.ali@example.com\nSenior Machine Learning Engineer with 4 years of experience in Python, Machine Learning, NLP, PyTorch, SQL, Docker, and Scikit-learn.\nEducation: Master of Science in Artificial Intelligence",
                ),
                (
                    "Usman_Tariq_Resume.pdf",
                    "Usman Tariq\nusman.tariq@example.com\nAI & Backend Engineer with 2.5 years of experience in Python, FastAPI, SQL, Machine Learning, and Docker.\nEducation: Bachelor of Science in Computer Science",
                ),
                (
                    "Ayesha_Malik_Resume.pdf",
                    "Ayesha Malik\nayesha.malik@example.com\nData Scientist with 3 years of experience in Python, SQL, Pandas, Scikit-learn, and NLP.\nEducation: Master of Science in Data Science",
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
            st.success("✅ Demo ranked! Click **🏆 Candidate Rankings & Why #1** or **📊 Visual Analytics & Charts** in the left menu.")

    with col_cta2:
        st.info("👈 Ready to screen real applicants? Select **📦 Upload Resumes (1K–5K)** or learn the engine on **⚙️ How It Works & Architecture**.")

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
                marker=dict(colors=["#38bdf8", "#818cf8", "#10b981", "#fbbf24", "#f472b6"]),
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
            st.success(f"🎉 Ranked all **{len(ranked):,}** candidates! Open **🏆 Candidate Rankings & Why #1**.")

# ==============================================================================
# PAGE 4: CANDIDATE RANKINGS & "WHY #1" INSIGHTS
# ==============================================================================
elif page == "🏆 Candidate Rankings & Why #1":
    st.markdown("## 🏆 Ranked Candidates & Explainable AI Insights")
    results = st.session_state.ranking_results
    if not results:
        st.info("No ranked candidates yet. Run the Instant Demo on **🦅 Home** or upload resumes on **📦 Upload Resumes (1K–5K)**.")
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
                    <span style="background:#10b981;color:#022c22;font-weight:800;padding:7px 16px;border-radius:999px;font-size:1rem;">
                        Match Score: {cand['final_score']}%
                    </span>
                </div>
                <p style="color:#94a3b8;margin:6px 0 12px 0;font-size:1rem;">
                    📧 {display_email} &nbsp;|&nbsp; 💼 <strong>{cand['experience_years']} yrs experience</strong> &nbsp;|&nbsp; 🎓 {cand['education_label']}
                </p>
                <p style="color:#f1f5f9;font-size:1.08rem;margin-bottom:12px;">
                    <strong>💡 Why Most Suitable:</strong> {sanitize_text(cand['why_suitable'])}
                </p>
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
