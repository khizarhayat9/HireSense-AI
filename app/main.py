"""HireSense AI — Main Entry Point & Multipage Router."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from config import DEFAULT_SCORING_WEIGHTS
from database.db import DatabaseManager

st.set_page_config(
    page_title="HireSense AI | Intelligent Resume Screening",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: -0.02em; color: #0F172A; }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.15s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "db" not in st.session_state:
    st.session_state.db = DatabaseManager()
if "parsed_resumes" not in st.session_state:
    st.session_state.parsed_resumes = []
if "parsed_jd" not in st.session_state:
    st.session_state.parsed_jd = None
if "ranked_results" not in st.session_state:
    st.session_state.ranked_results = []
if "scoring_weights" not in st.session_state:
    st.session_state.scoring_weights = DEFAULT_SCORING_WEIGHTS.copy()

pages = [
    st.Page("pages/01_welcome.py", title="Welcome", icon="🏠", default=True),
    st.Page("pages/02_dashboard.py", title="Recruitment Dashboard", icon="📊"),
    st.Page("pages/03_upload_resume.py", title="Resume Upload", icon="📄"),
    st.Page("pages/04_job_analyzer.py", title="Job Description Analyzer", icon="🧾"),
    st.Page("pages/05_candidate_matching.py", title="Candidate Matching", icon="⚡"),
    st.Page("pages/06_candidate_comparison.py", title="Candidate Comparison", icon="⚖️"),
    st.Page("pages/07_candidate_reports.py", title="Candidate Reports", icon="📑"),
    st.Page("pages/08_scan_history.py", title="Scan History", icon="🕒"),
    st.Page("pages/09_settings_about.py", title="Settings & About", icon="⚙️"),
]

pg = st.navigation(pages)
with st.sidebar:
    st.markdown("### 🎯 HireSense AI")
    st.caption("Explainable NLP Talent Matching")
    st.divider()
    st.metric("Loaded Resumes", len(st.session_state.parsed_resumes))
    st.metric("Active JD", st.session_state.parsed_jd["job_title"] if st.session_state.parsed_jd else "None")

pg.run()
