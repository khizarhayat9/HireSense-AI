"""Page 1: Landing / Welcome Page with one-click demo loader."""

import streamlit as st
from config import SAMPLE_JOBS_DIR, SAMPLE_RESUMES_DIR
from nlp.document_loader import load_document
from nlp.job_analyzer import analyze_job_description
from nlp.resume_parser import parse_resume
from nlp.scoring_engine import evaluate_candidate, rank_candidates

st.markdown(
    """
    <div style="background: linear-gradient(120deg, #1E3A8A 0%, #4F46E5 55%, #7C3AED 100%);
                padding: 2.4rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;">
        <h1 style="color: white; margin-bottom: 0.4rem;">HireSense AI</h1>
        <p style="font-size: 1.12rem; opacity: 0.94; max-width: 760px;">
            Intelligent Resume Screening &amp; Semantic Job Matching Platform.
            Combine TF-IDF lexical precision, Sentence-Transformer contextual embeddings,
            and transparent skill-gap explainability.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**1. Upload Resumes**\n\nBatch-parse PDF and DOCX resumes into structured JSON profiles with scanned-PDF detection.")
with col2:
    st.info("**2. Analyze Job Description**\n\nExtract required skills, bonus tools, experience thresholds, and education rules.")
with col3:
    st.info("**3. Explainable Ranking**\n\nInspect transparent sub-scores, skill gaps, radar charts, and exportable PDF reports.")

st.divider()
st.subheader("Quick Start: Load Sample Recruitment Dataset")
st.write("Test the complete NLP pipeline immediately using pre-built resumes and a Junior ML Engineer job description.")

if st.button("Load Sample Resumes & Run Demo Match", type="primary"):
    with st.spinner("Parsing sample documents and computing semantic embeddings..."):
        jd_path = SAMPLE_JOBS_DIR / "junior_ml_engineer.txt"
        raw_jd = jd_path.read_text(encoding="utf-8")
        parsed_jd = analyze_job_description(raw_jd)
        job_id = st.session_state.db.save_job_description(raw_jd, parsed_jd)

        parsed_list = []
        evaluations = []
        seen_candidates = set()
        for resume_file in sorted(SAMPLE_RESUMES_DIR.glob("*.txt")):
            doc = load_document(resume_file)
            parsed = parse_resume(doc.raw_text, doc.filename)
            if parsed["candidate_name"] in seen_candidates:
                continue
            seen_candidates.add(parsed["candidate_name"])
            parsed["filename"] = doc.filename
            res_id = st.session_state.db.save_resume(doc.filename, parsed)
            parsed_list.append(parsed)

            ev = evaluate_candidate(parsed, parsed_jd, st.session_state.scoring_weights)
            st.session_state.db.save_scan_result(res_id, job_id, ev)
            evaluations.append(ev)

        st.session_state.parsed_jd = parsed_jd
        st.session_state.parsed_resumes = parsed_list
        st.session_state.ranked_results = rank_candidates(evaluations)
        st.success(f"Loaded and ranked {len(parsed_list)} sample candidates! Open **Recruitment Dashboard** or **Candidate Matching** from the sidebar.")
