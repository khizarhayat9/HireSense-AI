"""Page 4: Job Description Analyzer."""

import streamlit as st
from app.components.cards import render_skill_pills
from config import SAMPLE_JOBS_DIR
from nlp.document_loader import load_document
from nlp.job_analyzer import analyze_job_description

st.title("Job Description Analyzer")

input_mode = st.radio("Input Source", ["Paste Job Description", "Upload JD Document (.pdf / .docx)", "Load Sample ML Engineer JD"], horizontal=True)

raw_jd_text = ""
if input_mode == "Paste Job Description":
    raw_jd_text = st.text_area("Paste full job description text", height=240)
elif input_mode == "Upload JD Document (.pdf / .docx)":
    jd_file = st.file_uploader("Upload job description", type=["pdf", "docx"])
    if jd_file:
        doc = load_document(jd_file.read(), jd_file.name)
        raw_jd_text = doc.raw_text
else:
    sample_path = SAMPLE_JOBS_DIR / "junior_ml_engineer.txt"
    raw_jd_text = st.text_area("Sample Job Description", value=sample_path.read_text(encoding="utf-8"), height=240)

if st.button("Extract Job Requirements", type="primary"):
    if not raw_jd_text.strip():
        st.error("Please provide a job description first.")
    else:
        parsed_jd = analyze_job_description(raw_jd_text)
        st.session_state.parsed_jd = parsed_jd
        st.session_state.db.save_job_description(raw_jd_text, parsed_jd)
        st.success(f"Analyzed requirements for **{parsed_jd['job_title']}**.")

if st.session_state.parsed_jd:
    jd = st.session_state.parsed_jd
    st.divider()
    st.subheader(f"Structured Requirements: {jd['job_title']}")
    c1, c2 = st.columns(2)
    c1.metric("Minimum Experience", f"{jd['min_experience_years']} Years")
    c2.metric("Minimum Education", jd["min_education_label"])

    st.write("**Required Technical Skills:**")
    render_skill_pills(jd["required_technical_skills"], "#DBEAFE", "#1E40AF")

    st.write("**Preferred / Bonus Skills:**")
    render_skill_pills(jd["preferred_technical_skills"], "#F3E8FF", "#6B21A8")

    st.write("**Soft Skills:**")
    render_skill_pills(jd["soft_skills"], "#DCFCE7", "#166534")
