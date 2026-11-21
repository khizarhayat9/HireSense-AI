"""Page 3: Multi-File Resume Upload & Structured Information Extraction."""

import streamlit as st
from app.components.cards import render_skill_pills
from nlp.document_loader import load_document
from nlp.resume_parser import parse_resume

st.title("Resume Upload & NLP Extraction")
st.write("Upload one or multiple candidate resumes (`.pdf` or `.docx`). Files are validated, extracted, and parsed into structured JSON.")

uploaded_files = st.file_uploader(
    "Drag and drop resumes here",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Process Uploaded Resumes", type="primary"):
    progress = st.progress(0)
    newly_parsed = []
    for idx, uf in enumerate(uploaded_files):
        doc = load_document(uf.read(), uf.name)
        if doc.error and not doc.raw_text:
            st.error(f"**{uf.name}**: {doc.error}")
        else:
            if doc.is_scanned_warning:
                st.warning(f"**{uf.name}**: {doc.error}")
            parsed = parse_resume(doc.raw_text, uf.name)
            parsed["filename"] = uf.name
            st.session_state.db.save_resume(uf.name, parsed)
            newly_parsed.append(parsed)
        progress.progress((idx + 1) / len(uploaded_files))

    if newly_parsed:
        existing_names = {r["filename"] for r in st.session_state.parsed_resumes}
        for item in newly_parsed:
            if item["filename"] not in existing_names:
                st.session_state.parsed_resumes.append(item)
        st.success(f"Successfully parsed {len(newly_parsed)} resume(s).")

st.divider()
st.subheader(f"Extracted Candidate Profiles ({len(st.session_state.parsed_resumes)})")

for cand in st.session_state.parsed_resumes:
    with st.expander(f"📄 {cand['candidate_name']} — {cand['education_level']} ({cand['estimated_experience_years']} yrs exp)"):
        col1, col2 = st.columns(2)
        col1.write(f"**Email:** {cand['email']}")
        col1.write(f"**Phone:** {cand['phone']}")
        col2.write(f"**Education:** {cand['education_level']}")
        col2.write(f"**Estimated Experience:** {cand['estimated_experience_years']} years")
        st.write("**Technical Skills:**")
        render_skill_pills(cand["technical_skills"])
        with st.expander("View Raw Extracted JSON"):
            display_json = {k: v for k, v in cand.items() if k != "cleaned_text"}
            st.json(display_json)
