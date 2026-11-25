"""Page 3: Multi-File Resume Upload & Structured Information Extraction."""

import io
import zipfile
import streamlit as st

from app.components.cards import render_skill_pills
from nlp.document_loader import load_document
from nlp.resume_parser import parse_resume

st.set_page_config(page_title="Resume Upload & NLP Extraction", layout="wide")

st.session_state.setdefault("parsed_resumes", [])

st.title("Resume Upload & NLP Extraction")
st.write(
    "Upload one or multiple candidate resumes (`.pdf`, `.docx`, or a `.zip` archive). "
    "Files are validated, extracted, and parsed into structured JSON."
)


def _iter_candidate_files(uploaded_items):
    """Yield (filename, raw_bytes) pairs from direct files and .zip archives."""
    for item in uploaded_items:
        raw = item.getvalue()
        if item.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        inner_name = info.filename.split("/")[-1]
                        if not inner_name or inner_name.startswith("._"):
                            continue
                        if inner_name.lower().endswith((".pdf", ".docx")):
                            yield inner_name, zf.read(info)
            except zipfile.BadZipFile:
                st.error(f"**{item.name}**: Invalid or corrupted ZIP file.")
        else:
            yield item.name, raw


uploaded_files = st.file_uploader(
    "Drag and drop resumes here",
    type=["pdf", "docx", "zip"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Process Uploaded Resumes", type="primary"):
    expanded_files = list(_iter_candidate_files(uploaded_files))
    if not expanded_files:
        st.warning("No `.pdf` or `.docx` resumes found in the uploaded files.")
    else:
        progress = st.progress(0)
        newly_parsed = []
        total = len(expanded_files)

        for idx, (fname, file_bytes) in enumerate(expanded_files):
            doc = load_document(file_bytes, fname)
            if doc.error and not doc.raw_text:
                st.error(f"**{fname}**: {doc.error}")
            else:
                if doc.is_scanned_warning:
                    st.warning(f"**{fname}**: {doc.error}")
                parsed = parse_resume(doc.raw_text, fname)
                parsed["filename"] = fname
                if "db" in st.session_state and hasattr(st.session_state.db, "save_resume"):
                    st.session_state.db.save_resume(fname, parsed)
                newly_parsed.append(parsed)
            progress.progress((idx + 1) / total)

        if newly_parsed:
            existing_names = {r.get("filename") for r in st.session_state.parsed_resumes}
            for item in newly_parsed:
                if item["filename"] not in existing_names:
                    st.session_state.parsed_resumes.append(item)
                    existing_names.add(item["filename"])
            st.success(f"Successfully parsed {len(newly_parsed)} resume(s).")

st.divider()
parsed_list = st.session_state.get("parsed_resumes", [])
st.subheader(f"Extracted Candidate Profiles ({len(parsed_list)})")

for cand in parsed_list:
    with st.expander(
        f"📄 {cand.get('candidate_name', 'Unknown')} — "
        f"{cand.get('education_level', 'N/A')} "
        f"({cand.get('estimated_experience_years', 0)} yrs exp)"
    ):
        col1, col2 = st.columns(2)
        col1.write(f"**Email:** {cand.get('email', 'N/A')}")
        col1.write(f"**Phone:** {cand.get('phone', 'N/A')}")
        col2.write(f"**Education:** {cand.get('education_level', 'N/A')}")
        col2.write(f"**Estimated Experience:** {cand.get('estimated_experience_years', 0)} years")
        st.write("**Technical Skills:**")
        render_skill_pills(cand.get("technical_skills", []))
        with st.expander("View Raw Extracted JSON"):
            display_json = {k: v for k, v in cand.items() if k != "cleaned_text"}
            st.json(display_json)
