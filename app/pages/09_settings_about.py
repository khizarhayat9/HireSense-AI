"""Page 9: Scoring Configuration & System Methodology."""

import streamlit as st

st.set_page_config(page_title="Scoring Configuration & System Methodology", layout="wide")

DEFAULT_WEIGHTS = {
    "skills": 40,
    "semantic": 25,
    "experience": 15,
    "tfidf": 10,
    "education": 10,
}

st.session_state.setdefault("scoring_weights", dict(DEFAULT_WEIGHTS))
if not isinstance(st.session_state.get("scoring_weights"), dict):
    st.session_state.scoring_weights = dict(DEFAULT_WEIGHTS)

st.title("Scoring Configuration & System Methodology")
st.subheader("Customize Scoring Engine Weights")
st.write(
    "Adjust how much weight each NLP signal contributes to the composite match score. "
    "Values are automatically normalized to 100%."
)

w = st.session_state.scoring_weights

c1, c2, c3 = st.columns(3)
skills_w = c1.slider(
    "Verified Skills Weight",
    min_value=0,
    max_value=100,
    value=int(w.get("skills", DEFAULT_WEIGHTS["skills"])),
    step=5,
)
semantic_w = c2.slider(
    "Neural Semantic Similarity Weight",
    min_value=0,
    max_value=100,
    value=int(w.get("semantic", DEFAULT_WEIGHTS["semantic"])),
    step=5,
)
exp_w = c3.slider(
    "Experience Years Weight",
    min_value=0,
    max_value=100,
    value=int(w.get("experience", DEFAULT_WEIGHTS["experience"])),
    step=5,
)

c4, c5 = st.columns(2)
tfidf_w = c4.slider(
    "TF-IDF Keyword Precision Weight",
    min_value=0,
    max_value=100,
    value=int(w.get("tfidf", DEFAULT_WEIGHTS["tfidf"])),
    step=5,
)
edu_w = c5.slider(
    "Education Level Weight",
    min_value=0,
    max_value=100,
    value=int(w.get("education", DEFAULT_WEIGHTS["education"])),
    step=5,
)

raw_weights = {
    "skills": skills_w,
    "semantic": semantic_w,
    "experience": exp_w,
    "tfidf": tfidf_w,
    "education": edu_w,
}
total_raw = sum(raw_weights.values())

col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    if st.button("Save & Normalize Weights", type="primary"):
        if total_raw <= 0:
            st.session_state.scoring_weights = dict(DEFAULT_WEIGHTS)
            st.warning("Total weight was 0; restored balanced default weights totaling 100%.")
        else:
            normalized = {
                k: round((v / total_raw) * 100, 1) for k, v in raw_weights.items()
            }
            diff = round(100.0 - sum(normalized.values()), 1)
            normalized["skills"] = round(normalized["skills"] + diff, 1)
            st.session_state.scoring_weights = normalized
            st.success("Scoring weights saved and normalized to 100%.")
with col_btn2:
    if st.button("Reset to Default Weights"):
        st.session_state.scoring_weights = dict(DEFAULT_WEIGHTS)
        st.info("Reset scoring weights to default values.")

st.divider()
st.subheader("Active Normalized Weights (%)")
st.json(st.session_state.scoring_weights)
