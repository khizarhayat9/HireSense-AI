"""Page 9: Configurable Scoring Weights, Ethics & Architecture Info."""

import streamlit as st
from config import DEFAULT_SCORING_WEIGHTS

st.title("Scoring Configuration & System Methodology")

st.subheader("Customize Scoring Engine Weights")
st.write("Adjust how much weight each NLP signal contributes to the composite match score. Values are automatically normalized to 100%.")

w = st.session_state.scoring_weights
w_skills = st.slider("Required Skills Coverage Weight", 0.0, 1.0, float(w["skills_match"]), 0.05)
w_sem = st.slider("Semantic Embedding Similarity Weight (SBERT)", 0.0, 1.0, float(w["semantic_similarity"]), 0.05)
w_tfidf = st.slider("Lexical TF-IDF Similarity Weight", 0.0, 1.0, float(w["tfidf_similarity"]), 0.05)
w_exp = st.slider("Experience Alignment Weight", 0.0, 1.0, float(w["experience_match"]), 0.05)
w_edu = st.slider("Education Alignment Weight", 0.0, 1.0, float(w["education_match"]), 0.05)

col1, col2 = st.columns(2)
if col1.button("Save Scoring Weights", type="primary"):
    total = w_skills + w_sem + w_tfidf + w_exp + w_edu
    if total == 0:
        st.error("At least one weight must be greater than zero.")
    else:
        st.session_state.scoring_weights = {
            "skills_match": round(w_skills / total, 4),
            "semantic_similarity": round(w_sem / total, 4),
            "tfidf_similarity": round(w_tfidf / total, 4),
            "experience_match": round(w_exp / total, 4),
            "education_match": round(w_edu / total, 4),
        }
        st.success("Updated scoring weights! Re-run matching to apply changes.")

if col2.button("Reset to Default Weights"):
    st.session_state.scoring_weights = DEFAULT_SCORING_WEIGHTS.copy()
    st.rerun()

st.divider()
st.subheader("Responsible AI & Limitations")
st.markdown(
    """
    - **Decision Support Only:** HireSense AI surfaces textual alignment signals to assist human recruiters; it never automates hiring decisions.
    - **Bias & Privacy Protection:** Protected attributes (age, gender, race, nationality, religion) are never extracted or used in scoring.
    - **Known Limitations:** Creative resume formatting, heavy graphics, or non-standard skill acronyms outside the taxonomy may reduce recall, which is why hybrid semantic embeddings (`all-MiniLM-L6-v2`) accompany exact keyword matching.
    """
)
