"""Page 5: Intelligent Matching Engine & Explainable Candidate Ranking."""

import streamlit as st
from app.components.cards import render_skill_pills
from nlp.scoring_engine import evaluate_candidate, rank_candidates

st.title("Intelligent Candidate Matching & Ranking")
st.caption("Rankings are quantitative decision-support signals combining lexical, semantic, and skill-coverage metrics—not automated hiring decisions.")

if not st.session_state.parsed_resumes or not st.session_state.parsed_jd:
    st.warning("Please ensure both **Resumes** and a **Job Description** are loaded before running matching.")
    st.stop()

if st.button("Run AI Matching & Ranking Engine", type="primary"):
    with st.spinner("Computing TF-IDF vectors, Sentence-Transformer embeddings, and explainable scores..."):
        evals = [
            evaluate_candidate(res, st.session_state.parsed_jd, st.session_state.scoring_weights)
            for res in st.session_state.parsed_resumes
        ]
        st.session_state.ranked_results = rank_candidates(evals)
        st.success("Matching complete!")

results = st.session_state.ranked_results
if results:
    st.divider()
    all_req_skills = st.session_state.parsed_jd.get("required_technical_skills", [])
    selected_filter_skills = st.multiselect("Filter candidates by required skill:", all_req_skills)
    min_score_filter = st.slider("Minimum Overall Score", 0, 100, 0)

    filtered = [
        r for r in results
        if r["overall_score"] >= min_score_filter
        and all(s in r["matching_skills"] for s in selected_filter_skills)
    ]

    for cand in filtered:
        with st.container(border=True):
            top_l, top_r = st.columns([3, 1])
            with top_l:
                st.markdown(f"### #{cand['rank']} — {cand['candidate_name']}")
                st.caption(f"Education: {cand['education_level']}  |  Est. Experience: {cand['estimated_experience_years']} yrs  |  {cand['email']}")
            with top_r:
                st.metric("Overall Match", f"{cand['overall_score']} / 100")

            sub = cand["sub_scores"]
            st.caption(
                f"**Breakdown:** Skills `{sub['skills_match']}%` • Semantic `{sub['semantic_similarity']}%` • "
                f"TF-IDF `{sub['tfidf_similarity']}%` • Experience `{sub['experience_match']}%` • Education `{sub['education_match']}%`"
            )

            c_left, c_right = st.columns(2)
            with c_left:
                st.write("**Matching Skills:**")
                render_skill_pills(cand["matching_skills"], "#DCFCE7", "#166534")
            with c_right:
                st.write("**Missing Required Skills:**")
                render_skill_pills(cand["missing_skills"], "#FEE2E2", "#991B1B")

            with st.expander("Explainable AI (XAI) Reasoning"):
                st.write("**Strengths:**")
                for s in cand["strengths"]:
                    st.write(f"- {s}")
                st.write("**Areas for Improvement:**")
                for g in cand["areas_for_improvement"]:
                    st.write(f"- {g}")
