"""Page 2: Recruitment Analytics Dashboard."""

import streamlit as st

from app.components.charts import (
    plot_candidate_scores_bar,
    plot_skills_frequency,
    render_candidate_radar,
    render_skill_supply_chart,
    render_talent_landscape,
)

st.set_page_config(page_title="Recruitment Analytics Dashboard", layout="wide")

st.session_state.setdefault("ranked_results", [])
st.session_state.setdefault("ranking_results", [])

results = st.session_state.get("ranked_results") or st.session_state.get("ranking_results") or []

st.title("Recruitment Analytics Dashboard")

if not results:
    st.info(
        "No ranked candidate results available yet. Upload resumes, analyze a job description, "
        "and run candidate matching first."
    )
    st.stop()

scores = [
    float(r.get("overall_score", r.get("final_score", 0)))
    for r in results
]
avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
top_score = round(max(scores), 1) if scores else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("Total Ranked Candidates", f"{len(results):,}")
m2.metric("Average Match Score", f"{avg_score}%")
m3.metric("Top Match Score", f"{top_score}%")

st.divider()

st.subheader("Candidate Match Scores")
st.plotly_chart(plot_candidate_scores_bar(results), use_container_width=True)

st.subheader("Talent Landscape (Experience vs. Match Score)")
st.plotly_chart(render_talent_landscape(results), use_container_width=True)

c_left, c_right = st.columns(2)
with c_left:
    st.subheader("Top Matching Skills")
    st.plotly_chart(
        plot_skills_frequency(results, "matching_skills", "#10B981"),
        use_container_width=True,
    )
with c_right:
    st.subheader("Top Missing Skills")
    st.plotly_chart(
        plot_skills_frequency(results, "missing_skills", "#EF4444"),
        use_container_width=True,
    )

st.divider()
st.subheader("Individual Candidate Breakdown")
candidate_labels = [
    f"#{r.get('rank', idx + 1)} — {r.get('candidate_name', r.get('name', f'Candidate {idx + 1}'))}"
    for idx, r in enumerate(results)
]
selected_label = st.selectbox("Select a candidate", candidate_labels)
selected_idx = candidate_labels.index(selected_label)

col_radar, col_supply = st.columns(2)
with col_radar:
    st.plotly_chart(render_candidate_radar(results[selected_idx]), use_container_width=True)
with col_supply:
    st.plotly_chart(render_skill_supply_chart(results), use_container_width=True)
