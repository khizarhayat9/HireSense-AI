"""Page 2: Interactive Recruitment Analytics Dashboard."""

import streamlit as st
from app.components.cards import render_kpi_card
from app.components.charts import plot_candidate_scores_bar, plot_skills_frequency

st.title("Recruitment Analytics Dashboard")

results = st.session_state.ranked_results
resumes = st.session_state.parsed_resumes

if not results:
    st.warning("No candidate evaluations available yet. Load sample data on the **Welcome** page or run a match in **Candidate Matching**.")
    st.stop()

avg_score = round(sum(r["overall_score"] for r in results) / len(results), 1)
top_candidate = results[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi_card("Resumes Uploaded", str(len(resumes)), "Active candidate pool", "#3B82F6")
with c2:
    render_kpi_card("Candidates Analyzed", str(len(results)), st.session_state.parsed_jd["job_title"], "#6366F1")
with c3:
    render_kpi_card("Average Match Score", f"{avg_score}%", "Weighted alignment index", "#8B5CF6")
with c4:
    render_kpi_card("Top Ranked", top_candidate["candidate_name"], f"Score: {top_candidate['overall_score']}/100", "#10B981")

st.divider()

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Candidate Score Distribution")
    st.plotly_chart(plot_candidate_scores_bar(results), width="stretch")
with right:
    st.subheader("Most Common Matching Skills")
    st.plotly_chart(plot_skills_frequency(results, "matching_skills", "#4F46E5"), width="stretch")

st.subheader("Most Frequent Missing Skills Across Pool")
st.plotly_chart(plot_skills_frequency(results, "missing_skills", "#EF4444"), width="stretch")
