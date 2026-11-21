"""Page 7: Detailed Individual Candidate Reports & PDF Export."""

import streamlit as st
from app.components.cards import render_skill_pills
from reports.pdf_generator import generate_candidate_pdf

st.title("Individual Candidate Evaluation Report")

results = st.session_state.ranked_results
if not results:
    st.warning("No evaluated candidates available. Run matching first.")
    st.stop()

candidate_map = {r["candidate_name"]: r for r in results}
chosen_name = st.selectbox("Select Candidate", list(candidate_map.keys()))
cand = candidate_map[chosen_name]
job_title = st.session_state.parsed_jd["job_title"] if st.session_state.parsed_jd else "Target Role"

st.divider()
h1, h2 = st.columns([3, 1])
with h1:
    st.subheader(cand["candidate_name"])
    st.write(f"**Email:** {cand['email']}  |  **Phone:** {cand['phone']}")
    st.write(f"**Education:** {cand['education_level']}  |  **Estimated Experience:** {cand['estimated_experience_years']} years")
with h2:
    st.metric("Match Score", f"{cand['overall_score']} / 100")

c1, c2 = st.columns(2)
with c1:
    st.write("**Matched Skills**")
    render_skill_pills(cand["matching_skills"], "#DCFCE7", "#166534")
with c2:
    st.write("**Missing Skills**")
    render_skill_pills(cand["missing_skills"], "#FEE2E2", "#991B1B")

st.write("**Strengths**")
for item in cand["strengths"]:
    st.success(item)

st.write("**Areas for Improvement**")
for item in cand["areas_for_improvement"]:
    st.warning(item)

st.write("**Actionable Recommendations**")
for item in cand["recommendations"]:
    st.info(item)

pdf_bytes = generate_candidate_pdf(cand, job_title)
st.download_button(
    label="Download Formal PDF Report",
    data=pdf_bytes,
    file_name=f"{cand['candidate_name'].replace(' ', '_')}_HireSense_Report.pdf",
    mime="application/pdf",
    type="primary",
)
