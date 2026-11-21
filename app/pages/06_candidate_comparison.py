"""Page 6: Side-by-Side Candidate Comparison & CSV/Excel Export."""

import io
import pandas as pd
import streamlit as st
from app.components.charts import plot_candidate_radar

st.title("Candidate Comparison Matrix")

results = st.session_state.ranked_results
if not results:
    st.warning("Run candidate matching first to populate comparison data.")
    st.stop()

names = [r["candidate_name"] for r in results]
selected_names = st.multiselect("Select candidates to compare on Radar Chart", names, default=names[:3])
selected_objs = [r for r in results if r["candidate_name"] in selected_names]

if selected_objs:
    st.plotly_chart(plot_candidate_radar(selected_objs), width="stretch")

rows = []
for r in results:
    rows.append({
        "Rank": r["rank"],
        "Candidate": r["candidate_name"],
        "Overall Score": r["overall_score"],
        "Skills (%)": r["sub_scores"]["skills_match"],
        "Semantic (%)": r["sub_scores"]["semantic_similarity"],
        "TF-IDF (%)": r["sub_scores"]["tfidf_similarity"],
        "Exp (Yrs)": r["estimated_experience_years"],
        "Education": r["education_level"],
        "Matching Skills": ", ".join(r["matching_skills"]),
        "Missing Skills": ", ".join(r["missing_skills"]),
    })

df = pd.DataFrame(rows)
st.dataframe(df, width="stretch", hide_index=True)

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Download Comparison as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="hiresense_candidate_comparison.csv",
        mime="text/csv",
    )
with col2:
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Candidate Ranking")
    st.download_button(
        "Download Comparison as Excel (.xlsx)",
        data=excel_buf.getvalue(),
        file_name="hiresense_candidate_comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
