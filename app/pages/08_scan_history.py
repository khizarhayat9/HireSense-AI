"""Page 8: SQLite Scan History, Search & Privacy Data Deletion."""

import pandas as pd
import streamlit as st

st.title("Scan History & Candidate Data Management")

search_term = st.text_input("Search by candidate name or job title")
history = st.session_state.db.get_scan_history(search_term)

if history:
    df = pd.DataFrame([
        {
            "Scan ID": h["id"],
            "Candidate": h["candidate_name"],
            "Target Role": h["job_title"],
            "Score": h["overall_score"],
            "Timestamp": h["scanned_at"],
        }
        for h in history
    ])
    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.info("No historical scans found in SQLite database.")

st.divider()
st.subheader("Privacy & Record Management")
st.caption("Delete individual candidate profiles or purge all stored records from SQLite.")

stored_resumes = st.session_state.db.get_all_resumes()
if stored_resumes:
    options = {f"#{r['id']} - {r['candidate_name']} ({r['filename']})": r["id"] for r in stored_resumes}
    selected_label = st.selectbox("Select candidate record to permanently delete", list(options.keys()))
    if st.button("Delete Selected Candidate Record"):
        st.session_state.db.delete_resume(options[selected_label])
        st.success("Candidate record permanently deleted.")
        st.rerun()

if st.button("Purge Entire Database", type="secondary"):
    st.session_state.db.purge_all_data()
    st.session_state.parsed_resumes = []
    st.session_state.ranked_results = []
    st.success("All candidate and scan records deleted.")
    st.rerun()
