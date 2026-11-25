"""Page 8: Scan History & Candidate Data Management."""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Scan History & Candidate Data Management", layout="wide")

st.session_state.setdefault("scan_history", [])
st.session_state.setdefault("ranked_results", [])
st.session_state.setdefault("ranking_results", [])

st.title("Scan History & Candidate Data Management")

search_term = st.text_input("Search by candidate name or job title", value="").strip()

db = st.session_state.get("db")
history = []

if db is not None and hasattr(db, "get_scan_history"):
    try:
        raw_history = db.get_scan_history(search_term)
        if isinstance(raw_history, list):
            history = raw_history
        elif isinstance(raw_history, pd.DataFrame):
            history = raw_history.to_dict(orient="records")
    except Exception as exc:
        st.warning(f"Could not load database scan history ({exc}). Showing session data instead.")

if not history:
    session_history = st.session_state.get("scan_history") or []
    ranked = (
        st.session_state.get("ranked_results")
        or st.session_state.get("ranking_results")
        or []
    )
    if session_history:
        history = session_history
    elif ranked:
        history = [
            {
                "Rank": r.get("rank", i + 1),
                "Candidate": r.get("candidate_name") or r.get("name") or f"Candidate {i + 1}",
                "Score": r.get("overall_score", r.get("final_score", 0)),
                "Education": r.get("education_level", "N/A"),
                "Matching Skills": ", ".join(r.get("matching_skills") or []),
            }
            for i, r in enumerate(ranked)
        ]

if search_term and history:
    q = search_term.lower()
    history = [
        item for item in history
        if q in str(item).lower()
    ]

if not history:
    st.info(
        "No scan history records found yet. Upload resumes, run candidate matching, "
        "or connect the session database to view saved scan history."
    )
else:
    df_history = pd.DataFrame(history)
    st.dataframe(df_history, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Scan History as CSV",
        data=df_history.to_csv(index=False).encode("utf-8"),
        file_name="hiresense_scan_history.csv",
        mime="text/csv",
    )
