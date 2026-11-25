"""Page 6: Side-by-Side Candidate Comparison & CSV/Excel Export."""

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Candidate Comparison Matrix", layout="wide")

st.session_state.setdefault("ranked_results", [])
st.session_state.setdefault("ranking_results", [])

st.title("Candidate Comparison Matrix")

results = (
    st.session_state.get("ranked_results")
    or st.session_state.get("ranking_results")
    or []
)

if not results:
    st.warning("Run candidate matching first to populate comparison data.")
    st.stop()


def _get_candidate_name(cand: dict, idx: int) -> str:
    return cand.get("candidate_name") or cand.get("name") or f"Candidate {idx + 1}"


def _build_comparison_radar(candidates: list[dict]) -> go.Figure:
    """Build a multi-candidate polar/radar comparison chart."""
    categories = [
        "Overall Match",
        "Skills Match",
        "Semantic AI",
        "Keyword TF-IDF",
        "Experience",
    ]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    palette = ["#38bdf8", "#fbbf24", "#34d399", "#f472b6", "#a78bfa", "#fb7185"]

    for idx, cand in enumerate(candidates):
        sub = cand.get("sub_scores") or {}
        overall = float(cand.get("overall_score", cand.get("final_score", 0)))
        skills_val = float(sub.get("skills_match", cand.get("skills_score", overall)))
        semantic_val = float(
            sub.get("semantic_similarity", cand.get("semantic_score", overall))
        )
        tfidf_val = float(sub.get("tfidf_similarity", cand.get("tfidf_score", overall)))
        exp_years = float(
            cand.get("estimated_experience_years", cand.get("experience_years", 0))
        )
        exp_val = min(100.0, exp_years * 15.0)

        values = [overall, skills_val, semantic_val, tfidf_val, exp_val]
        values_closed = values + [values[0]]
        label = _get_candidate_name(cand, idx)

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                opacity=0.22,
                line=dict(color=palette[idx % len(palette)], width=2.5),
                name=label,
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(15,23,42,0.6)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(148,163,184,0.2)",
            ),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=420,
        margin=dict(l=40, r=40, t=30, b=50),
    )
    return fig


names = [_get_candidate_name(r, i) for i, r in enumerate(results)]
selected_names = st.multiselect(
    "Select candidates to compare on Radar Chart",
    names,
    default=names[: min(3, len(names))],
)
selected_objs = [
    r for i, r in enumerate(results) if _get_candidate_name(r, i) in selected_names
]

if selected_objs:
    st.plotly_chart(_build_comparison_radar(selected_objs), use_container_width=True)

rows = []
for idx, r in enumerate(results):
    sub = r.get("sub_scores") or {}
    overall = r.get("overall_score", r.get("final_score", 0))
    rows.append(
        {
            "Rank": r.get("rank", idx + 1),
            "Candidate": _get_candidate_name(r, idx),
            "Overall Score": overall,
            "Skills (%)": sub.get("skills_match", r.get("skills_score", overall)),
            "Semantic (%)": sub.get(
                "semantic_similarity", r.get("semantic_score", overall)
            ),
            "TF-IDF (%)": sub.get("tfidf_similarity", r.get("tfidf_score", overall)),
            "Exp (Yrs)": r.get(
                "estimated_experience_years", r.get("experience_years", 0)
            ),
            "Education": r.get("education_level", "N/A"),
            "Matching Skills": ", ".join(r.get("matching_skills") or []),
            "Missing Skills": ", ".join(r.get("missing_skills") or []),
        }
    )

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

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
    try:
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Candidate Ranking")
        st.download_button(
            "Download Comparison as Excel (.xlsx)",
            data=excel_buf.getvalue(),
            file_name="hiresense_candidate_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.info("Excel export requires `openpyxl`; CSV download above works immediately.")
