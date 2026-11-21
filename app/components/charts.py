"""Interactive Plotly charts for dashboard analytics and candidate comparison."""

from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_candidate_scores_bar(ranked_results: list[dict]) -> go.Figure:
    df = pd.DataFrame([
        {"Candidate": r["candidate_name"], "Overall Score": r["overall_score"]}
        for r in ranked_results
    ])
    fig = px.bar(
        df,
        x="Overall Score",
        y="Candidate",
        orientation="h",
        text="Overall Score",
        color="Overall Score",
        color_continuous_scale=["#94A3B8", "#3B82F6", "#6366F1"],
        range_x=[0, 105],
    )
    fig.update_layout(
        height=max(260, len(df) * 55),
        margin=dict(l=10, r=20, t=20, b=10),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    return fig


def plot_skills_frequency(ranked_results: list[dict], field: str = "matching_skills", color: str = "#4F46E5") -> go.Figure:
    counter: Counter = Counter()
    for r in ranked_results:
        counter.update(r.get(field, []))
    if not counter:
        return go.Figure()

    top_items = counter.most_common(10)
    df = pd.DataFrame(top_items, columns=["Skill", "Count"])
    fig = px.bar(df, x="Skill", y="Count", text="Count", color_discrete_sequence=[color])
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(textposition="outside")
    return fig


def plot_candidate_radar(selected_results: list[dict]) -> go.Figure:
    categories = ["Skills Match", "Semantic Similarity", "TF-IDF Lexical", "Experience", "Education"]
    fig = go.Figure()
    for res in selected_results:
        sub = res["sub_scores"]
        values = [
            sub["skills_match"],
            sub["semantic_similarity"],
            sub["tfidf_similarity"],
            sub["experience_match"],
            sub["education_match"],
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=res["candidate_name"],
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=400,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    return fig
