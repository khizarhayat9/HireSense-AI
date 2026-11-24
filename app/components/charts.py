"""Interactive Plotly Data Visualizers for HireSense AI."""

from collections import Counter
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_talent_landscape(results: List[Dict]) -> go.Figure:
    """Interactive bubble scatter plot comparing Experience vs Semantic Alignment vs Final Score."""
    df = pd.DataFrame(results[:250])  # Render top 250 for smooth browser responsiveness
    df["Tier"] = df["final_score"].apply(
        lambda s: "Top Shortlist (80%+)" if s >= 80 else ("Strong Fit (65-79%)" if s >= 65 else "Review Pool (<65%)")
    )

    fig = px.scatter(
        df,
        x="semantic_score",
        y="experience_years",
        size="final_score",
        color="Tier",
        hover_name="name",
        hover_data={
            "rank": True,
            "final_score": True,
            "skills_score": True,
            "education_label": True,
            "Tier": False,
        },
        color_discrete_map={
            "Top Shortlist (80%+)": "#10b981",
            "Strong Fit (65-79%)": "#6366f1",
            "Review Pool (<65%)": "#f59e0b",
        },
        labels={
            "semantic_score": "Semantic AI Alignment (%)",
            "experience_years": "Experience (Years)",
            "final_score": "Overall Match Score",
        },
        title="Talent Landscape: Experience vs. Semantic Alignment (Top Candidates)",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_candidate_radar(candidate: Dict) -> go.Figure:
    """Multi-axis Explainable AI radar chart for a top-ranked candidate."""
    categories = ["Skills Match", "Semantic AI", "TF-IDF Keywords", "Experience", "Education"]
    values = [
        candidate["skills_score"],
        candidate["semantic_score"],
        candidate["tfidf_score"],
        candidate["experience_score"],
        candidate["education_score"],
    ]
    values_closed = values + [values[0]]
    cats_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.28)",
            line=dict(color="#38bdf8", width=2.5),
            name=candidate["name"],
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(15,23,42,0.6)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.2)"),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=340,
        margin=dict(l=35, r=35, t=30, b=30),
    )
    return fig


def render_skill_supply_chart(results: List[Dict]) -> go.Figure:
    """Horizontal bar chart showing top matched vs missing skills across the applicant pool."""
    matched_counter = Counter()
    missing_counter = Counter()
    for r in results:
        matched_counter.update(r.get("matched_skills", []))
        missing_counter.update(r.get("missing_skills", []))

    top_matched = matched_counter.most_common(10)
    if not top_matched:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300)
        return fig

    skills = [s for s, _ in reversed(top_matched)]
    counts = [c for _, c in reversed(top_matched)]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=skills,
            orientation="h",
            marker=dict(
                color=counts,
                colorscale=[[0, "#6366f1"], [1, "#06b6d4"]],
            ),
            text=counts,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Most Prevalent Verified Skills Across Candidate Pool",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        xaxis_title="Number of Candidates",
        height=380,
        margin=dict(l=20, r=30, t=50, b=20),
    )
    return fig
