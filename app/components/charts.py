"""Interactive Plotly visualization components for HireSense AI."""

from collections import Counter
from typing import Dict, List

import plotly.graph_objects as go


def render_talent_landscape(results: List[Dict]) -> go.Figure:
    """Bubble/scatter plot comparing candidates by experience, semantic score, and overall rank."""
    if not results:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.55)",
            height=380,
        )
        return fig

    x_exp = [r.get("experience_years", 0) for r in results]
    y_score = [r.get("final_score", r.get("overall_score", 0)) for r in results]
    labels = [
        f"#{r.get('rank', i + 1)} — {r.get('name', r.get('candidate_name', f'Candidate {i + 1}'))}"
        for i, r in enumerate(results)
    ]
    sizes = [max(10, min(32, len(r.get("matched_skills", [])) * 4 + 8)) for r in results]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_exp,
                y=y_score,
                mode="markers+text",
                text=[f"#{r.get('rank', i + 1)}" for i, r in enumerate(results[:25])]
                + [""] * max(0, len(results) - 25),
                textposition="top center",
                hovertext=labels,
                hoverinfo="text+x+y",
                marker=dict(
                    size=sizes,
                    color=y_score,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Match %"),
                    line=dict(width=1, color="#f8fafc"),
                ),
            )
        ]
    )
    fig.update_layout(
        title="Candidate Talent Landscape (Experience vs. Match Score)",
        xaxis_title="Estimated Experience (Years)",
        yaxis_title="Suitability Match Score (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        height=420,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def render_candidate_radar(candidate: Dict) -> go.Figure:
    """Radar/polar chart showing a candidate's breakdown across evaluation dimensions."""
    categories = [
        "Skills",
        "Semantic AI",
        "Experience",
        "Keywords",
        "Education",
    ]
    values = [
        float(candidate.get("skills_score", candidate.get("final_score", 0))),
        float(candidate.get("semantic_score", candidate.get("final_score", 0))),
        float(min(100, candidate.get("experience_years", 0) * 15)),
        float(candidate.get("tfidf_score", candidate.get("final_score", 0))),
        float(candidate.get("education_score", min(100, candidate.get("education_rank", 3) * 20))),
    ]

    cats_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    cand_name = candidate.get("name", candidate.get("candidate_name", "Candidate"))

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.28)",
            line=dict(color="#38bdf8", width=2.5),
            name=cand_name,
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


def plot_candidate_scores_bar(results: List[Dict]) -> go.Figure:
    """Bar chart of candidate overall match scores."""
    names = [
        r.get("candidate_name", r.get("name", f"Candidate {i + 1}"))
        for i, r in enumerate(results)
    ]
    scores = [
        r.get("overall_score", r.get("final_score", 0))
        for r in results
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=scores,
                text=[f"{s}%" for s in scores],
                textposition="outside",
                marker_color="#3B82F6",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Candidate",
        yaxis_title="Match Score (%)",
        yaxis=dict(range=[0, 110]),
        height=420,
        margin=dict(t=20, b=40, l=20, r=20),
    )
    return fig


def plot_skills_frequency(results: List[Dict], field_name: str, color: str) -> go.Figure:
    """Horizontal bar chart counting skill occurrences across candidate results."""
    counts: Dict[str, int] = {}
    for row in results:
        for skill in row.get(field_name, []) or []:
            counts[skill] = counts.get(skill, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    skills = [k for k, _ in reversed(sorted_items)]
    values = [v for _, v in reversed(sorted_items)]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=skills,
                orientation="h",
                text=values,
                textposition="outside",
                marker_color=color,
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Candidate Count",
        yaxis_title="Skill",
        height=320,
        margin=dict(t=20, b=30, l=20, r=20),
    )
    return fig
