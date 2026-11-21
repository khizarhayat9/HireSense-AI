"""Styled HTML metric and candidate summary cards."""

import streamlit as st


def render_kpi_card(title: str, value: str, subtitle: str = "", accent: str = "#4F46E5") -> None:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid #E2E8F0;
            border-left: 4px solid {accent};
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        ">
            <div style="font-size: 0.82rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em;">
                {title}
            </div>
            <div style="font-size: 1.85rem; font-weight: 700; color: #0F172A; margin: 4px 0;">
                {value}
            </div>
            <div style="font-size: 0.80rem; color: #475569;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skill_pills(skills: list[str], bg_color: str = "#EEF2FF", text_color: str = "#3730A3") -> None:
    if not skills:
        st.caption("None")
        return
    pills_html = " ".join(
        f'<span style="display:inline-block; background:{bg_color}; color:{text_color}; '
        f'padding:3px 10px; border-radius:999px; font-size:0.80rem; font-weight:600; margin:2px 4px 2px 0;">{s}</span>'
        for s in skills
    )
    st.markdown(pills_html, unsafe_allow_html=True)
