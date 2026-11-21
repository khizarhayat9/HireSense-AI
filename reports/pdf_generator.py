"""Phase 8: Individual Candidate PDF Report Generator using FPDF2."""

from fpdf import FPDF


def _safe_text(text: str) -> str:
    return (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .encode("latin-1", "replace")
        .decode("latin-1")
    )


def generate_candidate_pdf(evaluation: dict, job_title: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_fill_color(30, 58, 138)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(12, 8)
    pdf.cell(0, 8, "HireSense AI - Candidate Evaluation Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(12)
    pdf.cell(0, 6, _safe_text(f"Target Role: {job_title}"), ln=True)

    pdf.set_text_color(15, 23, 42)
    pdf.set_xy(12, 38)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _safe_text(f"Candidate: {evaluation['candidate_name']}"), ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _safe_text(f"Email: {evaluation['email']}  |  Phone: {evaluation['phone']}"), ln=True)
    pdf.cell(
        0,
        6,
        _safe_text(f"Education: {evaluation['education_level']}  |  Estimated Experience: {evaluation['estimated_experience_years']} yrs"),
        ln=True,
    )
    pdf.ln(3)

    pdf.set_fill_color(241, 245, 249)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 9, f"Overall Alignment Score: {evaluation['overall_score']} / 100", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    sub = evaluation["sub_scores"]
    pdf.cell(
        0,
        7,
        f"Skills: {sub['skills_match']}% | Semantic: {sub['semantic_similarity']}% | TF-IDF: {sub['tfidf_similarity']}% | Exp: {sub['experience_match']}% | Edu: {sub['education_match']}%",
        ln=True,
    )
    pdf.ln(4)

    def _add_section(title: str, items: list[str], fallback: str = "None identified."):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 7, _safe_text(title), ln=True)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 10)
        if not items:
            pdf.multi_cell(0, 6, _safe_text(f"- {fallback}"))
        else:
            for item in items:
                pdf.multi_cell(0, 6, _safe_text(f"- {item}"))
        pdf.ln(2)

    _add_section("Matching Required Skills", [", ".join(evaluation["matching_skills"])] if evaluation["matching_skills"] else [])
    _add_section("Missing Required Skills", [", ".join(evaluation["missing_skills"])] if evaluation["missing_skills"] else [])
    _add_section("Key Strengths", evaluation.get("strengths", []))
    _add_section("Areas for Improvement", evaluation.get("areas_for_improvement", []))
    _add_section("Interview & Onboarding Recommendations", evaluation.get("recommendations", []))

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(
        0,
        5,
        "Disclaimer: HireSense AI provides decision-support alignment signals based on document text. "
        "Scores do not measure human potential or guarantee hiring suitability.",
    )

    return bytes(pdf.output())
