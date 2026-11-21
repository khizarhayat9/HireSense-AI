"""Phase 10: Automated Unit & Integration Test Suite for HireSense AI."""

import io
from pathlib import Path
from docx import Document
from fpdf import FPDF

from config import SAMPLE_JOBS_DIR, SAMPLE_RESUMES_DIR
from database.db import DatabaseManager
from nlp.document_loader import extract_text_from_docx, extract_text_from_pdf, validate_file
from nlp.job_analyzer import analyze_job_description
from nlp.resume_parser import parse_resume
from nlp.scoring_engine import evaluate_candidate, rank_candidates
from nlp.semantic_matcher import compute_semantic_similarity
from reports.pdf_generator import generate_candidate_pdf


def test_file_validation():
    ok, _ = validate_file("resume.pdf", 500_000)
    assert ok is True
    bad_ext, msg = validate_file("resume.exe", 500)
    assert bad_ext is False and "Unsupported" in msg


def test_pdf_and_docx_extraction():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, "Jane Doe\njane@example.com\nSkills: Python, SQL, Machine Learning, Pandas, NumPy, Git, Docker, AWS")
    pdf_bytes = bytes(pdf.output())
    pdf_doc = extract_text_from_pdf(pdf_bytes, "jane.pdf")
    assert "Jane Doe" in pdf_doc.raw_text
    assert pdf_doc.is_scanned_warning is False

    doc = Document()
    doc.add_paragraph("John Smith")
    doc.add_paragraph("john@example.com | Python, PyTorch, SQL")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    docx_doc = extract_text_from_docx(buf, "john.docx")
    assert "John Smith" in docx_doc.raw_text


def test_end_to_end_ranking_order():
    raw_jd = (SAMPLE_JOBS_DIR / "junior_ml_engineer.txt").read_text(encoding="utf-8")
    parsed_jd = analyze_job_description(raw_jd)
    assert "Python" in parsed_jd["required_technical_skills"]

    strong = parse_resume((SAMPLE_RESUMES_DIR / "candidate_strong_ahmad.txt").read_text(encoding="utf-8"), "ahmad.txt")
    moderate = parse_resume((SAMPLE_RESUMES_DIR / "candidate_moderate_sara.txt").read_text(encoding="utf-8"), "sara.txt")
    weak = parse_resume((SAMPLE_RESUMES_DIR / "candidate_weak_bilal.txt").read_text(encoding="utf-8"), "bilal.txt")

    ranked = rank_candidates([
        evaluate_candidate(weak, parsed_jd),
        evaluate_candidate(strong, parsed_jd),
        evaluate_candidate(moderate, parsed_jd),
    ])

    assert ranked[0]["candidate_name"] == "Ahmad Raza"
    assert ranked[1]["candidate_name"] == "Sara Khan"
    assert ranked[2]["candidate_name"] == "Bilal Ahmed"
    assert ranked[0]["overall_score"] > ranked[1]["overall_score"] > ranked[2]["overall_score"]


def test_semantic_synonym_recognition():
    sim_related = compute_semantic_similarity(
        "Experienced in Statistical Modeling and Artificial Intelligence",
        "Looking for Machine Learning and Data Science engineer",
    )
    sim_unrelated = compute_semantic_similarity(
        "Pastry chef specializing in French croissants and sourdough",
        "Looking for Machine Learning and Data Science engineer",
    )
    assert sim_related > sim_unrelated


def test_database_and_pdf_report(tmp_path: Path):
    db = DatabaseManager(tmp_path / "test.db")
    raw_jd = (SAMPLE_JOBS_DIR / "junior_ml_engineer.txt").read_text(encoding="utf-8")
    parsed_jd = analyze_job_description(raw_jd)
    parsed_res = parse_resume((SAMPLE_RESUMES_DIR / "candidate_strong_ahmad.txt").read_text(encoding="utf-8"), "ahmad.txt")

    r_id = db.save_resume("ahmad.txt", parsed_res)
    j_id = db.save_job_description(raw_jd, parsed_jd)
    ev = evaluate_candidate(parsed_res, parsed_jd)
    db.save_scan_result(r_id, j_id, ev)

    history = db.get_scan_history("Ahmad")
    assert len(history) == 1

    pdf_bytes = generate_candidate_pdf(ev, parsed_jd["job_title"])
    assert pdf_bytes.startswith(b"%PDF")

    db.delete_resume(r_id)
    assert len(db.get_scan_history()) == 0
