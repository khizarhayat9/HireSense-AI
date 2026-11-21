"""Generate real .pdf and .docx sample files from the text fixtures in data/sample_resumes/."""

from docx import Document
from fpdf import FPDF
from config import SAMPLE_RESUMES_DIR


def build_sample_binaries() -> None:
    for txt_path in SAMPLE_RESUMES_DIR.glob("*.txt"):
        content = txt_path.read_text(encoding="utf-8")
        stem = txt_path.stem

        docx_path = SAMPLE_RESUMES_DIR / f"{stem}.docx"
        doc = Document()
        for line in content.splitlines():
            doc.add_paragraph(line)
        doc.save(docx_path)

        pdf_path = SAMPLE_RESUMES_DIR / f"{stem}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        safe = content.replace("—", "-").replace("–", "-").encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe)
        pdf.output(str(pdf_path))

    print("Sample .pdf and .docx files created in data/sample_resumes/.")


if __name__ == "__main__":
    build_sample_binaries()
