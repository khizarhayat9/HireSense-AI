"""Phase 2: Document Extraction Engine for PDF and DOCX files."""

from dataclasses import dataclass
import io
from pathlib import Path
from typing import BinaryIO, Union
import fitz
from docx import Document

from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, MIN_EXTRACTABLE_CHARS


@dataclass
class ExtractedDocument:
    filename: str
    file_type: str
    raw_text: str
    page_count: int
    char_count: int
    is_scanned_warning: bool
    error: str | None = None


def validate_file(filename: str, file_size_bytes: int) -> tuple[bool, str]:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}."
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        return False, f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB} MB."
    if file_size_bytes == 0:
        return False, "Uploaded file is empty."
    return True, ""


def extract_text_from_pdf(file_bytes: bytes, filename: str = "document.pdf") -> ExtractedDocument:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = [page.get_text("text", sort=True) for page in doc]
        page_count = len(doc)
        doc.close()

        full_text = "\n".join(pages_text).strip()
        char_count = len(full_text)
        is_scanned = char_count < MIN_EXTRACTABLE_CHARS

        return ExtractedDocument(
            filename=filename,
            file_type="pdf",
            raw_text=full_text,
            page_count=page_count,
            char_count=char_count,
            is_scanned_warning=is_scanned,
            error="Low extractable text detected. This PDF may be scanned/image-only." if is_scanned else None,
        )
    except Exception as exc:
        return ExtractedDocument(filename, "pdf", "", 0, 0, False, f"PDF parsing failed: {exc}")


def extract_text_from_docx(file_stream: Union[BinaryIO, Path, str], filename: str = "document.docx") -> ExtractedDocument:
    try:
        doc = Document(file_stream)
        chunks: list[str] = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        for table in doc.tables:
            for row in table.rows:
                row_cells = [c.text.strip().replace("\n", " ") for c in row.cells if c.text.strip()]
                if row_cells:
                    chunks.append(" | ".join(dict.fromkeys(row_cells)))

        full_text = "\n".join(chunks).strip()
        char_count = len(full_text)
        estimated_pages = max(1, round(char_count / 3000))

        return ExtractedDocument(
            filename=filename,
            file_type="docx",
            raw_text=full_text,
            page_count=estimated_pages,
            char_count=char_count,
            is_scanned_warning=False,
            error="No readable text found in DOCX." if char_count == 0 else None,
        )
    except Exception as exc:
        return ExtractedDocument(filename, "docx", "", 0, 0, False, f"DOCX parsing failed: {exc}")


def load_document(file_path_or_bytes: Union[Path, str, bytes], filename: str | None = None) -> ExtractedDocument:
    if isinstance(file_path_or_bytes, (str, Path)):
        path = Path(file_path_or_bytes)
        fname = filename or path.name
        ext = path.suffix.lower()
        data = path.read_bytes()
    else:
        fname = filename or "uploaded_file"
        ext = Path(fname).suffix.lower()
        data = file_path_or_bytes

    if ext == ".txt":
        text = data.decode("utf-8", errors="ignore").strip()
        return ExtractedDocument(fname, "txt", text, 1, len(text), False, None)

    is_valid, err = validate_file(fname, len(data))
    if not is_valid:
        return ExtractedDocument(fname, ext.lstrip("."), "", 0, 0, False, err)

    if ext == ".pdf":
        return extract_text_from_pdf(data, fname)
    if ext == ".docx":
        return extract_text_from_docx(io.BytesIO(data), fname)

    return ExtractedDocument(fname, ext, "", 0, 0, False, "Unsupported format.")
