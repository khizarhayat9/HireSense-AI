"""Zero-Trust Cybersecurity & Privacy Hardening Module for HireSense AI."""

import html
import io
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

# Binary magic headers for strict file verification
PDF_MAGIC = b"%PDF-"
DOCX_MAGIC = b"PK\x03\x04"

# Enterprise batch safety caps
MAX_ZIP_FILES = 5000
MAX_SINGLE_FILE_BYTES = 10 * 1024 * 1024       # 10 MB per resume
MAX_TOTAL_UNCOMPRESSED_BYTES = 500 * 1024 * 1024  # 500 MB ZIP bomb ceiling
MAX_COMPRESSION_RATIO = 100


def sanitize_text(value: str) -> str:
    """Escape HTML/script payloads and strip null bytes to prevent XSS."""
    if not value:
        return ""
    cleaned = value.replace("\x00", "")
    return html.escape(cleaned, quote=True)


def redact_pii(text: str, candidate_name: str = "") -> str:
    """Mask emails, phone numbers, URLs, and candidate names for blind hiring."""
    if not text:
        return ""
    masked = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL REDACTED]", text)
    masked = re.sub(r"(\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}", "[PHONE REDACTED]", masked)
    masked = re.sub(r"https?://\S+|www\.\S+", "[LINK REDACTED]", masked)
    if candidate_name and len(candidate_name.strip()) > 2:
        pattern = re.compile(re.escape(candidate_name.strip()), re.IGNORECASE)
        masked = pattern.sub("[CANDIDATE]", masked)
    return masked


def verify_magic_bytes(filename: str, data: bytes) -> Tuple[bool, str]:
    """Verify binary magic headers match the declared extension (.pdf or .docx)."""
    ext = Path(filename).suffix.lower()
    if len(data) < 8:
        return False, "File is empty or corrupted."
    if len(data) > MAX_SINGLE_FILE_BYTES:
        return False, f"Exceeds {MAX_SINGLE_FILE_BYTES // (1024 * 1024)} MB per-file limit."

    if ext == ".pdf":
        if not data.startswith(PDF_MAGIC):
            return False, "Security Block: File extension is .pdf but binary header is invalid."
        # Scan for embedded JavaScript launch actions inside PDF
        if b"/JavaScript" in data or b"/Launch" in data:
            return False, "Security Block: Embedded PDF JavaScript/Launch action detected."
        return True, "OK"

    if ext == ".docx":
        if not data.startswith(DOCX_MAGIC):
            return False, "Security Block: File extension is .docx but ZIP/OOXML header is invalid."
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                if "word/document.xml" not in names:
                    return False, "Security Block: Malformed DOCX structure."
                # Block macro-enabled payloads
                if any(n.endswith(".bin") or "vbaProject" in n for n in names):
                    return False, "Security Block: Embedded VBA macro detected."
        except zipfile.BadZipFile:
            return False, "Security Block: Corrupted OOXML archive."
        return True, "OK"

    return False, f"Unsupported file extension: {ext}"


def safe_extract_zip(zip_bytes: bytes) -> Tuple[List[Tuple[str, bytes]], List[str]]:
    """Safely extract PDF/DOCX resumes from a ZIP archive with ZIP-bomb & path-traversal checks."""
    valid_files: List[Tuple[str, bytes]] = []
    security_logs: List[str] = []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ZIP_FILES * 2:
                security_logs.append(f"Archive contains too many entries ({len(infos)}). Truncated safely.")

            total_uncompressed = 0
            for info in infos:
                if info.is_dir():
                    continue

                # Block directory traversal (../) and hidden macOS metadata
                norm_name = info.filename.replace("\\", "/")
                base_name = os.path.basename(norm_name)
                if ".." in norm_name.split("/") or norm_name.startswith("/") or base_name.startswith("._"):
                    security_logs.append(f"Blocked suspicious archive path: {sanitize_text(info.filename)}")
                    continue

                ext = Path(base_name).suffix.lower()
                if ext not in {".pdf", ".docx"}:
                    continue

                # Check compression ratio (ZIP bomb defense)
                if info.compress_size > 0 and (info.file_size / info.compress_size) > MAX_COMPRESSION_RATIO:
                    security_logs.append(f"Blocked high-ratio compression anomaly: {sanitize_text(base_name)}")
                    continue

                total_uncompressed += info.file_size
                if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                    security_logs.append("Reached 500 MB uncompressed safety ceiling; remaining files skipped.")
                    break

                if len(valid_files) >= MAX_ZIP_FILES:
                    security_logs.append(f"Reached {MAX_ZIP_FILES} resume batch cap.")
                    break

                raw = zf.read(info.filename)
                ok, reason = verify_magic_bytes(base_name, raw)
                if ok:
                    valid_files.append((base_name, raw))
                else:
                    security_logs.append(f"{sanitize_text(base_name)}: {reason}")

    except zipfile.BadZipFile:
        security_logs.append("Security Block: Uploaded ZIP file is corrupted or invalid.")

    return valid_files, security_logs
