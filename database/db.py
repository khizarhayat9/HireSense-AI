"""CRUD operations, history retrieval, and GDPR-style data deletion for SQLite."""

import json
from pathlib import Path
import sqlite3
from config import DB_PATH
from database.models import SCHEMA_SQL


class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def save_resume(self, filename: str, parsed_resume: dict) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO resumes (filename, candidate_name, email, phone, parsed_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    parsed_resume.get("candidate_name", "Unknown"),
                    parsed_resume.get("email", ""),
                    parsed_resume.get("phone", ""),
                    json.dumps(parsed_resume),
                ),
            )
            return int(cur.lastrowid)

    def save_job_description(self, raw_text: str, parsed_jd: dict) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO job_descriptions (job_title, raw_text, parsed_json)
                VALUES (?, ?, ?)
                """,
                (
                    parsed_jd.get("job_title", "Untitled Role"),
                    raw_text,
                    json.dumps(parsed_jd),
                ),
            )
            return int(cur.lastrowid)

    def save_scan_result(self, resume_id: int, job_id: int, evaluation: dict) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO scan_results (resume_id, job_id, overall_score, result_json)
                VALUES (?, ?, ?, ?)
                """,
                (resume_id, job_id, evaluation["overall_score"], json.dumps(evaluation)),
            )
            return int(cur.lastrowid)

    def get_all_resumes(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM resumes ORDER BY uploaded_at DESC").fetchall()
            return [{**dict(r), "parsed": json.loads(r["parsed_json"])} for r in rows]

    def get_all_jobs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM job_descriptions ORDER BY created_at DESC").fetchall()
            return [{**dict(r), "parsed": json.loads(r["parsed_json"])} for r in rows]

    def get_scan_history(self, search_query: str = "") -> list[dict]:
        sql = """
            SELECT s.id, s.overall_score, s.scanned_at, s.result_json,
                   r.id AS resume_id, r.candidate_name, r.filename,
                   j.id AS job_id, j.job_title
            FROM scan_results s
            JOIN resumes r ON s.resume_id = r.id
            JOIN job_descriptions j ON s.job_id = j.id
        """
        params: tuple = ()
        if search_query.strip():
            sql += " WHERE r.candidate_name LIKE ? OR j.job_title LIKE ?"
            like = f"%{search_query.strip()}%"
            params = (like, like)
        sql += " ORDER BY s.scanned_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [{**dict(r), "result": json.loads(r["result_json"])} for r in rows]

    def delete_resume(self, resume_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM resumes WHERE id = ?", (resume_id,))

    def purge_all_data(self) -> None:
        with self._connect() as conn:
            conn.executescript("DELETE FROM scan_results; DELETE FROM resumes; DELETE FROM job_descriptions;")
