"""High-Volume Enterprise Batch Engine (1,000 - 5,000+ Resumes) with Explainable Ranking."""

from concurrent.futures import ThreadPoolExecutor
import io
import re
from typing import Callable, Dict, List, Optional, Tuple

import docx
import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.security import sanitize_text

EDUCATION_LEVELS = {
    "phd": (5, "PhD / Doctorate"),
    "doctorate": (5, "PhD / Doctorate"),
    "master": (4, "Master's Degree"),
    "m.sc": (4, "Master's Degree"),
    "msc": (4, "Master's Degree"),
    "m.s.": (4, "Master's Degree"),
    "mba": (4, "Master's / MBA"),
    "bachelor": (3, "Bachelor's Degree"),
    "b.sc": (3, "Bachelor's Degree"),
    "bsc": (3, "Bachelor's Degree"),
    "b.tech": (3, "Bachelor's Degree"),
    "b.e.": (3, "Bachelor's Degree"),
    "associate": (2, "Associate Degree"),
    "diploma": (2, "Diploma / Certification"),
}


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    """Fast in-memory text extraction for PDF and DOCX files."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        text_parts = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts).strip()

    if lower.endswith(".docx"):
        doc = docx.Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells if cell.text.strip()))
        return "\n".join(parts).strip()

    return ""


def parse_experience_years(text: str) -> float:
    """Extract estimated years of professional experience from resume text."""
    matches = re.findall(r"(\d{1,2}(?:\.\d)?)\+?\s*(?:years|yrs)\s*(?:of)?\s*(?:experience|exp)?", text.lower())
    if matches:
        vals = [float(m) for m in matches if 0 < float(m) <= 40]
        if vals:
            return max(vals)

    # Fallback: detect year ranges like 2021 - 2026
    ranges = re.findall(r"(20\d{2})\s*[-–to]+\s*(20\d{2}|present|current)", text.lower())
    total = 0.0
    for start, end in ranges:
        s = int(start)
        e = 2026 if end in {"present", "current"} else int(end)
        if 0 <= (e - s) <= 25:
            total += (e - s)
    return min(round(total, 1), 35.0)


def parse_education(text: str) -> Tuple[int, str]:
    """Detect highest degree level from resume text."""
    lower = text.lower()
    best_rank = 1
    best_label = "Not Specified"
    for kw, (rank, label) in EDUCATION_LEVELS.items():
        if re.search(rf"\b{re.escape(kw)}\b", lower) and rank > best_rank:
            best_rank = rank
            best_label = label
    return best_rank, best_label


def extract_candidate_name(filename: str, text: str) -> str:
    """Extract clean candidate name from top lines or fallback to sanitized filename."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:4]:
        if 2 <= len(line.split()) <= 4 and not any(ch.isdigit() for ch in line) and "@" not in line:
            if not any(w in line.lower() for w in ["resume", "curriculum", "vitae", "summary", "profile"]):
                return sanitize_text(line.title())
    stem = re.sub(r"[_\-]+", " ", filename.rsplit(".", 1)[0]).strip()
    return sanitize_text(stem.title())


def parse_single_resume(
    item: Tuple[str, bytes],
    skill_vocab: List[str],
) -> Optional[Dict]:
    """Worker function to parse one resume into a structured dictionary."""
    filename, data = item
    try:
        raw_text = extract_text_from_bytes(filename, data)
        if len(raw_text) < 80:
            return None

        lower_text = raw_text.lower()
        matched_skills = sorted({
            skill for skill in skill_vocab
            if re.search(rf"(?<![a-z0-9]){re.escape(skill.lower())}(?![a-z0-9])", lower_text)
        })
        exp_years = parse_experience_years(raw_text)
        edu_rank, edu_label = parse_education(raw_text)
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw_text)

        return {
            "filename": sanitize_text(filename),
            "name": extract_candidate_name(filename, raw_text),
            "email": sanitize_text(email_match.group(0)) if email_match else "Not Provided",
            "experience_years": exp_years,
            "education_rank": edu_rank,
            "education_label": edu_label,
            "skills": matched_skills,
            "text": raw_text[:8000],  # Cap per-doc tokens for fast vectorized encoding
        }
    except Exception:
        return None


def batch_parse_and_rank(
    files: List[Tuple[str, bytes]],
    jd_text: str,
    required_skills: List[str],
    min_exp_years: float,
    req_edu_rank: int,
    skill_vocab: List[str],
    weights: Dict[str, float],
    embedder=None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[Dict]:
    """
    Parse up to 5,000+ resumes in parallel and compute vectorized TF-IDF + Semantic rankings
    with explainable 'Why Suitable' insights.
    """
    total_files = len(files)
    if total_files == 0:
        return []

    if progress_callback:
        progress_callback(0.10, f"Parsing {total_files:,} resumes across multi-core workers...")

    parsed_candidates: List[Dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(parse_single_resume, item, skill_vocab) for item in files]
        for idx, fut in enumerate(futures, 1):
            res = fut.result()
            if res:
                parsed_candidates.append(res)
            if progress_callback and (idx % max(1, total_files // 20) == 0 or idx == total_files):
                pct = 0.10 + 0.45 * (idx / total_files)
                progress_callback(pct, f"Parsed {idx:,} / {total_files:,} resumes ({len(parsed_candidates):,} valid)...")

    if not parsed_candidates:
        return []

    if progress_callback:
        progress_callback(0.62, "Running vectorized TF-IDF lexical matrix scoring...")

    corpus = [jd_text] + [c["text"] for c in parsed_candidates]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    tfidf_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    if progress_callback:
        progress_callback(0.75, "Computing dense neural semantic embeddings...")

    if embedder is not None:
        jd_vec = embedder.encode([jd_text], normalize_embeddings=True)
        cand_vecs = embedder.encode(
            [c["text"] for c in parsed_candidates],
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        sem_sims = np.dot(cand_vecs, jd_vec.T).flatten()
        sem_sims = np.clip((sem_sims + 1.0) / 2.0, 0.0, 1.0)
    else:
        sem_sims = tfidf_sims.copy()

    if progress_callback:
        progress_callback(0.90, "Generating Explainable AI (XAI) rankings & suitability insights...")

    req_set = {s.lower() for s in required_skills}
    results: List[Dict] = []

    for i, cand in enumerate(parsed_candidates):
        cand_skills_lower = {s.lower() for s in cand["skills"]}
        matched_req = [s for s in required_skills if s.lower() in cand_skills_lower]
        missing_req = [s for s in required_skills if s.lower() not in cand_skills_lower]

        skill_score = (len(matched_req) / len(req_set) * 100.0) if req_set else min(len(cand["skills"]) * 12.0, 100.0)
        sem_score = float(sem_sims[i]) * 100.0
        tfidf_score = float(tfidf_sims[i]) * 100.0

        if min_exp_years <= 0:
            exp_score = 100.0
        else:
            exp_score = min((cand["experience_years"] / min_exp_years) * 100.0, 100.0)

        edu_score = 100.0 if cand["education_rank"] >= req_edu_rank else (cand["education_rank"] / max(req_edu_rank, 1)) * 100.0

        final_score = (
            weights.get("skills_match", 0.40) * skill_score
            + weights.get("semantic_similarity", 0.25) * sem_score
            + weights.get("tfidf_similarity", 0.10) * tfidf_score
            + weights.get("experience_match", 0.15) * exp_score
            + weights.get("education_match", 0.10) * edu_score
        )

        # Build clear human-readable "Why Suitable" explanation
        reasons = []
        if matched_req:
            reasons.append(f"Matches {len(matched_req)}/{len(req_set)} core skills ({', '.join(matched_req[:5])})")
        if cand["experience_years"] >= min_exp_years:
            reasons.append(f"Exceeds experience bar ({cand['experience_years']:.1f} yrs vs {min_exp_years:.1f} yrs required)")
        else:
            reasons.append(f"Has {cand['experience_years']:.1f} yrs relevant experience")
        if sem_score >= 70:
            reasons.append(f"Strong contextual domain alignment ({sem_score:.0f}% semantic similarity)")
        if cand["education_rank"] >= req_edu_rank and cand["education_label"] != "Not Specified":
            reasons.append(f"Holds {cand['education_label']}")

        why_suitable = " • ".join(reasons) if reasons else "General profile alignment with job requirements."

        results.append({
            "name": cand["name"],
            "filename": cand["filename"],
            "email": cand["email"],
            "final_score": round(final_score, 1),
            "skills_score": round(skill_score, 1),
            "semantic_score": round(sem_score, 1),
            "tfidf_score": round(tfidf_score, 1),
            "experience_score": round(exp_score, 1),
            "education_score": round(edu_score, 1),
            "experience_years": cand["experience_years"],
            "education_label": cand["education_label"],
            "matched_skills": matched_req,
            "missing_skills": missing_req,
            "all_skills": cand["skills"],
            "why_suitable": why_suitable,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    for rank_idx, item in enumerate(results, 1):
        item["rank"] = rank_idx

    if progress_callback:
        progress_callback(1.0, f"Completed ranking {len(results):,} candidates!")

    return results
