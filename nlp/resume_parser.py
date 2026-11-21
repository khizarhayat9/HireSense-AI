"""Phase 3: Structured Resume Information Extractor."""

from datetime import datetime
import re
from nlp.text_preprocessing import clean_whitespace
from nlp.skill_extractor import SkillExtractor

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}")
URL_REGEX = re.compile(r"(?:https?://)?(?:www\.)?(?:github\.com|linkedin\.com)/[a-zA-Z0-9_-]+/?", re.IGNORECASE)

SECTION_HEADERS = {
    "summary": re.compile(r"^(?:summary|profile|objective|about me)\s*:?$", re.IGNORECASE),
    "education": re.compile(r"^(?:education|academic background|qualifications)\s*:?$", re.IGNORECASE),
    "experience": re.compile(r"^(?:experience|work experience|employment|work history)\s*:?$", re.IGNORECASE),
    "projects": re.compile(r"^(?:projects|personal projects|key projects|research projects)\s*:?$", re.IGNORECASE),
    "certifications": re.compile(r"^(?:certifications|certificates|licenses|courses)\s*:?$", re.IGNORECASE),
    "skills": re.compile(r"^(?:skills|technical skills|core competencies|technologies)\s*:?$", re.IGNORECASE),
}

DEGREE_PATTERNS = [
    (4, re.compile(r"\b(?:ph\.?d|doctorate)\b", re.IGNORECASE), "Ph.D."),
    (3, re.compile(r"\b(?:m\.?s\.?c?|m\.?tech|master(?:'s)?|mba)\b", re.IGNORECASE), "Master's"),
    (2, re.compile(r"\b(?:b\.?s\.?c?|b\.?e\.?|b\.?tech|b\.?a\.?|bachelor(?:'s)?)\b", re.IGNORECASE), "Bachelor's"),
    (1, re.compile(r"\b(?:associate(?:'s)?|diploma)\b", re.IGNORECASE), "Associate/Diploma"),
]


def _extract_candidate_name(text: str, filename: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()][:6]
    for line in lines[:3]:
        if EMAIL_REGEX.search(line) or PHONE_REGEX.search(line) or "http" in line.lower():
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words):
            return line.title()
    return filename.rsplit(".", 1)[0].replace("_", " ").title()


def _segment_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {k: [] for k in SECTION_HEADERS}
    sections["other"] = []
    current_section = "other"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched_header = None
        if len(line.split()) <= 4:
            for sec_name, pattern in SECTION_HEADERS.items():
                if pattern.match(line):
                    matched_header = sec_name
                    break
        if matched_header:
            current_section = matched_header
        else:
            sections[current_section].append(line)

    return sections


def _estimate_experience_years(text: str) -> float:
    explicit = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:hands-on\s+)?experience", text, re.I)
    if explicit:
        return max(float(x) for x in explicit)

    current_year = datetime.now().year
    ranges = re.findall(r"\b(20\d{2})\s*[-–to]+\s*(20\d{2}|present|current)\b", text, re.I)
    total_years = 0.0
    for start_str, end_str in ranges:
        start_y = int(start_str)
        end_y = current_year if end_str.lower() in {"present", "current"} else int(end_str)
        if 1995 <= start_y <= end_y <= current_year + 1:
            total_years += max(0.5, float(end_y - start_y))
    return round(min(total_years, 30.0), 1)


def _extract_education_level(edu_text: str) -> tuple[int, str, list[str]]:
    lines = [l for l in edu_text.splitlines() if l.strip()]
    best_rank = 0
    best_label = "Not Specified"
    for rank, pattern, label in DEGREE_PATTERNS:
        if pattern.search(edu_text) and rank > best_rank:
            best_rank = rank
            best_label = label
    return best_rank, best_label, lines


def parse_resume(raw_text: str, filename: str = "resume.pdf") -> dict:
    cleaned = clean_whitespace(raw_text)
    sections = _segment_sections(cleaned)
    skill_extractor = SkillExtractor()
    skills_data = skill_extractor.extract_skills(cleaned)

    email_match = EMAIL_REGEX.search(cleaned)
    phone_match = PHONE_REGEX.search(cleaned)
    urls = list(dict.fromkeys(URL_REGEX.findall(cleaned)))

    edu_source = "\n".join(sections["education"]) if sections["education"] else cleaned
    edu_rank, edu_level, edu_lines = _extract_education_level(edu_source)

    exp_source = "\n".join(sections["experience"]) if sections["experience"] else cleaned
    est_years = _estimate_experience_years(exp_source)

    return {
        "candidate_name": _extract_candidate_name(cleaned, filename),
        "email": email_match.group(0) if email_match else "Not Provided",
        "phone": phone_match.group(0).strip() if phone_match else "Not Provided",
        "links": urls,
        "summary": " ".join(sections["summary"]),
        "education_level": edu_level,
        "education_rank": edu_rank,
        "education_details": edu_lines[:6],
        "estimated_experience_years": est_years,
        "experience_details": sections["experience"][:15],
        "projects": sections["projects"][:12],
        "certifications": sections["certifications"][:8],
        "technical_skills": skills_data["technical_skills"],
        "soft_skills": skills_data["soft_skills"],
        "skills_by_category": skills_data["by_category"],
        "all_skills": skills_data["all_skills"],
        "cleaned_text": cleaned,
    }