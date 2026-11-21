"""Phase 4: Job Description Analyzer."""

import re
from nlp.text_preprocessing import clean_whitespace
from nlp.skill_extractor import SkillExtractor
from nlp.resume_parser import DEGREE_PATTERNS


def analyze_job_description(raw_jd: str) -> dict:
    cleaned = clean_whitespace(raw_jd)
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]

    title = "Target Role"
    for line in lines[:5]:
        m = re.match(r"^(?:job\s*title|role|position)\s*:\s*(.+)$", line, re.I)
        if m:
            title = m.group(1).strip()
            break
    if title == "Target Role" and lines:
        title = lines[0][:80]

    pref_split = re.split(r"\b(?:preferred|bonus|nice to have|optional)\s*(?:skills|qualifications)?\s*:?", cleaned, flags=re.I)
    required_block = pref_split[0]
    preferred_block = pref_split[1] if len(pref_split) > 1 else ""

    extractor = SkillExtractor()
    req_skills_data = extractor.extract_skills(required_block)
    pref_skills_data = extractor.extract_skills(preferred_block) if preferred_block else {"technical_skills": [], "soft_skills": []}

    exp_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:[-–to]+\s*\d+(?:\.\d+)?)?\+?\s*(?:years?|yrs?)", cleaned, re.I)
    min_experience = float(exp_match.group(1)) if exp_match else 0.0

    min_edu_rank = 0
    min_edu_label = "Not Specified"
    for rank, pattern, label in DEGREE_PATTERNS:
        if pattern.search(cleaned):
            if min_edu_rank == 0 or rank < min_edu_rank:
                min_edu_rank = rank
                min_edu_label = label

    bullet_lines = [l.lstrip("- ").strip() for l in lines if l.startswith("- ")][:15]

    return {
        "job_title": title,
        "required_technical_skills": req_skills_data["technical_skills"],
        "preferred_technical_skills": [
            s for s in pref_skills_data["technical_skills"] if s not in req_skills_data["technical_skills"]
        ],
        "soft_skills": sorted(set(req_skills_data["soft_skills"] + pref_skills_data["soft_skills"])),
        "min_experience_years": min_experience,
        "min_education_rank": min_edu_rank,
        "min_education_label": min_edu_label,
        "key_requirements": bullet_lines,
        "cleaned_text": cleaned,
    }
