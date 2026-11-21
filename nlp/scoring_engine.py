"""Phase 7: Explainable Scoring & Candidate Ranking Engine."""

from config import DEFAULT_SCORING_WEIGHTS
from nlp.tfidf_matcher import compute_tfidf_similarity
from nlp.semantic_matcher import compute_semantic_similarity, find_partial_skill_matches


def evaluate_candidate(
    parsed_resume: dict,
    parsed_jd: dict,
    weights: dict[str, float] | None = None,
) -> dict:
    active_weights = weights or DEFAULT_SCORING_WEIGHTS.copy()
    total_w = sum(active_weights.values()) or 1.0
    w = {k: v / total_w for k, v in active_weights.items()}

    req_tech = set(parsed_jd.get("required_technical_skills", []))
    pref_tech = set(parsed_jd.get("preferred_technical_skills", []))
    cand_tech = set(parsed_resume.get("technical_skills", []))

    matching_skills = sorted(req_tech.intersection(cand_tech))
    missing_skills = sorted(req_tech - cand_tech)
    bonus_matching = sorted(pref_tech.intersection(cand_tech))
    additional_skills = sorted(cand_tech - req_tech - pref_tech)

    partial_matches = find_partial_skill_matches(missing_skills, list(cand_tech))
    partially_matched_names = {pm["required_skill"] for pm in partial_matches}

    if req_tech:
        exact_credit = len(matching_skills)
        partial_credit = 0.5 * len(partially_matched_names)
        base_skill_ratio = min(1.0, (exact_credit + partial_credit) / len(req_tech))
        bonus_boost = min(0.10, 0.03 * len(bonus_matching))
        skills_score = min(1.0, base_skill_ratio + bonus_boost)
    else:
        skills_score = 1.0

    semantic_score = compute_semantic_similarity(
        parsed_resume.get("cleaned_text", ""),
        parsed_jd.get("cleaned_text", ""),
    )

    tfidf_res = compute_tfidf_similarity(
        parsed_resume.get("cleaned_text", ""),
        parsed_jd.get("cleaned_text", ""),
    )
    tfidf_score = tfidf_res["similarity"]

    req_years = parsed_jd.get("min_experience_years", 0.0)
    cand_years = parsed_resume.get("estimated_experience_years", 0.0)
    if req_years <= 0 or cand_years >= req_years:
        experience_score = 1.0
    else:
        experience_score = round(max(0.2, cand_years / req_years), 4)

    req_edu = parsed_jd.get("min_education_rank", 0)
    cand_edu = parsed_resume.get("education_rank", 0)
    if req_edu <= 0 or cand_edu >= req_edu:
        education_score = 1.0
    elif cand_edu > 0:
        education_score = round(cand_edu / req_edu, 4)
    else:
        education_score = 0.3

    raw_composite = (
        w["skills_match"] * skills_score
        + w["semantic_similarity"] * semantic_score
        + w["tfidf_similarity"] * tfidf_score
        + w["experience_match"] * experience_score
        + w["education_match"] * education_score
    )
    overall_score = round(raw_composite * 100, 1)

    strengths: list[str] = []
    gaps: list[str] = []
    recommendations: list[str] = []

    if matching_skills:
        strengths.append(f"Demonstrated core required skills: {', '.join(matching_skills)}.")
    if bonus_matching:
        strengths.append(f"Possesses preferred/bonus qualifications: {', '.join(bonus_matching)}.")
    if partial_matches:
        pairs = [f"{p['required_skill']} (via {p['related_candidate_skill']})" for p in partial_matches]
        strengths.append(f"Transferable domain overlap detected: {', '.join(pairs)}.")
    if cand_years >= req_years and req_years > 0:
        strengths.append(f"Meets experience benchmark ({cand_years} yrs estimated vs. {req_years} yrs required).")
    if parsed_resume.get("projects"):
        strengths.append(f"Includes {len(parsed_resume['projects'])} documented project lines demonstrating applied execution.")
    if not strengths:
        strengths.append("General professional background captured; review portfolio links for domain depth.")

    true_missing = [s for s in missing_skills if s not in partially_matched_names]
    if true_missing:
        gaps.append(f"Missing required technical skills: {', '.join(true_missing)}.")
        recommendations.append(f"Assess familiarity or plan onboarding around: {', '.join(true_missing[:4])}.")
    if cand_years < req_years:
        gaps.append(f"Estimated experience ({cand_years} yrs) is below target ({req_years} yrs).")
        recommendations.append("Evaluate depth of internships, open-source contributions, or capstone projects to offset tenure.")
    if cand_edu < req_edu:
        gaps.append(f"Listed education ({parsed_resume.get('education_level')}) is below stated preference ({parsed_jd.get('min_education_label')}).")
    if not gaps:
        gaps.append("No major technical or qualification gaps identified against parsed JD criteria.")
        recommendations.append("Proceed to structured technical interview focusing on system design and practical implementation.")

    return {
        "candidate_name": parsed_resume["candidate_name"],
        "email": parsed_resume["email"],
        "phone": parsed_resume["phone"],
        "overall_score": overall_score,
        "sub_scores": {
            "skills_match": round(skills_score * 100, 1),
            "semantic_similarity": round(semantic_score * 100, 1),
            "tfidf_similarity": round(tfidf_score * 100, 1),
            "experience_match": round(experience_score * 100, 1),
            "education_match": round(education_score * 100, 1),
        },
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "partial_matches": partial_matches,
        "bonus_skills": bonus_matching,
        "additional_skills": additional_skills,
        "estimated_experience_years": cand_years,
        "education_level": parsed_resume["education_level"],
        "top_shared_terms": tfidf_res["top_shared_terms"],
        "strengths": strengths,
        "areas_for_improvement": gaps,
        "recommendations": recommendations,
        "parsed_resume": parsed_resume,
    }


def rank_candidates(results: list[dict]) -> list[dict]:
    sorted_list = sorted(results, key=lambda x: x["overall_score"], reverse=True)
    for idx, item in enumerate(sorted_list, start=1):
        item["rank"] = idx
    return sorted_list
