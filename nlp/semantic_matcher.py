"""
Phase 6: Contextual Semantic Similarity Engine.
Uses Sentence-Transformers (all-MiniLM-L6-v2) when available, and gracefully falls back
to SVD Latent Semantic Analysis (LSA) + concept-synonym embeddings if Windows Application
Control blocks compiled pyarrow/torch DLLs locally.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from config import EMBEDDING_MODEL

_MODEL_INSTANCE = None
_USE_FALLBACK = False

# Domain concept synonyms so semantic matching recognizes ML/AI relationships even in fallback mode
CONCEPT_EXPANSIONS = {
    "machine learning": "machine learning ml predictive modeling statistical modeling supervised learning",
    "artificial intelligence": "artificial intelligence ai machine learning deep learning neural networks",
    "natural language processing": "natural language processing nlp text classification text mining",
    "data analysis": "data analysis exploratory data analysis eda data visualization analytics statistics",
    "deep learning": "deep learning neural networks pytorch tensorflow transformers",
}


def _expand_concepts(text: str) -> str:
    lowered = text.lower()
    additions = [expansion for key, expansion in CONCEPT_EXPANSIONS.items() if key in lowered]
    return lowered + " " + " ".join(additions)


def get_embedding_model():
    global _MODEL_INSTANCE, _USE_FALLBACK
    if _USE_FALLBACK:
        return None
    if _MODEL_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL)
        except Exception:
            _USE_FALLBACK = True
            return None
    return _MODEL_INSTANCE


def _lsa_similarity(text_a: str, text_b: str) -> float:
    exp_a = _expand_concepts(text_a)
    exp_b = _expand_concepts(text_b)
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf = vec.fit_transform([exp_a, exp_b])
    n_components = min(2, tfidf.shape[0], tfidf.shape[1] - 1)
    if n_components >= 1:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = svd.fit_transform(tfidf)
        denom = np.linalg.norm(dense[0]) * np.linalg.norm(dense[1])
        score = float(np.dot(dense[0], dense[1]) / denom) if denom > 0 else 0.0
    else:
        score = float((tfidf[0] @ tfidf[1].T).toarray()[0][0])
    return round(max(0.0, min(1.0, score)), 4)


def compute_semantic_similarity(resume_text: str, jd_text: str) -> float:
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    model = get_embedding_model()
    if model is None:
        return _lsa_similarity(resume_text, jd_text)

    embeddings = model.encode([jd_text[:4000], resume_text[:4000]], normalize_embeddings=True)
    score = float(np.dot(embeddings[0], embeddings[1]))
    return round(max(0.0, min(1.0, score)), 4)


def find_partial_skill_matches(
    missing_skills: list[str],
    candidate_skills: list[str],
    threshold: float = 0.62,
) -> list[dict]:
    if not missing_skills or not candidate_skills:
        return []

    model = get_embedding_model()
    if model is None:
        partial_matches = []
        for req in missing_skills:
            req_exp = _expand_concepts(req)
            best_cand, best_score = "", 0.0
            for cand in candidate_skills:
                sim = _lsa_similarity(req_exp, cand)
                if sim > best_score:
                    best_score = sim
                    best_cand = cand
            if best_score >= 0.45:
                partial_matches.append({
                    "required_skill": req,
                    "related_candidate_skill": best_cand,
                    "similarity": round(best_score, 2),
                })
        return partial_matches

    missing_embs = model.encode(missing_skills, normalize_embeddings=True)
    cand_embs = model.encode(candidate_skills, normalize_embeddings=True)

    partial_matches = []
    for idx, req_skill in enumerate(missing_skills):
        sims = np.dot(cand_embs, missing_embs[idx])
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= threshold:
            partial_matches.append({
                "required_skill": req_skill,
                "related_candidate_skill": candidate_skills[best_idx],
                "similarity": round(best_score, 2),
            })
    return partial_matches