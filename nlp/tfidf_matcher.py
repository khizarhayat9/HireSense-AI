"""Phase 5: Lexical TF-IDF & Cosine Similarity Matcher."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlp.text_preprocessing import preprocess_for_tfidf


def compute_tfidf_similarity(resume_text: str, jd_text: str) -> dict:
    proc_resume = preprocess_for_tfidf(resume_text)
    proc_jd = preprocess_for_tfidf(jd_text)

    if not proc_resume.strip() or not proc_jd.strip():
        return {"similarity": 0.0, "top_shared_terms": []}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    tfidf_matrix = vectorizer.fit_transform([proc_jd, proc_resume])
    sim_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])

    feature_names = vectorizer.get_feature_names_out()
    jd_vec = tfidf_matrix[0].toarray()[0]
    res_vec = tfidf_matrix[1].toarray()[0]
    overlap_weights = jd_vec * res_vec
    top_indices = overlap_weights.argsort()[::-1][:8]
    shared_terms = [feature_names[i] for i in top_indices if overlap_weights[i] > 0]

    return {
        "similarity": round(max(0.0, min(1.0, sim_score)), 4),
        "top_shared_terms": shared_terms,
    }
