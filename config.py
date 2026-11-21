"""Central configuration for HireSense AI."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_RESUMES_DIR = DATA_DIR / "sample_resumes"
SAMPLE_JOBS_DIR = DATA_DIR / "sample_jobs"
DB_PATH = DATA_DIR / "hiresense.db"
TAXONOMY_PATH = DATA_DIR / "skills_taxonomy.json"

SPACY_MODEL = "en_core_web_sm"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_MB = 10
MIN_EXTRACTABLE_CHARS = 100

DEFAULT_SCORING_WEIGHTS = {
    "skills_match": 0.40,
    "semantic_similarity": 0.25,
    "tfidf_similarity": 0.10,
    "experience_match": 0.15,
    "education_match": 0.10,
}
