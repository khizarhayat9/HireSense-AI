# 🎯 HireSense AI — Intelligent Resume Screening & Job Matching Platform
 DEMO LINK 
 https://hiresense-ai-by-mana.streamlit.app/

HireSense AI is an end-to-end Natural Language Processing (NLP) recruitment decision-support platform that parses candidate resumes (`.pdf`, `.docx`), structures job descriptions, evaluates lexical and semantic alignment, and produces explainable candidate rankings and downloadable PDF/Excel reports.

## Quick Start

```bash
cd HireSense-AI
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python setup_samples.py
pytest -v
streamlit run app/main.py
```
