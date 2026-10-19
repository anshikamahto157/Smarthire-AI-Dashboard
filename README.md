# ⚡ SmartHire - AI Talent Intelligence Hub

SmartHire is an intelligent ATS evaluation and job recommendation dashboard built using Machine Learning, Natural Language Processing (NLP), and Streamlit.

## Key Features
- **Resume Domain Classification:** Multi-class role classification trained on diverse technical domains.
- **Location-Aware Semantic Matching:** Matches candidate profiles with 3,000+ curated jobs across Indian metropolitan hubs using Cosine Similarity.
- **5-Star Resume Health Audit:** Evaluates contact completeness, action verbs strength, quantifiable metrics, section structure, and content depth.
- **Skill Gap Analysis:** Highlights candidate competencies versus target position expectations.
- **Exportable Audit Report:** Instant candidate assessment summary download.

## Tech Stack
- **Frontend / Dashboard:** Streamlit, Plotly
- **NLP & Machine Learning:** Scikit-Learn (TF-IDF Vectorizer, LinearSVC, Cosine Similarity)
- **Document Parsing:** PyMuPDF (`fitz`), Regular Expressions (Regex)

## How to Run Locally
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd smarthire