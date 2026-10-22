# Resume Checker

A web app that compares a resume (PDF) with a job description. It shows an
overall match score, which skills matched, and which skills are missing.

![App screenshot](screenshot.png)

## Features
- Upload your resume as a **PDF** and paste a job description
- **Overall match %** shown as an animated ring
- **Matched vs missing skills** donut chart
- **Missing skills bars**, sorted by how often the JD mentions them
- Handles long resumes by comparing chunk by chunk

## How it works
1. **PDF to text:** the resume text is extracted with `pypdf`.
2. **Meaning similarity:** sentence embeddings (`all-MiniLM-L6-v2`) compare
   each part of the JD with the best-fitting part of the resume.
3. **Skills coverage:** known skills are detected in both texts, with aliases
   (for example, MySQL counts as SQL).
4. **Overall score** = 60% skills coverage + 40% meaning similarity.

## Why embeddings instead of only TF-IDF?
My first version used TF-IDF, which only matches exact words, so "ML" and
"machine learning" scored 0%. Embeddings understand meaning, and the same pair
scored about 61%.

## Tech stack
Python, Flask, sentence-transformers, scikit-learn, NLTK, pypdf, HTML/CSS/JavaScript

## Run it locally
```bash
git clone https://github.com/Jonnagiri12/resume-jd-matcher.git
cd resume-jd-matcher
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Limitations
- The score is a rough guide to similarity, not a hiring prediction.
- The skills list is hand-made, so skills not in it are not detected.
- Scanned (image-only) PDFs can't be read.

## Future improvements
- Larger skills database
- Support for DOCX resumes