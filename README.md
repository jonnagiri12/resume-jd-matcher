# resume-jd-matcher
# Resume vs JD Matcher

A small NLP tool that compares a resume with a job description. It gives a
match score and lists the important keywords the resume is missing.

## Features
- **Match score** using sentence embeddings (meaning-based, not just exact words)
- **Missing keywords** shown as skills, including two-word phrases like "power bi"
- Handles long resumes by comparing chunk by chunk
- Simple web interface built with Gradio

## How it works
1. **Embeddings:** the text is converted into vectors with the
   `all-MiniLM-L6-v2` model (sentence-transformers).
2. **Chunked scoring:** each part of the job description is matched to the
   best-fitting part of the resume, and the best matches are averaged.
3. **Keyword gap:** words are cleaned, reduced to their roots (stemming), and
   filler words are removed. Skills in the JD that are not in the resume
   are listed.
4. **Aliases:** equivalent terms are treated as the same skill (for example,
   MySQL counts as SQL).

## Why embeddings instead of only TF-IDF?
My first version used TF-IDF with cosine similarity, but it only matches exact
words. "ML" and "machine learning" scored 0%. Embeddings understand meaning,
and the same pair scored about 61%, so the final version uses embeddings for
the score and keeps simple keyword logic for the missing-skills list.

## Tech stack
Python, sentence-transformers, scikit-learn, NLTK, Gradio

## Run it locally
```bash
git clone https://github.com/Jonnagiri12/resume-jd-matcher.git
cd resume-jd-matcher
pip install -r requirements.txt
python app.py
```
Then open the local link shown in the terminal.

## Limitations
- The score is a rough guide to similarity, not a hiring prediction.
- The keyword list is rule-based, so unusual skill names may be missed.
- The alias list is small and can be extended.

## Future improvements
- Extract skills with a proper skills database
- Support PDF resume upload
- Deploy online