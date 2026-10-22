# %%writefile app.py
import re
import gradio as gr
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
stemmer = PorterStemmer()

extra_stop = {"need", "experience", "looking", "skilled", "plus", "required",
              "strong", "knowledge", "ability", "work", "working", "team",
              "good", "skills", "years", "etc"}
stop = set(ENGLISH_STOP_WORDS) | extra_stop

aliases = {"mysql": "sql", "postgresql": "sql", "postgres": "sql",
           "sqlite": "sql", "ml": "machine learning", "js": "javascript",
           "powerbi": "power bi"}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for short, full in aliases.items():
        text = re.sub(r"\b" + short + r"\b", full, text)
    return text

def keywords(text):
    out = {}
    for w in clean_text(text).split():
        if w in stop or len(w) < 2:
            continue
        out.setdefault(stemmer.stem(w), w)
    return out

def phrases(text):
    out = {}
    for chunk in re.split(r"[,.;:\n]", text.lower()):
        toks = clean_text(chunk).split()
        for a, b in zip(toks, toks[1:]):
            if a in stop or b in stop or len(a) < 2 or len(b) < 2:
                continue
            key = stemmer.stem(a) + " " + stemmer.stem(b)
            out.setdefault(key, a + " " + b)
    return out

def split_chunks(text, max_words=60):
    parts = re.split(r"\n|(?<=[.;])\s+", text)
    chunks = []
    for part in parts:
        words = part.split()
        if len(words) < 3:
            continue
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i:i + max_words]))
    return chunks or [text]

def smart_score(resume_text, jd_text):
    r_emb = model.encode(split_chunks(resume_text))
    j_emb = model.encode(split_chunks(jd_text))
    sims = util.cos_sim(j_emb, r_emb)
    return sims.max(dim=1).values.mean().item() * 100

def match(resume_text, jd_text, top_n=8):
    score = smart_score(resume_text, jd_text)
    jd_kw = keywords(jd_text)
    res_kw = keywords(resume_text)
    missing_roots = {r for r in jd_kw if r not in res_kw}

    result, covered = [], set()
    for key, phrase in phrases(jd_text).items():
        roots = key.split()
        if all(r in missing_roots for r in roots):
            result.append(phrase)
            covered.update(roots)
    for root, word in jd_kw.items():
        if root in missing_roots and root not in covered:
            result.append(word)
    return round(score, 1), result[:top_n]

def run(resume_text, jd_text):
    score, missing = match(resume_text, jd_text)
    return f"{score}%", ", ".join(missing) if missing else "None, great coverage!"

demo = gr.Interface(
    fn=run,
    inputs=[gr.Textbox(lines=10, label="Paste your resume"),
            gr.Textbox(lines=10, label="Paste the job description")],
    outputs=[gr.Textbox(label="Match score"),
             gr.Textbox(label="Missing keywords")],
    title="Resume vs JD Matcher",
)

if __name__ == "__main__":
    demo.launch()