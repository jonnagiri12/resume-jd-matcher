import io
import re
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB upload limit

model = SentenceTransformer("all-MiniLM-L6-v2")
stemmer = PorterStemmer()

extra_stop = {"need", "experience", "looking", "skilled", "plus", "required",
              "strong", "knowledge", "ability", "work", "working", "team",
              "good", "skills", "years", "etc"}
stop = set(ENGLISH_STOP_WORDS) | extra_stop

aliases = {"mysql": "sql", "postgresql": "sql", "postgres": "sql",
           "sqlite": "sql", "ml": "machine learning", "js": "javascript",
           "powerbi": "power bi"}

# Lowercase, no punctuation. Add more skills here anytime.
SKILLS = {
    "python", "java", "javascript", "typescript", "sql", "html", "css",
    "react", "angular", "flask", "django", "fastapi", "spring boot",
    "git", "github", "docker", "kubernetes", "aws", "azure", "gcp", "linux",
    "rest api", "api", "microservices", "devops", "ci cd", "jenkins",
    "machine learning", "deep learning", "nlp", "computer vision", "opencv",
    "tensorflow", "pytorch", "scikit learn", "pandas", "numpy", "matplotlib",
    "seaborn", "llm", "rag", "langchain", "generative ai",
    "power bi", "tableau", "excel", "statistics", "data analysis",
    "data visualization", "etl", "spark", "hadoop", "kafka",
    "mongodb", "nosql", "redis", "data structures", "algorithms", "oop",
    "agile", "scrum", "jira", "testing", "unit testing", "selenium",
    "iot", "mqtt", "arduino", "raspberry pi", "cloud", "cybersecurity",
    "networking",
}

# ------------------------------------------------------------ logic
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

def semantic_score(resume_text, jd_text):
    r_emb = model.encode(split_chunks(resume_text))
    j_emb = model.encode(split_chunks(jd_text))
    raw = util.cos_sim(j_emb, r_emb).max(dim=1).values.mean().item()
    return min(max((raw - 0.20) / 0.40, 0), 1) * 100

def find_skills(text):
    padded = " " + clean_text(text) + " "
    return {s for s in SKILLS if f" {s} " in padded}

def mentions(skill, text):
    return len(re.findall(r"\b" + re.escape(skill) + r"\b", clean_text(text)))

def other_missing(resume_text, jd_text, skip, top_n=8):
    jd_kw = keywords(jd_text)
    res_kw = keywords(resume_text)
    missing_roots = {r for r in jd_kw if r not in res_kw}
    result, covered = [], set()
    for key, phrase in phrases(jd_text).items():
        roots = key.split()
        if all(r in missing_roots for r in roots) and phrase not in skip:
            result.append(phrase)
            covered.update(roots)
    for root, word in jd_kw.items():
        if root in missing_roots and root not in covered and word not in skip:
            result.append(word)
    return result[:top_n]

def read_pdf(data):
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)

# ----------------------------------------------------------- routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("resume")
    jd = request.form.get("jd", "").strip()

    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify(error="Please upload your resume as a PDF."), 400
    if not jd:
        return jsonify(error="Please paste the job description."), 400

    try:
        resume = read_pdf(f.read())
    except Exception:
        return jsonify(error="Could not open that PDF. Please try another file."), 400
    if len(resume.strip()) < 30:
        return jsonify(error="No text found in this PDF. It may be a scanned image. "
                             "Please upload a text-based PDF."), 400

    sem = semantic_score(resume, jd)
    jd_skills = find_skills(jd)
    res_skills = find_skills(resume)
    matched = sorted(jd_skills & res_skills)
    missing = sorted(jd_skills - res_skills)

    if len(jd_skills) >= 3:
        coverage = len(matched) / len(jd_skills) * 100
        overall = 0.6 * coverage + 0.4 * sem
    else:
        coverage = None
        overall = sem

    verdict = ("Strong match" if overall >= 70
               else "Moderate match" if overall >= 45 else "Low match")

    missing_list = sorted(
        [{"skill": s, "count": max(mentions(s, jd), 1)} for s in missing],
        key=lambda x: -x["count"],
    )

    return jsonify(
        overall=round(overall),
        semantic=round(sem),
        coverage=None if coverage is None else round(coverage),
        verdict=verdict,
        matched=matched,
        missing=missing_list,
        others=other_missing(resume, jd, skip=set(missing)),
    )

if __name__ == "__main__":
    app.run(debug=False)