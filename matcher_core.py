import os, re, json
from io import BytesIO, StringIO
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
    _EMB_MODEL = None
except Exception:
    _HAS_ST = False
    _EMB_MODEL = None
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

def norm(s): return re.sub(r"\s+", " ", (s or "")).strip()

def read_pdf_text(file_bytes):
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(file_bytes))
        t = "\n".join([(p.extract_text() or "") for p in reader.pages]).strip()
        if t: return t
    except: pass
    try:
        from pdfminer.high_level import extract_text_to_fp
        output = StringIO()
        extract_text_to_fp(BytesIO(file_bytes), output)
        return output.getvalue().strip()
    except: return ""

def _skill_set(text):
    pat = re.compile(r"\b[A-Za-z][A-Za-z0-9\.\+\#\-]{1,}\b")
    return set(t.lower() for t in pat.findall(text or ""))

def compute_signals(job_text, resume_text, use_local_emb=True):
    j, r = norm(job_text), norm(resume_text)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform([j, r])
    tfidf_sim = float(cosine_similarity(X[0], X[1])[0, 0])
    if use_local_emb and _HAS_ST:
        global _EMB_MODEL
        if _EMB_MODEL is None:
            _EMB_MODEL = SentenceTransformer("intfloat/multilingual-e5-base")
        def _prep(t, is_query=False): return ("query: " if is_query else "passage: ") + (t or "")
        ej = _EMB_MODEL.encode([_prep(j, True)], normalize_embeddings=True)[0]
        er = _EMB_MODEL.encode([_prep(r, False)], normalize_embeddings=True)[0]
        emb_sim = float((np.dot(ej, er) + 1) / 2)
    else:
        emb_sim = 0.0
    jt, rt = (j.split("\n")[0] if j else "").lower(), (r.split("\n")[0] if r else "").lower()
    title_sim = fuzz.token_set_ratio(jt, rt) / 100.0
    js, rs = _skill_set(j), _skill_set(r)
    u = len(js | rs)
    skill_sim = (len(js & rs) / u) if u else 0.0
    return {"tfidf": tfidf_sim, "emb": emb_sim, "title": title_sim, "skills": skill_sim}

def combine_base(signals, weights):
    return float(weights["tfidf"]*signals["tfidf"] + weights["embed"]*signals["emb"] +
                 weights["title"]*signals["title"] + weights["skills"]*signals["skills"])

SYSTEM_JSON = (
    "Ты HR-аналитик. Дай объективную оценку соответствия кандидата вакансии. "
    "Ответ строго в JSON с ключами: verdict (fit|partial|no_fit), explanation (строка)."
)

def _safe_json_loads(s):
    try: return json.loads(s)
    except:
        m = re.search(r"\{.*\}", s, flags=re.S)
        return json.loads(m.group(0)) if m else {}

def llm_verdict(job_text, resume_text, api_key=None, model="gpt-4o-mini", head_chars=900):
    key = api_key or OPENAI_API_KEY
    client = OpenAI(api_key=key)
    j, r = norm(job_text)[:head_chars], norm(resume_text)[:head_chars]
    user = f"Вакансия:\n{j}\n\nРезюме:\n{r}"
    resp = client.chat.completions.create(
        model=model, temperature=0.2, max_tokens=160,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":SYSTEM_JSON},{"role":"user","content":user}],
    )
    data = _safe_json_loads(resp.choices[0].message.content or "{}")
    v = data.get("verdict", "partial")
    if v not in {"fit", "partial", "no_fit"}: v = "partial"
    e = data.get("explanation", "")
    return v, e

def adjust_score(base, title_raw, skill_raw, verdict):
    vf = {"fit":1.10, "partial":1.00, "no_fit":0.70}[verdict]
    gate_t, gate_s = 0.25, 0.20
    g = (0.85 if title_raw<gate_t else 1)*(0.85 if skill_raw<gate_s else 1)
    score = base * vf * g
    return float(max(0, min(1, score)))
