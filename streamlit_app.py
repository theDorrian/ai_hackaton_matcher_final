import os
import re
import time
from io import StringIO
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

st.set_page_config(page_title="Resume Matcher", layout="wide")

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def read_pdf_text_from_bytes(file) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        parts = []
        for p in reader.pages:
            try:
                parts.append(p.extract_text() or "")
            except Exception:
                parts.append("")
        t = "\n".join(parts).strip()
        if t:
            return t
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text_to_fp
        output = StringIO()
        file.seek(0)
        extract_text_to_fp(file, output)
        return output.getvalue().strip()
    except Exception:
        return ""

def minmax01(M: np.ndarray) -> np.ndarray:
    m, Mx = M.min(), M.max()
    return (M - m) / (Mx - m + 1e-9)

def compute_blocks(resumes_df: pd.DataFrame,
                   jobs_df: pd.DataFrame,
                   use_local_emb: bool,
                   emb_model_name: str = "intfloat/multilingual-e5-base") -> Dict[str, np.ndarray]:
    R = resumes_df["text"].tolist()
    J = jobs_df["text"].tolist()

    tfidf = TfidfVectorizer(max_features=45000, ngram_range=(1, 2))
    X_res = tfidf.fit_transform(R)
    X_job = tfidf.transform(J)
    C_tfidf = cosine_similarity(X_job, X_res)

    if use_local_emb and _HAS_ST:
        model = SentenceTransformer(emb_model_name)

        def _prep(texts, is_query=False):
            if "e5" in (emb_model_name or "").lower():
                return [("query: " if is_query else "passage: ") + (t or "") for t in texts]
            return texts

        E_res = model.encode(_prep(R, is_query=False), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        E_job = model.encode(_prep(J, is_query=True),  batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        C_emb = np.asarray(E_job) @ np.asarray(E_res).T
    else:
        C_emb = np.zeros_like(C_tfidf)

    job_titles = [norm((t.split("\n")[0] if t else "")).lower() for t in J]
    res_titles = [norm((t.split("\n")[0] if t else "")).lower() for t in R]
    TitleSim = np.zeros_like(C_tfidf, dtype=np.float32)
    for i in range(TitleSim.shape[0]):
        jt = job_titles[i]
        for j in range(TitleSim.shape[1]):
            TitleSim[i, j] = fuzz.token_set_ratio(jt, res_titles[j]) / 100.0

    skill_pat = re.compile(r"\b[A-Za-z][A-Za-z0-9\.\+\#\-]{1,}\b")
    def skill_set(text: str) -> set:
        return set(t.lower() for t in skill_pat.findall(text))

    job_skill_sets = [skill_set(t) for t in J]
    res_skill_sets = [skill_set(t) for t in R]
    SkillJac = np.zeros_like(C_tfidf, dtype=np.float32)
    for i in range(SkillJac.shape[0]):
        js = job_skill_sets[i]
        for j in range(SkillJac.shape[1]):
            rs = res_skill_sets[j]
            u = len(js | rs)
            SkillJac[i, j] = (len(js & rs) / u) if u else 0.0

    S_tfidf = minmax01(C_tfidf)
    S_emb   = minmax01(C_emb) if use_local_emb and _HAS_ST else C_emb
    S_title = minmax01(TitleSim)
    S_skill = minmax01(SkillJac)

    return dict(S_tfidf=S_tfidf, S_emb=S_emb, S_title=S_title, S_skill=S_skill)

def combine_score(blocks: Dict[str, np.ndarray], w: Dict[str, float]) -> np.ndarray:
    return (w["tfidf"] * blocks["S_tfidf"]
          + w["embed"] * blocks["S_emb"]
          + w["title"] * blocks["S_title"]
          + w["skills"] * blocks["S_skill"])

def compact_text(text: str, head=800) -> str:
    return norm(text)[:head]

def llm_explain_pair(client: "OpenAI", job_text: str, res_text: str, model="gpt-4o-mini", temperature=0.4) -> str:
    system = (
        "Ты HR-аналитик. Даны текст вакансии и резюме. "
        "Дай короткое объяснение на русском (2–3 предложения), почему кандидат подходит или нет. "
        "Упоминай совпадающие навыки/технологии и релевантный опыт. Не используй числовые метрики."
    )
    user = f"Вакансия:\n{job_text}\n\nРезюме:\n{res_text}"
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=160,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[API error: {e}]"

st.header("Automated Resume Matching")

st.sidebar.header("Settings")
use_local_emb = st.sidebar.checkbox("Local embeddings (E5)", value=_HAS_ST)
weights = {
    "tfidf": st.sidebar.slider("Weight: TF-IDF", 0.0, 1.0, 0.25, 0.05),
    "embed": st.sidebar.slider("Weight: Embeddings", 0.0, 1.0, 0.45, 0.05),
    "title": st.sidebar.slider("Weight: Title", 0.0, 1.0, 0.15, 0.05),
    "skills": st.sidebar.slider("Weight: Skills", 0.0, 1.0, 0.15, 0.05),
}
if abs(sum(weights.values()) - 1.0) > 1e-6:
    st.sidebar.warning("Weights must sum to 1.0")

topk_view = st.sidebar.slider("Top-K per job to display", 5, 50, 10, 1)
OPENAI_KEY = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
model_name = st.sidebar.selectbox("LLM model", ["gpt-4o-mini"], index=0)
sleep_sec = st.sidebar.slider("Pause between LLM calls (sec.)", 0.0, 1.0, 0.2, 0.1)

st.subheader("Upload")
c1, c2 = st.columns(2)
with c1:
    jobs_files = st.file_uploader("Jobs: Vacancy*.pdf", type=["pdf"], accept_multiple_files=True, key="jobs")
with c2:
    resumes_files = st.file_uploader("Resumes: Resume*.pdf", type=["pdf"], accept_multiple_files=True, key="resumes")

process = st.button("Build ranking", type="primary", use_container_width=True)

if process:
    if not jobs_files or not resumes_files:
        st.error("Upload at least one job and one resume PDF.")
        st.stop()

    with st.spinner("Reading PDFs..."):
        job_rows = []
        for f in jobs_files:
            text = read_pdf_text_from_bytes(f)
            job_rows.append({"name": f.name, "text": norm(text)})
        jobs_df = pd.DataFrame(job_rows)

        res_rows = []
        for f in resumes_files:
            text = read_pdf_text_from_bytes(f)
            res_rows.append({"name": f.name, "text": norm(text)})
        resumes_df = pd.DataFrame(res_rows)

    with st.spinner("Scoring..."):
        blocks = compute_blocks(resumes_df, jobs_df, use_local_emb=use_local_emb)
        Score = combine_score(blocks, weights)

    tabs = st.tabs([n for n in jobs_df["name"].tolist()])

    if not _HAS_OPENAI and not OPENAI_KEY:
        st.error("Install openai and provide API key to generate explanations.")
        st.stop()

    client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY and _HAS_OPENAI else None

    for i, tab in enumerate(tabs):
        with tab:
            row = Score[i]
            order = np.argsort(-row)
            top_idx = order[:topk_view]

            job_text = compact_text(jobs_df.iloc[i]["text"])

            records = []
            prog = st.progress(0.0, text="Generating explanations...")
            for k, j in enumerate(top_idx, start=1):
                res_text = compact_text(resumes_df.iloc[j]["text"])
                expl = llm_explain_pair(client, job_text, res_text, model=model_name) if client else ""
                records.append({
                    "Rank": k,
                    "Score": float(row[j]),
                    "Resume": resumes_df.iloc[j]["name"],
                    "Explanation": expl
                })
                prog.progress(k/len(top_idx))
                time.sleep(sleep_sec)
            prog.empty()

            st.subheader("Ranking table (Top-K)")
            df_show = pd.DataFrame(records)
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            job_full = pd.DataFrame({
                "rank": list(range(1, len(order)+1)),
                "score": [float(row[j]) for j in order],
                "resume": [resumes_df.iloc[j]["name"] for j in order]
            })
            st.download_button(
                "Download CSV for this job",
                job_full.to_csv(index=False).encode("utf-8"),
                file_name=f"ranking_{jobs_df.iloc[i]['name'].replace('.pdf','')}.csv",
                mime="text/csv"
            )
