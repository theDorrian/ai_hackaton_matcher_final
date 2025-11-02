import os, re, time
from io import StringIO
from typing import List, Dict
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
from openai import OpenAI

# ---------- base ui ----------
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.markdown(
    """
    <style>
    .main .block-container {max-width: 1200px;}
    .mode-btn {display:inline-block;margin-right:8px}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- helpers ----------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def read_pdf_text_from_bytes(file) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        parts = [(p.extract_text() or "") for p in reader.pages]
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
        E_res = model.encode(_prep(R, False), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        E_job = model.encode(_prep(J, True),  batch_size=128, show_progress_bar=False, normalize_embeddings=True)
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
    return dict(
        S_tfidf=minmax01(C_tfidf),
        S_emb=minmax01(C_emb) if use_local_emb and _HAS_ST else C_emb,
        S_title=minmax01(TitleSim),
        S_skill=minmax01(SkillJac),
        TitleRaw=TitleSim, SkillRaw=SkillJac
    )

def combine_score(B: Dict[str, np.ndarray], w: Dict[str, float]) -> np.ndarray:
    return w["tfidf"]*B["S_tfidf"] + w["embed"]*B["S_emb"] + w["title"]*B["S_title"] + w["skills"]*B["S_skill"]

def compact_text(text: str, head=900) -> str:
    return norm(text)[:head]

SYSTEM_JSON = (
    "Ты HR-аналитик. Дай объективную оценку соответствия кандидата вакансии. "
    "Ответ строго в JSON с ключами: verdict(one of: fit, partial, no_fit), explanation(str, 1-2 предложения). "
    "Учитывай навыки/стек/роль/опыт. Избегай противоречий."
)

def llm_verdict(client: OpenAI, job_text: str, res_text: str, model="gpt-4o-mini") -> Dict[str, str]:
    user = f"Вакансия:\n{job_text}\n\nРезюме:\n{res_text}"
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=120,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_JSON},
                      {"role": "user", "content": user}],
        )
        data = pd.io.json.loads(resp.choices[0].message.content)
        v = data.get("verdict", "partial")
        e = data.get("explanation", "")
        if v not in {"fit","partial","no_fit"}:
            v = "partial"
        return {"verdict": v, "explanation": e}
    except Exception as e:
        return {"verdict": "partial", "explanation": f"[API error: {e}]"}

def adjust_score(base: float, title_raw: float, skill_raw: float, verdict: str) -> float:
    vf = {"fit": 1.10, "partial": 1.00, "no_fit": 0.70}[verdict]
    gate_t, gate_s = 0.25, 0.20
    g = (0.85 if title_raw < gate_t else 1.0) * (0.85 if skill_raw < gate_s else 1.0)
    adj = base * vf * g
    return float(max(0.0, min(1.0, adj)))

# ---------- presentation ----------
PRESENTATION = [
    ("Проблема", "Много резюме на каждую вакансию. Ручная оценка медленная и субъективная."),
    ("Цель", "Автоматически сопоставлять резюме и вакансии. Давать оценку и краткое объяснение."),
    ("Архитектура", "PDF → Text → TF-IDF/Embeddings/Title/Skills → BaseScore → LLM-verdict → AdjScore → Ранжирование."),
    ("Модель", "Score = w1*TF-IDF + w2*Emb + w3*Title + w4*Skills. Вердикт LLM: fit/partial/no_fit. AdjScore = Base*factor*gates."),
    ("Интерфейс", "Streamlit. Отдельная загрузка Jobs и Resumes. Вкладки по вакансиям. Экспорт CSV. Объяснения от GPT-4o-mini."),
    ("Результаты", "Согласованность вердикта с человеком высокая, расходы низкие за счёт gpt-4o-mini и предвыбора кандидатов."),
    ("Дальше", "Кэш LLM, тонкая настройка весов, интеграция в ATS, мультиязычность.")
]

def render_presentation():
    st.subheader("Презентация")
    if "slide" not in st.session_state: st.session_state.slide = 0
    c1, c2, c3 = st.columns([1,1,6])
    prev = c1.button("Назад", use_container_width=True, key="prev")
    nxt  = c2.button("Дальше", use_container_width=True, key="next")
    if prev: st.session_state.slide = max(0, st.session_state.slide - 1)
    if nxt:  st.session_state.slide = min(len(PRESENTATION)-1, st.session_state.slide + 1)
    title, content = PRESENTATION[st.session_state.slide]
    st.markdown(f"### {title}")
    st.write(content)
    st.progress((st.session_state.slide+1)/len(PRESENTATION))
    md = "\n\n".join([f"## {t}\n{c}" for t,c in PRESENTATION])
    st.download_button("Скачать презентацию (Markdown)", md.encode("utf-8"), "presentation.md", "text/markdown")

# ---------- matcher ----------
def render_matcher():
    st.header("Automated Resume Matching")
    st.sidebar.header("Settings")
    use_local_emb = st.sidebar.checkbox("Local embeddings (E5)", value=True)
    weights = {
        "tfidf":  st.sidebar.slider("Weight TF-IDF",   0.0, 1.0, 0.25, 0.05),
        "embed":  st.sidebar.slider("Weight Embeddings", 0.0, 1.0, 0.20, 0.05),
        "title":  st.sidebar.slider("Weight Title",   0.0, 1.0, 0.15, 0.05),
        "skills": st.sidebar.slider("Weight Skills",  0.0, 1.0, 0.40, 0.05),
    }
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        st.sidebar.warning("Weights must sum to 1.0")
    topk_view = st.sidebar.slider("Top-K per job", 5, 50, 10, 1)
    pre_mult  = st.sidebar.slider("Pre-select multiplier (Top-K×)", 1, 5, 2, 1)
    api_key   = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY",""))
    model_name= st.sidebar.selectbox("LLM model", ["gpt-4o-mini"], index=0)
    sleep_sec = st.sidebar.slider("Pause between LLM calls (sec.)", 0.0, 1.0, 0.20, 0.05)

    st.subheader("Загрузка")
    c1, c2 = st.columns(2)
    with c1:
        jobs_files = st.file_uploader("Jobs: Vacancy*.pdf", type=["pdf"], accept_multiple_files=True, key="jobs")
    with c2:
        resumes_files = st.file_uploader("Resumes: Resume*.pdf", type=["pdf"], accept_multiple_files=True, key="resumes")
    go = st.button("Построить ранжирование", type="primary", use_container_width=True)

    if not go:
        return
    if not jobs_files or not resumes_files:
        st.error("Загрузите хотя бы один job и одно resume.")
        return

    with st.spinner("Чтение PDF"):
        jobs_df = pd.DataFrame([{"name": f.name, "text": norm(read_pdf_text_from_bytes(f))} for f in jobs_files])
        resumes_df = pd.DataFrame([{"name": f.name, "text": norm(read_pdf_text_from_bytes(f))} for f in resumes_files])

    with st.spinner("Scoring"):
        B = compute_blocks(resumes_df, jobs_df, use_local_emb=use_local_emb)
        Score = combine_score(B, weights)

    if not api_key:
        st.error("Укажите OpenAI API key.")
        return
    client = OpenAI(api_key=api_key)

    tabs = st.tabs([n for n in jobs_df["name"].tolist()])
    for i, tab in enumerate(tabs):
        with tab:
            base_row = Score[i]
            order_base = np.argsort(-base_row)
            pre_k = min(len(order_base), max(topk_view*pre_mult, topk_view))
            pre_idx = order_base[:pre_k]
            job_text = compact_text(jobs_df.iloc[i]["text"])

            results = []
            prog = st.progress(0.0, text="LLM")
            for k, j in enumerate(pre_idx, start=1):
                res_text = compact_text(resumes_df.iloc[j]["text"])
                out = llm_verdict(client, job_text, res_text, model=model_name)
                verdict, expl = out["verdict"], out["explanation"]
                adj = adjust_score(
                    base=float(base_row[j]),
                    title_raw=float(B["TitleRaw"][i, j]),
                    skill_raw=float(B["SkillRaw"][i, j]),
                    verdict=verdict,
                )
                results.append({
                    "Resume": resumes_df.iloc[j]["name"],
                    "BaseScore": float(base_row[j]),
                    "AdjScore": adj,
                    "Verdict": verdict,
                    "Explanation": expl
                })
                prog.progress(k/pre_k)
                time.sleep(sleep_sec)
            prog.empty()

            df = pd.DataFrame(results).sort_values("AdjScore", ascending=False).reset_index(drop=True)
            df.insert(0, "Rank", range(1, len(df)+1))
            st.subheader("Ranking")
            st.dataframe(df.head(topk_view), use_container_width=True, hide_index=True)
            st.download_button(
                "Скачать CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"ranking_{jobs_df.iloc[i]['name'].replace('.pdf','')}.csv",
                mime="text/csv"
            )

# ---------- top mode switch ----------
if "mode" not in st.session_state:
    st.session_state.mode = "program"
c1, c2 = st.columns([1,1])
if c1.button("Программа", use_container_width=True):
    st.session_state.mode = "program"
if c2.button("Презентация", use_container_width=True):
    st.session_state.mode = "presentation"
st.write("")

if st.session_state.mode == "program":
    render_matcher()
else:
    render_presentation()
