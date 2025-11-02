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

# ===== Optional: local embeddings
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# ===== Optional: OpenAI explanations
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


# =========================
#     UI CONFIG & THEME
# =========================
st.set_page_config(
    page_title="AI Hackathon — Resume Matcher",
    page_icon="🧭",
    layout="wide",
)

PRIMARY = "#6C5CE7"   # фиолетовый акцент
ACCENT  = "#00C2FF"

st.markdown(
    f"""
    <style>
    .main .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}
    .pill {{
        display:inline-block; padding:2px 8px; border-radius:999px; 
        background:{PRIMARY}20; color:{PRIMARY}; font-weight:600; font-size:12px;
        border:1px solid {PRIMARY}40; margin-right:6px;
    }}
    .rank-badge {{
        display:inline-flex; align-items:center; justify-content:center;
        width:28px; height:28px; border-radius:50%;
        background:{ACCENT}20; color:#222; font-weight:700; border:1px solid {ACCENT}60;
    }}
    .score-badge {{
        display:inline-block; padding:2px 8px; border-radius:8px; 
        background:#e8f9f1; color:#0b6; font-weight:700; border:1px solid #b9f0d8;
    }}
    .subtle {{
        color:#666; font-size:12px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧭 Automated Resume Matching (AI Hackathon — Stream 1)")
st.caption("Загрузите PDF-вакансии и PDF-резюме. Для каждой вакансии построим ранжирование резюме — ранг 1 = наилучшее соответствие. Опционально: GPT-объяснения (gpt-4o-mini).")


# =========================
#         UTILS
# =========================
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def read_pdf_text_from_bytes(file) -> str:
    """
    Достаёт текст из загруженного PDF (BytesIO).
    Сначала PyPDF2, затем pdfminer fallback.
    """
    # PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        txt_parts = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            txt_parts.append(t)
        txt = "\n".join(txt_parts).strip()
        if txt:
            return txt
    except Exception:
        pass

    # pdfminer fallback
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
                   emb_model_name: str = "intfloat/multilingual-e5-base"
                   ) -> Dict[str, np.ndarray]:
    """
    Возвращает словарь:
      S_tfidf, S_emb, S_title, S_skill,
      job_titles, res_titles (первые строки)
    """

    R_texts = resumes_df["text"].tolist()
    J_texts = jobs_df["text"].tolist()

    # ---- TF-IDF
    tfidf = TfidfVectorizer(max_features=45000, ngram_range=(1, 2))
    X_res = tfidf.fit_transform(R_texts)
    X_job = tfidf.transform(J_texts)
    C_tfidf = cosine_similarity(X_job, X_res)

    # ---- Local Embeddings (optional)
    if use_local_emb and _HAS_ST:
        model = SentenceTransformer(emb_model_name)

        def _prep(texts, is_query=False):
            # E5 требует query:/passage: префиксы
            if "e5" in (emb_model_name or "").lower():
                return [("query: " if is_query else "passage: ") + (t or "") for t in texts]
            return texts

        E_res = model.encode(_prep(R_texts, is_query=False), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        E_job = model.encode(_prep(J_texts, is_query=True),  batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        C_emb = np.asarray(E_job) @ np.asarray(E_res).T
    else:
        C_emb = np.zeros_like(C_tfidf)

    # ---- Title similarity (грубая эвристика: первая строка)
    job_titles = [norm((t.split("\n")[0] if t else "")).lower() for t in J_texts]
    res_titles = [norm((t.split("\n")[0] if t else "")).lower() for t in R_texts]
    TitleSim = np.zeros_like(C_tfidf, dtype=np.float32)
    for i in range(TitleSim.shape[0]):
        jt = job_titles[i]
        for j in range(TitleSim.shape[1]):
            TitleSim[i, j] = fuzz.token_set_ratio(jt, res_titles[j]) / 100.0

    # ---- Skills Jaccard (эвристика по токенам)
    skill_pat = re.compile(r"\b[A-Za-z][A-Za-z0-9\.\+\#\-]{1,}\b")
    def skill_set(text: str) -> set:
        toks = [t.lower() for t in skill_pat.findall(text)]
        return set(toks)

    job_skill_sets = [skill_set(t) for t in J_texts]
    res_skill_sets = [skill_set(t) for t in R_texts]
    SkillJac = np.zeros_like(C_tfidf, dtype=np.float32)
    for i in range(SkillJac.shape[0]):
        js = job_skill_sets[i]
        for j in range(SkillJac.shape[1]):
            rs = res_skill_sets[j]
            u = len(js | rs)
            SkillJac[i, j] = (len(js & rs) / u) if u else 0.0

    # ---- scale to [0..1]
    S_tfidf = minmax01(C_tfidf)
    S_emb   = minmax01(C_emb) if use_local_emb and _HAS_ST else C_emb
    S_title = minmax01(TitleSim)
    S_skill = minmax01(SkillJac)

    return dict(
        S_tfidf=S_tfidf, S_emb=S_emb, S_title=S_title, S_skill=S_skill,
        job_titles=job_titles, res_titles=res_titles
    )


def combine_score(blocks: Dict[str, np.ndarray], w: Dict[str, float]) -> np.ndarray:
    return (w["tfidf"] * blocks["S_tfidf"]
          + w["embed"] * blocks["S_emb"]
          + w["title"] * blocks["S_title"]
          + w["skills"] * blocks["S_skill"])


def make_ranking_dataframe(Score: np.ndarray,
                           resumes_df: pd.DataFrame,
                           jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает длинную таблицу со всеми парами job×resume:
      job_idx, job_name, resume_idx, resume_name, score, rank (1=лучший)
    """
    rows = []
    for i in range(Score.shape[0]):  # по вакансиям
        row = Score[i]
        order = np.argsort(-row)  # индексы резюме по убыванию
        for rank, j in enumerate(order, start=1):
            rows.append({
                "job_idx": i,
                "job_name": jobs_df.iloc[i]["name"],
                "resume_idx": j,
                "resume_name": resumes_df.iloc[j]["name"],
                "score": float(row[j]),
                "rank": rank,
            })
    return pd.DataFrame(rows)


def compact_text(text: str, head=800) -> str:
    txt = norm(text or "")
    return txt[:head]


def explain_pair(client: "OpenAI", job_text: str, res_text: str, model="gpt-4o-mini", temperature=0.4) -> str:
    system = (
        "Ты HR-аналитик. На вход подаются текст вакансии и текст резюме. "
        "Дай короткое (2–3 предложения) человеческое объяснение на русском, почему кандидат подходит или нет. "
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


# =========================
#     SIDEBAR SETTINGS
# =========================
st.sidebar.header("⚙️ Настройки")

use_local_emb = st.sidebar.checkbox("Добавить локальные эмбеддинги (E5)", value=_HAS_ST,
                                    help="Повышает качество семантики. Если SentenceTransformers не установлен — опция отключена.")

weights = {
    "tfidf": st.sidebar.slider("Вес: TF-IDF", 0.0, 1.0, 0.25, 0.05),
    "embed": st.sidebar.slider("Вес: Embedding", 0.0, 1.0, 0.45, 0.05),
    "title": st.sidebar.slider("Вес: Title", 0.0, 1.0, 0.15, 0.05),
    "skills": st.sidebar.slider("Вес: Skills", 0.0, 1.0, 0.15, 0.05),
}
if abs(sum(weights.values()) - 1.0) > 1e-6:
    st.sidebar.warning("Сумма весов должна быть равна 1.0")

topk_view = st.sidebar.slider("Сколько показывать в таблице (Top-K)", 5, 50, 10, 1)

st.sidebar.markdown("---")
OPENAI_KEY = st.sidebar.text_input("🔑 OpenAI API Key (для объяснений)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
model_name = st.sidebar.selectbox("Модель для объяснений", ["gpt-4o-mini"], index=0)
sleep_sec = st.sidebar.slider("Пауза между LLM-вызовами (сек.)", 0.0, 1.0, 0.2, 0.1)


# =========================
#        UPLOAD FORMS
# =========================
st.subheader("📥 Загрузка данных")

col_jobs, col_res = st.columns(2)

with col_jobs:
    st.markdown("#### Вакансии (PDF)")
    jobs_files = st.file_uploader("Загрузите Vacancy*.pdf", type=["pdf"], accept_multiple_files=True, key="jobs")
    st.caption("Можно перетащить несколько файлов сразу.")

with col_res:
    st.markdown("#### Резюме (PDF)")
    resumes_files = st.file_uploader("Загрузите Resume*.pdf", type=["pdf"], accept_multiple_files=True, key="resumes")
    st.caption("Файлы вакансий и резюме загружаются отдельно.")

process = st.button("🚀 Построить ранжирование", type="primary", use_container_width=True)


# =========================
#          PIPELINE
# =========================
if process:
    if not jobs_files or not resumes_files:
        st.error("Загрузите хотя бы один PDF с вакансией и один PDF с резюме.")
        st.stop()

    # --- Read PDFs
    with st.spinner("Чтение PDF и извлечение текста..."):
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

    # --- Compute similarities
    with st.spinner("Вычисляем сходство (TF-IDF / Embeddings / Title / Skills)..."):
        blocks = compute_blocks(resumes_df, jobs_df, use_local_emb=use_local_emb)
        Score = combine_score(blocks, weights)

    st.success("Готово! Ниже — ранжирование для каждой вакансии.")

    # =========================
    #        PRESENTATION
    # =========================
    tabs = st.tabs([f"📄 {n}" for n in jobs_df["name"].tolist()])

    for i, tab in enumerate(tabs):
        with tab:
            st.markdown(f"**Вакансия:** `{jobs_df.iloc[i]['name']}`")
            st.markdown(
                f"<span class='pill'>TF-IDF</span><span class='pill'>Embeddings</span>"
                f"<span class='pill'>Title</span><span class='pill'>Skills</span>",
                unsafe_allow_html=True
            )
            st.write(" ")

            row = Score[i]
            order = np.argsort(-row)
            top_idx = order[:topk_view]

            # ===== Hero: Top-3 cards
            st.markdown("##### 🏆 Топ-3 кандидата")
            c1, c2, c3 = st.columns(3)
            cards_cols = [c1, c2, c3]
            for k, j in enumerate(top_idx[:3]):
                with cards_cols[k]:
                    st.markdown(
                        f"<div class='rank-badge'>{k+1}</div> "
                        f"**{resumes_df.iloc[j]['name']}**  "
                        f"<span class='score-badge'>{row[j]:.4f}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(resumes_df.iloc[j]["text"][:140] + ("…" if len(resumes_df.iloc[j]["text"]) > 140 else ""))

            st.write(" ")

            # ===== Full table (Top-K)
            records = []
            for rank, j in enumerate(top_idx, start=1):
                records.append({
                    "Rank": rank,
                    "Score": float(row[j]),
                    "Resume": resumes_df.iloc[j]["name"],
                    "Preview": resumes_df.iloc[j]["text"][:160] + ("…" if len(resumes_df.iloc[j]["text"])>160 else "")
                })
            st.markdown("#### 📋 Таблица ранжирования (Top-K)")
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

            # ===== Export per-job CSV
            job_df = pd.DataFrame({
                "rank": list(range(1, len(order)+1)),
                "score": [float(row[j]) for j in order],
                "resume": [resumes_df.iloc[j]["name"] for j in order]
            })
            st.download_button(
                "⬇️ Скачать CSV для этой вакансии",
                job_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ranking_{jobs_df.iloc[i]['name'].replace('.pdf','')}.csv",
                mime="text/csv"
            )

            st.markdown("---")

            # ===== Explanations (on-demand)
            st.markdown("#### 🧠 Объяснения (gpt-4o-mini, опционально)")
            colL, colR = st.columns([1,2])
            with colL:
                want_expl_topk = st.checkbox("Сгенерировать для Top-K", value=False, key=f"expl_topk_{i}")
                want_expl_all  = st.checkbox("Сгенерировать для всех резюме", value=False, key=f"expl_all_{i}")
            with colR:
                st.caption("Включите один из чекбоксов и нажмите кнопку ниже. Понадобится OpenAI API Key.")

            go_expl = st.button("Запустить генерацию объяснений", key=f"go_expl_{i}")

            if go_expl:
                if not _HAS_OPENAI:
                    st.error("Пакет openai не установлен. Установите `pip install openai`.")
                elif not OPENAI_KEY:
                    st.error("Укажите OpenAI API Key в сайдбаре.")
                else:
                    client = OpenAI(api_key=OPENAI_KEY)
                    job_text = compact_text(jobs_df.iloc[i]["text"])
                    idx_list = top_idx if want_expl_topk else order  # top-K или все

                    rows_out = []
                    prog = st.progress(0.0, text="Генерация объяснений…")
                    for k, j in enumerate(idx_list, start=1):
                        res_text = compact_text(resumes_df.iloc[j]["text"])
                        expl = explain_pair(client, job_text, res_text, model=model_name)
                        rows_out.append({
                            "rank": int(np.where(order == j)[0][0]) + 1,
                            "score": float(row[j]),
                            "resume": resumes_df.iloc[j]["name"],
                            "explanation": expl
                        })
                        prog.progress(k/len(idx_list))
                        time.sleep(sleep_sec)
                    prog.empty()

                    df_expl = pd.DataFrame(rows_out).sort_values(["rank"]).reset_index(drop=True)
                    st.success("Готово!")
                    st.dataframe(df_expl, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Скачать объяснения (CSV)",
                        df_expl.to_csv(index=False).encode("utf-8"),
                        file_name=f"explanations_{jobs_df.iloc[i]['name'].replace('.pdf','')}.csv",
                        mime="text/csv"
                    )

    # ===== Combined CSV for all jobs (ranks for every resume)
    st.markdown("### 📦 Экспорт полного ранжирования для всех вакансий")
    full_rank_df = make_ranking_dataframe(Score, resumes_df.assign(name=resumes_df["name"]), jobs_df.assign(name=jobs_df["name"]))
    st.dataframe(full_rank_df.head(50), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Скачать полное ранжирование (CSV)",
        full_rank_df.to_csv(index=False).encode("utf-8"),
        "all_jobs_ranking.csv",
        "text/csv"
    )


# =========================
#         FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<span class='subtle'>© AI Hackathon — Code. Create. Conquer. | Demo: TF-IDF + (optional) local embeddings + GPT-4o-mini explanations</span>",
    unsafe_allow_html=True
)
