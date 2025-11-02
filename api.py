import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from matcher_core import read_pdf_text, compute_signals, combine_base, llm_verdict, adjust_score

app = FastAPI(title="Resume–Job Matcher API")

DEFAULT_WEIGHTS = {"tfidf":0.25,"embed":0.20,"title":0.15,"skills":0.40}
USE_LOCAL_EMB = True

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/match")
async def match(job: UploadFile = File(...), resume: UploadFile = File(...)):
    job_text = read_pdf_text(await job.read())
    resume_text = read_pdf_text(await resume.read())
    if not job_text or not resume_text:
        raise HTTPException(status_code=400, detail="Empty text after parsing")
    signals = compute_signals(job_text, resume_text, use_local_emb=USE_LOCAL_EMB)
    base = combine_base(signals, DEFAULT_WEIGHTS)
    verdict, explanation = llm_verdict(job_text, resume_text, api_key=None)
    score = adjust_score(base, signals["title"], signals["skills"], verdict)
    return JSONResponse(content={
        "score": round(score, 6),
        "verdict": verdict,
        "explanation": explanation
    })
