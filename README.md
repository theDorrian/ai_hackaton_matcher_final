# 🧭 AI Hackathon – Resume Matching Demo

### Stream 1: Automated Resume Matching System

Этот проект был разработан для **AI Hackathon: Code. Create. Conquer.**  
Задача — построить систему, автоматически сопоставляющую **резюме кандидатов** с **вакансиями**.

---

## 🚀 Запуск локально

```bash
git clone https://github.com/<your_team>/ai_hack_matcher
cd ai_hack_matcher
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env .env
# вставь свой OpenAI API ключ
streamlit run app.py
