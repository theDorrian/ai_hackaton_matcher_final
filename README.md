# 🧭 AI Hackathon – Resume Matching Demo

https://ai-hackaton-matcher-final.streamlit.app/

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


🌐 Public API

Наш API позволяет протестировать систему автоматического сопоставления резюме и вакансий без Streamlit-интерфейса.
Он полностью повторяет логику финальной модели (TF-IDF + эмбеддинги + GPT-оценка).

🔧 Параметры
Поле	Тип	Описание
job	PDF file	Файл с описанием вакансии
resume	PDF file	Файл с резюме кандидата

Ответ
{
  "score": 0.845132,
  "explanation": "Кандидат имеет релевантный опыт в Python и SQL, а также работал в аналогичной отрасли."
}

🧩 Пример запроса
curl -X POST https://<your-deployed-domain>/match \
  -F "job=@Vacancy1.pdf" \
  -F "resume=@Resume3.pdf"



