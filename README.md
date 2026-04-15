# 🤖 AI Resume Shortlisting System

> **Author:** Sanapathi Karthik | 23STCHH010481 | CSE AI&ML — IFHE 2026  
> **Tech:** Python · Streamlit · TF-IDF · Cosine Similarity · Scikit-learn

---

## 🚀 Features
- Upload `.txt` resume files for real candidates
- Select from preset job roles or write a custom job description
- TF-IDF + Cosine Similarity scoring for all candidates
- Adjustable shortlist threshold and top-N display
- Tabs: All Results · Shortlisted · Score Chart · ML Evaluation Metrics
- Download results as CSV
- ML evaluation: Accuracy, Precision, Recall, F1-Score

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Go to → [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo
4. Set **Main file path** → `app.py`
5. Click **Deploy** — your public URL is ready in ~1 minute!

## 📁 Project Structure

```
AI resume shortlisting/
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml         ← Theme & server config
└── README.md
```
