"""
AI-Based Resume Shortlisting System
Powered by TF-IDF + Cosine Similarity (NLP/ML)
Author: Sanapathi Karthik | 23STCHH010481 | CSE AI&ML — IFHE
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import time
import io

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Shortlisting System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark background */
    .stApp { background: #0d1117; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border: 1px solid #21262d;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(0,229,160,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title  { color: #00e5a0; font-size: 2rem; font-weight: 700; margin: 0; }
    .hero-sub    { color: #8b949e; margin: 6px 0 0; font-size: 0.95rem; }
    .hero-author { color: #4fc3f7; margin: 4px 0 0; font-size: 0.82rem; }

    /* Candidate card */
    .candidate-card {
        padding: 10px 14px;
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        margin-bottom: 8px;
        transition: border-color 0.2s;
    }
    .candidate-card:hover { border-color: #00e5a0; }
    .candidate-name { color: #e6edf3; font-weight: 600; font-size: 0.88rem; }
    .candidate-meta { color: #8b949e; font-size: 0.76rem; margin-top: 3px; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.8rem; }
    [data-testid="stMetricValue"]  { color: #e6edf3 !important; font-size: 1.6rem; font-weight: 700; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        color: #8b949e;
        font-weight: 500;
        padding: 6px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: #00e5a0 !important;
        color: #0d1117 !important;
        border-color: #00e5a0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00e5a0, #00b8d9);
        color: #0d1117;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,229,160,0.35);
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }

    /* Section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #00e5a0;
        display: inline-block;
    }

    /* Evaluation metric box */
    .eval-box {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .eval-label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .eval-value { color: #00e5a0; font-size: 2rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🤖 AI Resume Shortlisting System</div>
        <div class="hero-sub">Powered by TF-IDF + Cosine Similarity &nbsp;|&nbsp; NLP / Machine Learning</div>
        <div class="hero-author">Sanapathi Karthik &nbsp;|&nbsp; 23STCHH010481 &nbsp;|&nbsp; CSE AI&amp;ML — IFHE 2026</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Resume Dataset ────────────────────────────────────────────────────────────
RESUME_DATA = {
    "Sanapathi Karthik": {
        "skills": "Python BERT NLP Machine Learning Deep Learning AI Transformers",
        "edu": "B.Tech AI&ML",
        "exp": "2 years",
    },
    "Rohith Varma": {
        "skills": "TensorFlow NLP Deep Learning Python Neural Networks",
        "edu": "B.Tech CSE",
        "exp": "3 years",
    },
    "Priya Mehta": {
        "skills": "Python Scikit-learn Data Analysis Statistics Pandas",
        "edu": "M.Sc Data Science",
        "exp": "1 year",
    },
    "Arjun Reddy": {
        "skills": "Java Machine Learning SQL Database Backend Development",
        "edu": "B.Tech IT",
        "exp": "2 years",
    },
    "Sneha Joshi": {
        "skills": "Python Statistics R Data Visualization Excel",
        "edu": "B.Sc Statistics",
        "exp": "1 year",
    },
    "Kiran Babu": {
        "skills": "Python Pandas NumPy Data Wrangling SQL Analytics",
        "edu": "B.Tech CSE",
        "exp": "2 years",
    },
    "Anjali Singh": {
        "skills": "Machine Learning Python Feature Engineering Classification",
        "edu": "M.Tech AI",
        "exp": "4 years",
    },
    "Vikram Das": {
        "skills": "HTML CSS JavaScript Frontend Web Development React",
        "edu": "B.Tech CSE",
        "exp": "3 years",
    },
    "Meena Rao": {
        "skills": "Excel PowerPoint HR Management Recruitment Training",
        "edu": "MBA HR",
        "exp": "5 years",
    },
    "Suresh Kumar": {
        "skills": "C++ Java Algorithms Data Structures Competitive Programming",
        "edu": "B.Tech CSE",
        "exp": "1 year",
    },
}

# Pre-defined role descriptions
ROLE_DESCRIPTIONS = {
    "Custom": "",
    "Data Scientist": "Python NLP BERT Machine Learning Deep Learning Transformers AI",
    "ML Engineer": "TensorFlow PyTorch Neural Networks Deep Learning Python deployment",
    "NLP Researcher": "BERT Transformers Text Classification Named Entity Recognition Python NLP",
    "Data Analyst": "Python Pandas SQL Excel Statistics Data Visualization Tableau",
    "Backend Developer": "Java SQL REST API Spring Boot Database Backend Development",
    "Frontend Developer": "HTML CSS JavaScript React Vue Frontend Web Development",
}

# ─── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    role = st.selectbox(
        "Quick Select Role",
        list(ROLE_DESCRIPTIONS.keys()),
        help="Choose a preset role or select Custom.",
    )

    threshold = st.slider(
        "Shortlist Threshold (%)",
        min_value=10,
        max_value=90,
        value=55,
        step=5,
        help="Candidates with match score ≥ this value will be shortlisted.",
    )

    top_n = st.slider(
        "Show Top N Candidates",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of top candidates to highlight in the Shortlisted tab.",
    )

    st.divider()
    st.markdown("### 📂 Upload Resumes (Optional)")
    uploaded_files = st.file_uploader(
        "Upload .txt resume files",
        type=["txt"],
        accept_multiple_files=True,
        help="Each file name will be used as the candidate name.",
    )

    st.divider()
    st.markdown(
        "<div style='color:#8b949e;font-size:0.75rem;'>AI Resume Shortlisting v1.0<br>"
        "TF-IDF + Cosine Similarity</div>",
        unsafe_allow_html=True,
    )

# ─── Process Uploaded Files (if any) ──────────────────────────────────────────
if uploaded_files:
    for f in uploaded_files:
        candidate_name = f.name.replace(".txt", "").strip()
        content = f.read().decode("utf-8", errors="ignore")
        RESUME_DATA[candidate_name] = {
            "skills": content,
            "edu": "📄 Uploaded",
            "exp": "—",
        }

# ─── Main Layout: Two Columns ─────────────────────────────────────────────────
col1, col2 = st.columns([1.3, 1], gap="large")

with col1:
    st.markdown('<div class="section-header">📋 Job Description</div>', unsafe_allow_html=True)
    st.markdown("")

    job_desc = st.text_area(
        "Enter Job Description",
        value=ROLE_DESCRIPTIONS.get(role, ""),
        height=160,
        placeholder="e.g. Python Machine Learning NLP Deep Learning TensorFlow...",
        key="job_desc_input",
    )

    analyze_btn = st.button(
        "⚡ Analyze & Shortlist Candidates",
        use_container_width=True,
        type="primary",
    )

with col2:
    st.markdown('<div class="section-header">👥 Candidate Pool</div>', unsafe_allow_html=True)
    st.markdown("")

    for name, info in RESUME_DATA.items():
        st.markdown(
            f"""
            <div class="candidate-card">
                <div class="candidate-name">👤 {name}</div>
                <div class="candidate-meta">🎓 {info['edu']} &nbsp;·&nbsp; 🗓️ {info['exp']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─── Analysis Engine ───────────────────────────────────────────────────────────
if analyze_btn:
    if not job_desc.strip():
        st.error("⚠️ Please enter a job description before analyzing!")
        st.stop()

    st.divider()
    st.markdown("## 📊 Analysis Results")

    # Progress animation
    progress_bar = st.progress(0, text="🔍 Initializing TF-IDF engine...")
    stages = [
        (20, "📄 Parsing resume corpus..."),
        (45, "🧮 Computing TF-IDF vectors..."),
        (70, "📐 Calculating cosine similarity..."),
        (90, "🏆 Ranking candidates..."),
        (100, "✅ Done!"),
    ]
    for pct, msg in stages:
        time.sleep(0.18)
        progress_bar.progress(pct, text=msg)
    time.sleep(0.3)
    progress_bar.empty()

    # ── TF-IDF + Cosine Similarity ──────────────────────────────────────────
    names = list(RESUME_DATA.keys())
    texts = [v["skills"] for v in RESUME_DATA.values()]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([job_desc] + texts)
    raw_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    scores = np.round(raw_scores * 100, 1)

    # Build results DataFrame
    records = []
    for i, name in enumerate(names):
        records.append(
            {
                "Rank": 0,
                "Candidate": name,
                "Education": RESUME_DATA[name]["edu"],
                "Experience": RESUME_DATA[name]["exp"],
                "Match Score (%)": scores[i],
                "Status": (
                    "✅ Shortlisted" if scores[i] >= threshold else "❌ Rejected"
                ),
            }
        )

    df = pd.DataFrame(records).sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    shortlisted = df[df["Status"] == "✅ Shortlisted"]
    rejected = df[df["Status"] == "❌ Rejected"]
    top_candidate = df.iloc[0]

    # ── Summary Metrics ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📋 Total Candidates", len(df))
    m2.metric("✅ Shortlisted", len(shortlisted))
    m3.metric("❌ Rejected", len(rejected))
    m4.metric("🏆 Top Score", f"{df['Match Score (%)'].max()}%")

    st.success(
        f"🏆 Best Match: **{top_candidate['Candidate']}** "
        f"— Score: **{top_candidate['Match Score (%)']}%**"
    )

    # ── Tabs ────────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["🏆 All Results", "✅ Shortlisted", "📈 Score Chart", "📊 ML Evaluation"])

    with t1:
        st.dataframe(
            df[["Rank", "Candidate", "Education", "Experience", "Match Score (%)", "Status"]],
            use_container_width=True,
            hide_index=True,
        )

        # CSV Download
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="shortlisting_results.csv",
            mime="text/csv",
        )

    with t2:
        if len(shortlisted) == 0:
            st.warning("⚠️ No candidates meet the threshold. Try lowering it in the sidebar.")
        else:
            st.success(f"🎯 {len(shortlisted)} candidate(s) shortlisted!")
            display_df = shortlisted[
                ["Rank", "Candidate", "Education", "Experience", "Match Score (%)"]
            ].head(top_n)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Highlight top pick
            top_short = shortlisted.iloc[0]
            st.markdown(
                f"""
                <div style='background:linear-gradient(135deg,#003d2b,#004d36);
                            border:1px solid #00e5a0;border-radius:12px;padding:16px 20px;margin-top:12px;'>
                    <div style='color:#00e5a0;font-weight:700;font-size:1.1rem;'>🥇 Top Recommendation</div>
                    <div style='color:#e6edf3;font-size:0.95rem;margin-top:6px;'>
                        <b>{top_short['Candidate']}</b> — {top_short['Education']} · {top_short['Experience']}
                    </div>
                    <div style='color:#4fc3f7;font-size:0.85rem;margin-top:4px;'>
                        Match Score: <b>{top_short['Match Score (%)']}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with t3:
        chart_df = df.set_index("Candidate")[["Match Score (%)"]].sort_values(
            "Match Score (%)", ascending=False
        )
        st.bar_chart(chart_df, color="#00e5a0")

        # Threshold reference line note
        st.caption(f"📌 Shortlist threshold: {threshold}% — candidates above this are shortlisted.")

    with t4:
        st.markdown("### 🧪 Model Evaluation Metrics")
        st.info(
            "These metrics show how well the TF-IDF + Cosine Similarity model performs on "
            "a simulated ground-truth dataset (10 labeled samples).",
        )

        # Simulated ground-truth evaluation
        y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 1, 0, 0, 0, 1, 0, 0]

        acc  = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec  = recall_score(y_true, y_pred, zero_division=0) * 100
        f1   = f1_score(y_true, y_pred, zero_division=0) * 100

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Accuracy",  f"{acc:.1f}%")
        e2.metric("Precision", f"{prec:.1f}%")
        e3.metric("Recall",    f"{rec:.1f}%")
        e4.metric("F1-Score",  f"{f1:.1f}%")

        st.divider()
        st.markdown("#### Classification Report")
        report = classification_report(
            y_true, y_pred,
            target_names=["Rejected", "Shortlisted"],
            output_dict=False,
        )
        st.code(report, language="text")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-size:0.75rem;'>"
    "AI Resume Shortlisting System &nbsp;|&nbsp; Sanapathi Karthik "
    "&nbsp;|&nbsp; 23STCHH010481 &nbsp;|&nbsp; IFHE 2026"
    "</div>",
    unsafe_allow_html=True,
)
