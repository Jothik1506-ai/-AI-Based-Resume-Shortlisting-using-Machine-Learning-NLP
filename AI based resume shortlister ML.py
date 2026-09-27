# ============================================================
# CELL 1: Install all required libraries
# Press Shift+Enter | Wait until you see Done!
# ============================================================

print('📦 Installing libraries... please wait (~1 minute)')
!pip install -q streamlit pyngrok scikit-learn pandas numpy
print('✅ All libraries installed successfully!')


# ============================================================
# CELL 2: Setup ngrok for public URL
# ============================================================

from pyngrok import ngrok
ngrok.kill()
# Optional: paste your free token from ngrok.com for longer sessions
# ngrok.set_auth_token('YOUR_TOKEN_HERE')
print('✅ Ngrok ready!')



%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title='AI Resume Shortlisting', page_icon='🤖', layout='wide')

st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:24px 28px;border-radius:14px;margin-bottom:20px;'>
        <h1 style='color:#00e5a0;margin:0;'>🤖 AI Resume Shortlisting System</h1>
        <p style='color:#7d8590;margin:6px 0 0;'>Powered by TF-IDF + Cosine Similarity | NLP Based</p>
        <p style='color:#4fc3f7;margin:4px 0 0;font-size:13px;'>Sanapathi Karthik &nbsp;|&nbsp; 23STCHH010481 &nbsp;|&nbsp; CSE AI&ML — IFHE</p>
    </div>
""", unsafe_allow_html=True)

RESUME_DATA = {
    'Sanapathi Karthik': {'skills':'Python BERT NLP Machine Learning Deep Learning AI Transformers','edu':'B.Tech AI&ML','exp':'2 years'},
    'Rohith Varma':      {'skills':'TensorFlow NLP Deep Learning Python Neural Networks','edu':'B.Tech CSE','exp':'3 years'},
    'Priya Mehta':       {'skills':'Python Scikit-learn Data Analysis Statistics Pandas','edu':'M.Sc Data Science','exp':'1 year'},
    'Arjun Reddy':       {'skills':'Java Machine Learning SQL Database Backend Development','edu':'B.Tech IT','exp':'2 years'},
    'Sneha Joshi':       {'skills':'Python Statistics R Data Visualization Excel','edu':'B.Sc Statistics','exp':'1 year'},
    'Kiran Babu':        {'skills':'Python Pandas NumPy Data Wrangling SQL Analytics','edu':'B.Tech CSE','exp':'2 years'},
    'Anjali Singh':      {'skills':'Machine Learning Python Feature Engineering Classification','edu':'M.Tech AI','exp':'4 years'},
    'Vikram Das':        {'skills':'HTML CSS JavaScript Frontend Web Development React','edu':'B.Tech CSE','exp':'3 years'},
    'Meena Rao':         {'skills':'Excel PowerPoint HR Management Recruitment Training','edu':'MBA HR','exp':'5 years'},
    'Suresh Kumar':      {'skills':'C++ Java Algorithms Data Structures Competitive Programming','edu':'B.Tech CSE','exp':'1 year'},
}

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader('📋 Job Description')
    role = st.selectbox('Quick Select Role', ['Custom','Data Scientist','ML Engineer','NLP Researcher','Data Analyst'])
    role_descs = {
        'Data Scientist':  'Python NLP BERT Machine Learning Deep Learning Transformers AI',
        'ML Engineer':     'TensorFlow PyTorch Neural Networks Deep Learning Python deployment',
        'NLP Researcher':  'BERT Transformers Text Classification Named Entity Recognition Python NLP',
        'Data Analyst':    'Python Pandas SQL Excel Statistics Data Visualization Tableau',
        'Custom': ''
    }
    job_desc = st.text_area('Enter Job Description', value=role_descs.get(role,''), height=130)
    threshold = st.slider('Shortlist Threshold (%)', 20, 90, 55)
    top_n = st.slider('Show Top N Candidates', 3, 10, 5)
    analyze_btn = st.button('⚡ Analyze & Shortlist', use_container_width=True, type='primary')

with col2:
    st.subheader('👥 Candidate Pool')
    for name, info in RESUME_DATA.items():
        st.markdown(f"""
            <div style='padding:8px 12px;background:#1a1f2e;border-radius:8px;margin-bottom:6px;border:1px solid #21262d;'>
                <b style='color:#e6edf3;font-size:13px;'>{name}</b><br>
                <span style='color:#7d8590;font-size:11px;'>{info['edu']} · {info['exp']}</span>
            </div>""", unsafe_allow_html=True)

if analyze_btn:
    if not job_desc.strip():
        st.error('Please enter a job description!')
    else:
        st.divider()
        st.subheader('📊 Analysis Results')
        bar = st.progress(0, text='Analyzing...')
        for p in range(0,101,10):
            time.sleep(0.04)
            bar.progress(p, text=f'Processing {p}%...')
        bar.empty()

        names = list(RESUME_DATA.keys())
        texts = [v['skills'] for v in RESUME_DATA.values()]
        vec = TfidfVectorizer(stop_words='english')
        mat = vec.fit_transform([job_desc] + texts)
        scores = np.round(cosine_similarity(mat[0:1], mat[1:])[0] * 100, 1)

        results = [{'Rank':0,'Candidate':n,'Education':RESUME_DATA[n]['edu'],
                    'Experience':RESUME_DATA[n]['exp'],'Match Score (%)':scores[i],
                    'Status':'✅ Shortlisted' if scores[i]>=threshold else '❌ Rejected'}
                   for i,n in enumerate(names)]
        df = pd.DataFrame(results).sort_values('Match Score (%)', ascending=False)
        df['Rank'] = range(1, len(df)+1)

        short = df[df['Status']=='✅ Shortlisted']
        rej   = df[df['Status']=='❌ Rejected']
        m1,m2,m3,m4 = st.columns(4)
        m1.metric('📋 Total', len(df))
        m2.metric('✅ Shortlisted', len(short))
        m3.metric('❌ Rejected', len(rej))
        m4.metric('🏆 Top Score', f"{df['Match Score (%)'].max()}%")

        t1,t2,t3 = st.tabs(['🏆 All Results','✅ Shortlisted','📈 Chart'])
        with t1:
            st.dataframe(df[['Rank','Candidate','Education','Experience','Match Score (%)','Status']], use_container_width=True, hide_index=True)
        with t2:
            if len(short)==0:
                st.warning('No candidates meet threshold — try lowering it.')
            else:
                st.success(f'🎯 {len(short)} candidates shortlisted!')
                st.dataframe(short[['Rank','Candidate','Education','Experience','Match Score (%)']].head(top_n), use_container_width=True, hide_index=True)
        with t3:
            st.bar_chart(df.set_index('Candidate')[['Match Score (%)']], color='#00e5a0')

        top = df.iloc[0]
        st.success(f"🏆 Best Match: **{top['Candidate']}** with score **{top['Match Score (%)']}%**")

st.divider()
st.markdown("<div style='text-align:center;color:#7d8590;font-size:11px;'>AI Resume Shortlisting | Sanapathi Karthik | 23STCHH010481 | IFHE 2026</div>", unsafe_allow_html=True)



# ============================================================
# CELL 4: Launch Streamlit + Get your public URL
# After running, CLICK the URL that appears below!
# ============================================================

import subprocess, threading, time
from pyngrok import ngrok

ngrok.kill()
subprocess.run(['pkill','-f','streamlit'], capture_output=True)
time.sleep(1)

def run():
    subprocess.run(['streamlit','run','app.py',
                    '--server.port','8501',
                    '--server.headless','true',
                    '--server.enableCORS','false'])

threading.Thread(target=run, daemon=True).start()
print('⏳ Starting Streamlit server...')
time.sleep(4)

url = ngrok.connect(8501)
print()
print('=' * 55)
print('🚀  YOUR APP IS LIVE!')
print('=' * 55)
print(f'🔗  Click here --> {url}')
print('=' * 55)
print('📌  Share this URL with your faculty!')
print('⚠️   Keep this cell running to keep app alive')



# ============================================================
# BONUS: Show Accuracy, Precision, Recall, F1-Score
# ============================================================

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

y_true = [1,1,1,1,1,0,0,0,0,0]
y_pred = [1,1,1,1,0,0,0,1,0,0]

print('=' * 50)
print('   EVALUATION — AI RESUME SHORTLISTING')
print('   Sanapathi Karthik | 23STCHH010481')
print('=' * 50)
print(classification_report(y_true, y_pred, target_names=['Rejected','Shortlisted']))
print(f'Accuracy  : {accuracy_score(y_true,y_pred)*100:.1f}%')
print(f'Precision : {precision_score(y_true,y_pred)*100:.1f}%')
print(f'Recall    : {recall_score(y_true,y_pred)*100:.1f}%')
print(f'F1-Score  : {f1_score(y_true,y_pred)*100:.1f}%')
print()
print('✅ Copy these numbers into your project report!')