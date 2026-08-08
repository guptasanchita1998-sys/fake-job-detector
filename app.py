import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page config (must be first Streamlit command)
st.set_page_config(page_title="Fake Job Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# ---------------- Load model & data (auto-train if model files missing) ----------------
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('fake_job_postings.csv')

@st.cache_resource
def load_or_train_model():
    if os.path.exists('fake_job_model.pkl') and os.path.exists('vectorizer.pkl'):
        model = joblib.load('fake_job_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer

    # Train from scratch (runs once, then cached for the session)
    data = df.copy()
    data['description'] = data['description'].fillna('')
    data['title'] = data['title'].fillna('')
    data['text'] = data['title'] + ' ' + data['description']

    X = data['text']
    y = data['fraudulent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_vec, y_train)

    joblib.dump(model, 'fake_job_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    return model, vectorizer

with st.spinner("Setting up the model (first run only)..."):
    model, vectorizer = load_or_train_model()

if 'history' not in st.session_state:
    st.session_state.history = []

# ---------------- Custom CSS (dashboard look) ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [class*="st-"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Don't override Streamlit's icon font - fixes icons showing as literal text */
    [data-testid="stIconMaterial"], span[class*="material"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* Force light backgrounds everywhere, override dark mode */
    .stApp, .main, [data-testid="stAppViewContainer"], body {
        background-color: #F5F6FB !important;
    }
    [data-testid="stAppViewContainer"] * {
        color: #111827;
    }
    p, span, label, .stMarkdown, .stCaption, h1, h2, h3, h4, h5 {
        color: #111827 !important;
    }

    /* Shrink the header bar (keeps sidebar collapse arrow, removes excess space) */
    header[data-testid="stHeader"] {
        height: 1.6rem !important;
        min-height: 1.6rem !important;
    }
    header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"],
    header[data-testid="stHeader"] [data-testid="stCollapseSidebarButton"] {
        margin-top: -0.3rem !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #EEF0F5;
        height: 100vh;
        overflow: hidden;
    }
    section[data-testid="stSidebar"] * {
        color: #111827 !important;
    }
    section[data-testid="stSidebar"] > div {
        height: 100%;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        padding-top: 0.2rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 0.4rem !important;
        overflow: hidden;
        height: 100%;
        min-height: 85vh;
    }
    section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sidebar-bottom-anchor) {
        margin-top: auto !important;
    }
    section[data-testid="stSidebar"] *:has(> .sidebar-bottom-anchor) {
        margin-top: auto !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        margin-top: 0 !important;
    }
    section[data-testid="stSidebar"] label {
        color: #374151 !important;
        font-weight: 500;
        font-size: 15px !important;
    }

    /* Sidebar nav items styled like a menu list with active highlight */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 10px 14px;
        border-radius: 10px;
        margin-bottom: 2px;
        transition: background 0.15s;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: #EEF0FE;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #4F46E5 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    /* Text areas / inputs force light */
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #E5E7EB !important;
        font-size: 15px !important;
    }
    textarea::placeholder, input::placeholder {
        color: #9CA3AF !important;
    }

    div[data-baseweb="notification"] {
        background-color: #EAF0FE !important;
    }
    div[data-baseweb="notification"] * {
        color: #1E3A8A !important;
    }

    /* Stat cards */
    .stat-card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 22px 24px;
        border: 1px solid #EEF0F5;
        box-shadow: 0 1px 3px rgba(16,24,40,0.05);
        height: 100%;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .stat-icon {
        width: 48px; height: 48px;
        min-width: 48px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 0;
    }
    .stat-text { display: flex; flex-direction: column; }
    .stat-label { color: #6B7280; font-size: 14px; font-weight: 500; margin-bottom: 2px; }
    .stat-value { color: #111827; font-size: 28px; font-weight: 800; line-height: 1.2; margin: 0; }
    .stat-sub { color: #9CA3AF; font-size: 12px; margin-top: 2px; }

    .icon-green { background: #E7F7EF; color: #16A34A; }
    .icon-red { background: #FDEBEC; color: #E11D48; }
    .icon-amber { background: #FEF6E7; color: #D97706; }
    .icon-blue { background: #EAF0FE; color: #4F46E5; }

    .card-box {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 30px;
        border: 1px solid #EEF0F5;
        box-shadow: 0 1px 3px rgba(16,24,40,0.05);
    }

    .card-title { font-size: 20px; font-weight: 700; color: #111827; margin-bottom: 4px; }
    .card-desc { color: #6B7280; font-size: 14px; margin-bottom: 18px; }

    .badge-fake {
        background: #FDEBEC; color: #E11D48;
        padding: 5px 14px; border-radius: 8px; font-size: 13px; font-weight: 600;
        display: inline-block;
    }
    .badge-safe {
        background: #E7F7EF; color: #16A34A;
        padding: 5px 14px; border-radius: 8px; font-size: 13px; font-weight: 600;
        display: inline-block;
    }

    h1 { font-weight: 800 !important; color: #111827; font-size: 40px !important; }
    .dash-subtitle { color: #6B7280; font-size: 16px; margin-top: -8px; margin-bottom: 8px; }

    div.stButton > button {
        background: #4F46E5;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.65rem 1.4rem;
        font-weight: 600;
        font-size: 15px;
    }
    div.stButton > button:hover {
        background: #4338CA;
        color: white;
    }

    /* Sidebar nav buttons: override the general button style above */
    section[data-testid="stSidebar"] div.stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
        border-radius: 10px !important;
        padding: 0.55rem 0.9rem !important;
        margin-bottom: 2px;
        font-size: 17px !important;
        width: 100%;
    }
    section[data-testid="stSidebar"] div.stButton > button > div {
        justify-content: flex-start !important;
        width: 100%;
    }
    section[data-testid="stSidebar"] div.stButton > button p {
        text-align: left !important;
    }
    /* Active nav item (type="primary") */
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        background: #EEF0FE !important;
        color: #4F46E5 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p {
        color: #4F46E5 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
        background: #E4E7FD !important;
        color: #4F46E5 !important;
    }
    /* Inactive nav items (type="tertiary") */
    section[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] {
        background: transparent !important;
        color: #374151 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] p {
        color: #374151 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:hover {
        background: #F3F4F6 !important;
        color: #111827 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:hover p {
        color: #111827 !important;
    }

    /* Table header row */
    .table-header {
        display: flex;
        padding: 10px 4px;
        border-bottom: 1px solid #EEF0F5;
        color: #9CA3AF;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .table-row {
        display: flex;
        align-items: center;
        padding: 16px 4px;
        border-bottom: 1px solid #F3F4F6;
        font-size: 15px;
        color: #111827;
    }
    .col-title { flex: 3; font-weight: 500; }
    .col-result { flex: 1; }
    .col-conf { flex: 1.4; }
    .col-date { flex: 1.2; color: #6B7280; }

    /* Progress bar for confidence */
    .conf-bar-bg {
        background: #F3F4F6;
        border-radius: 6px;
        height: 6px;
        width: 100%;
        margin-top: 6px;
        overflow: hidden;
    }
    .conf-bar-fill-fake { background: #E11D48; height: 100%; border-radius: 6px; }
    .conf-bar-fill-safe { background: #16A34A; height: 100%; border-radius: 6px; }

    /* Top right header bar */
    .top-bar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 12px;
        margin-top: -6px;
    }

    /* Reduce top spacing (config.toml toolbarMode=minimal already hides Deploy/menu) */
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
# ---------------- Sidebar ----------------
nav_items = [
    ("Dashboard", "grid_view"),
    ("Check Job", "search"),
    ("Scan History", "description"),
    ("Alerts", "notifications"),
    ("Tips", "lightbulb"),
    ("About", "info"),
]

if 'sidebar_nav' not in st.session_state:
    st.session_state.sidebar_nav = "Dashboard"

shield_icon = '''<svg width="26" height="26" viewBox="0 0 24 24" fill="#4F46E5"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/><path d="M10.5 15.5L7 12l1.4-1.4 2.1 2.1 5.1-5.1L17 9l-6.5 6.5z" fill="white"/></svg>'''

with st.sidebar:
    st.markdown(f'''
    <div style="display:flex; align-items:center; gap:10px; margin-top: 0rem; margin-bottom: 18px;">
        <div>{shield_icon}</div>
        <div style="font-weight:800; font-size:24px; line-height:1.25; color:#111827;">Fake Job<br>Detection</div>
    </div>
    ''', unsafe_allow_html=True)

    for label, icon in nav_items:
        is_active = (st.session_state.sidebar_nav == label)
        btn_type = "primary" if is_active else "tertiary"
        if st.button(label, icon=f":material/{icon}:", key=f"nav_{label}", use_container_width=True, type=btn_type):
            st.session_state.sidebar_nav = label
            st.rerun()

    st.markdown('''
    <div class="sidebar-bottom-anchor">
    <hr style="margin: 8px 0 14px 0; border-color: #EEF0F5;">
    <div style="background:#F3F4FE; border-radius:14px; padding:16px;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            <span style="color:#4F46E5;">🛡️</span>
            <span style="color:#4F46E5; font-weight:700; font-size:17px;">Stay Safe!</span>
        </div>
        <div style="color:#6B7280; font-size:14px;">Verify before you apply.</div>
    </div>
    </div>
    ''', unsafe_allow_html=True)

page = st.session_state.sidebar_nav

# ---------------- Derived stats ----------------
history_df = pd.DataFrame(st.session_state.history)
jobs_checked = len(history_df)
fake_count = (history_df['Result'].str.contains("FAKE").sum()) if jobs_checked else 0
safe_count = jobs_checked - fake_count
model_accuracy = 95  # from training report

# ---------------- PAGE: Dashboard ----------------
if page == "Dashboard":
    col_t, col_r = st.columns([3, 1.3])
    with col_t:
        st.markdown("# Dashboard")
        st.markdown('<p class="dash-subtitle">Detect fake job postings and stay safe.</p>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns([1, 2.2])
        with btn_col1:
            st.markdown('<div style="background:#fff;border:1px solid #EEF0F5;border-radius:10px;padding:8px 14px; font-size:16px; text-align:center;">⚙️</div>', unsafe_allow_html=True)
        with btn_col2:
            def _go_to_check_job():
                st.session_state.sidebar_nav = "Check Job"
            st.button("➕ Check New Job", key="top_check_new_job", use_container_width=True, on_click=_go_to_check_job)

    st.write("")
    icon_doc = '<svg width="22" height="22" viewBox="0 0 24 24" fill="#16A34A"><path d="M6 2c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6H6zm7 7V3.5L18.5 9H13z"/></svg>'
    icon_warn = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#E11D48" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
    icon_shield = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg>'
    icon_chart = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon icon-green">{icon_doc}</div>
            <div class="stat-text">
                <div class="stat-label">Jobs Checked</div>
                <div class="stat-value">{jobs_checked}</div>
                <div class="stat-sub">Total jobs analyzed</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon icon-red">{icon_warn}</div>
            <div class="stat-text">
                <div class="stat-label">Fake Jobs</div>
                <div class="stat-value">{fake_count}</div>
                <div class="stat-sub">Identified as fake</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon icon-amber">{icon_shield}</div>
            <div class="stat-text">
                <div class="stat-label">Safe Jobs</div>
                <div class="stat-value">{safe_count}</div>
                <div class="stat-sub">Looks safe</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon icon-blue">{icon_chart}</div>
            <div class="stat-text">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value">{model_accuracy}%</div>
                <div class="stat-sub">Model accuracy</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    col_form, col_illus = st.columns([2.2, 1])
    with col_form:
        st.markdown('<div class="card-title">Check a Job Posting</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">Paste job description or details to check if it\'s fake or real.</div>', unsafe_allow_html=True)
        quick_text = st.text_area("job_input", placeholder="Paste job description here...", height=130, label_visibility="collapsed")
        if st.button("🔍  Analyze Job"):
            if quick_text.strip():
                text_vec = vectorizer.transform([quick_text])
                prediction = model.predict(text_vec)[0]
                probability = model.predict_proba(text_vec)[0]
                confidence = probability[prediction] * 100
                result_label = "FAKE" if prediction == 1 else "SAFE"
                first_line = quick_text.strip().split("\n")[0]
                st.session_state.history.append({
                    "Title": first_line[:45] + ("..." if len(first_line) > 45 else ""),
                    "Result": result_label,
                    "Confidence": round(confidence),
                    "Checked On": datetime.now().strftime("%b %d, %Y")
                })
                st.rerun()
            else:
                st.warning("Please paste a job description first.")
    with col_illus:
        st.markdown(
            '<div style="text-align:center; padding-top:10px;">'
            '<div style="font-size:70px;">🖥️🔍</div>'
            '<p style="color:#6B7280; font-size:13px; margin-top:10px;">Our AI model analyzes the job post and detects red flags instantly.</p>'
            '</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    rs_col1, rs_col2 = st.columns([4, 1])
    with rs_col1:
        st.markdown('<div class="card-title">Recent Scans</div>', unsafe_allow_html=True)
    with rs_col2:
        if jobs_checked > 0:
            def _go_to_history():
                st.session_state.sidebar_nav = "Scan History"
            st.button("View All →", key="view_all_scans", on_click=_go_to_history, use_container_width=True)
    st.write("")
    if jobs_checked == 0:
        st.caption("No jobs checked yet. Try analyzing one above!")
    else:
        st.markdown("""
        <div class="table-header">
            <div class="col-title">Job Title</div>
            <div class="col-result">Result</div>
            <div class="col-conf">Confidence</div>
            <div class="col-date">Checked On</div>
            <div style="width:20px;"></div>
        </div>
        """, unsafe_allow_html=True)
        recent = history_df.tail(5).iloc[::-1]
        for _, row in recent.iterrows():
            is_fake = row["Result"] == "FAKE"
            badge_html = f'<span class="badge-fake">Fake</span>' if is_fake else f'<span class="badge-safe">Safe</span>'
            bar_class = "conf-bar-fill-fake" if is_fake else "conf-bar-fill-safe"
            conf_val = row["Confidence"]
            st.markdown(f"""
            <div class="table-row">
                <div class="col-title">{row["Title"]}</div>
                <div class="col-result">{badge_html}</div>
                <div class="col-conf">
                    {conf_val}%
                    <div class="conf-bar-bg"><div class="{bar_class}" style="width:{conf_val}%;"></div></div>
                </div>
                <div class="col-date">{row["Checked On"]}</div>
                <div style="width:20px; color:#C4C9D4; font-size:16px;">›</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PAGE: Check Job (detailed, with explainability) ----------------
elif page == "Check Job":
    st.markdown("# Check a Job Posting")
    st.markdown('<p class="dash-subtitle">Paste full details for a deeper AI analysis with reasoning.</p>', unsafe_allow_html=True)
    st.write("")

    title = st.text_input("Job Title")
    description = st.text_area("Job Description", height=200)

    if st.button("Analyze Job", type="primary"):
        if title.strip() == "" and description.strip() == "":
            st.warning("Please job title ya description daalo!")
        else:
            text = title + ' ' + description
            text_vec = vectorizer.transform([text])
            prediction = model.predict(text_vec)[0]
            probability = model.predict_proba(text_vec)[0]
            confidence = probability[prediction] * 100
            result_label = "FAKE" if prediction == 1 else "SAFE"

            st.session_state.history.append({
                "Title": title if title else "(no title)",
                "Result": result_label,
                "Confidence": round(confidence),
                "Checked On": datetime.now().strftime("%b %d, %Y")
            })

            col_a, col_b = st.columns(2)
            with col_a:
                if prediction == 1:
                    st.markdown(f'<span class="badge-fake" style="font-size:16px;">⚠️ FAKE</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-safe" style="font-size:16px;">✅ SAFE</span>', unsafe_allow_html=True)
            with col_b:
                st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("🔍 Why this prediction?")
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]
            text_vec_array = text_vec.toarray()[0]
            present_word_indices = np.where(text_vec_array > 0)[0]
            word_scores = [(feature_names[i], coefficients[i]) for i in present_word_indices]
            word_scores.sort(key=lambda x: x[1], reverse=True)

            fake_indicators = [w for w in word_scores if w[1] > 0][:5]
            real_indicators = [w for w in word_scores if w[1] < 0][:5]

            col1, col2 = st.columns(2)
            with col1:
                st.write("**🚩 Red Flag Words (suggest FAKE):**")
                if fake_indicators:
                    chart_df = pd.DataFrame(fake_indicators, columns=["Word", "Weight"])
                    fig = px.bar(chart_df, x="Weight", y="Word", orientation="h", color_discrete_sequence=["#E11D48"])
                    fig.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(color="#111827"),
                        xaxis=dict(color="#111827", gridcolor="#EEF0F5"),
                        yaxis=dict(color="#111827")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Koi strong fake indicator nahi mila")
            with col2:
                st.write("**✅ Trust Words (suggest SAFE):**")
                if real_indicators:
                    chart_df = pd.DataFrame(real_indicators, columns=["Word", "Weight"])
                    chart_df["Weight"] = chart_df["Weight"].abs()
                    fig = px.bar(chart_df, x="Weight", y="Word", orientation="h", color_discrete_sequence=["#16A34A"])
                    fig.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(color="#111827"),
                        xaxis=dict(color="#111827", gridcolor="#EEF0F5"),
                        yaxis=dict(color="#111827")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Koi strong real indicator nahi mila")

# ---------------- PAGE: Scan History ----------------
elif page == "Scan History":
    st.markdown("# Scan History")
    st.markdown('<p class="dash-subtitle">All jobs you\'ve checked in this session.</p>', unsafe_allow_html=True)
    st.write("")
    if jobs_checked == 0:
        st.info("No jobs checked yet. Go to 'Check Job' to try one!")
    else:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        display_df = history_df.iloc[::-1].copy()
        display_df["Confidence"] = display_df["Confidence"].astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# ---------------- PAGE: Tips ----------------
elif page == "Tips":
    st.markdown("# Tips to Spot Fake Jobs")
    st.write("")
    tips = [
        ("💰", "Too Good To Be True Salary", "Extremely high pay for minimal work or experience is a major red flag."),
        ("💳", "Upfront Payment Requests", "Legitimate employers never ask you to pay registration or training fees."),
        ("🏢", "No Verifiable Company Info", "Search the company name — no website or LinkedIn presence is suspicious."),
        ("⚡", "Urgency & Pressure", "Scammers push you to decide or pay immediately without giving time to think."),
        ("📧", "Unprofessional Communication", "Generic email domains, poor grammar, and vague job descriptions are warning signs."),
    ]
    for icon, t, d in tips:
        st.markdown(f"""
        <div class="card-box" style="margin-bottom:12px;">
            <b>{icon} {t}</b><br>
            <span style="color:#6B7280;">{d}</span>
        </div>
        """, unsafe_allow_html=True)

# ---------------- PAGE: Alerts ----------------
elif page == "Alerts":
    st.markdown("# Alerts")
    st.markdown('<p class="dash-subtitle">Recent fake job detections from your scans.</p>', unsafe_allow_html=True)
    st.write("")
    if fake_count == 0:
        st.info("No fake jobs flagged yet. Check a job posting to see alerts here.")
    else:
        fake_rows = history_df[history_df["Result"] == "FAKE"].iloc[::-1]
        for _, row in fake_rows.iterrows():
            st.markdown(f"""
            <div class="card-box" style="margin-bottom:12px; border-left: 4px solid #E11D48;">
                <b>⚠️ {row["Title"]}</b><br>
                <span style="color:#6B7280;">Flagged as FAKE with {row["Confidence"]}% confidence — {row["Checked On"]}</span>
            </div>
            """, unsafe_allow_html=True)

# ---------------- PAGE: About ----------------
elif page == "About":
    st.markdown("# About")
    st.write("")
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown("""
    **Fake Job Detection** is an AI-powered tool that helps job seekers identify fraudulent job postings before applying.

    It uses a **TF-IDF + Logistic Regression** model trained on thousands of real and fake job listings to flag suspicious postings, and explains *why* a posting was flagged by highlighting the specific words that influenced the prediction.

    **Model accuracy:** ~95% &nbsp;|&nbsp; **Built with:** Python, scikit-learn, Streamlit
    """)
    st.markdown('</div>', unsafe_allow_html=True)