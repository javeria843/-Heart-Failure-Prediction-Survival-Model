import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             classification_report, ConfusionMatrixDisplay,
                             roc_curve, auc, f1_score, precision_score, recall_score)
import warnings, io, base64
from datetime import datetime
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Failure Clinical System", layout="wide", page_icon="🫀")

# ── MEDICAL THEME ──────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }

    .stApp { background-color: #f4f6f9; }

    [data-testid="stSidebar"] { background: #0b1f3a !important; border-right: 4px solid #b8001f; }
    [data-testid="stSidebar"] * { color: #d0dae8 !important; }
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255,255,255,0.05); border-left: 3px solid transparent;
        border-radius: 3px; padding: 9px 14px; margin: 2px 0; display: block;
        font-size: 0.875rem; transition: all 0.15s; color: #a8bcd0 !important;
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        border-left-color: #b8001f !important; background: rgba(184,0,31,0.12) !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMarkdown p { color: #6a85a0 !important; font-size: 0.78rem; }
    [data-testid="stSidebar"] .stMarkdown strong { color: #c8d8e8 !important; }

    h1 {
        font-family: 'Lora', serif !important; color: #0b1f3a !important;
        font-weight: 700 !important; font-size: 1.85rem !important;
        border-bottom: 3px solid #b8001f; padding-bottom: 10px;
        margin-bottom: 22px !important;
    }
    h2, h3 { font-family: 'Lora', serif !important; color: #0b1f3a !important; font-weight: 600 !important; }

    .metric-card {
        background: #ffffff; border-radius: 4px; padding: 16px 20px;
        border: 1px solid #dce3ec; border-top: 4px solid #b8001f;
        box-shadow: 0 1px 6px rgba(11,31,58,0.07); margin-bottom: 14px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #0b1f3a; font-family: 'Lora', serif; line-height: 1.1; margin-top: 4px; }
    .metric-label { font-size: 0.7rem; color: #7a8fa6; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }

    .section-card {
        background: #ffffff; border-radius: 4px; padding: 22px 26px;
        border: 1px solid #dce3ec; box-shadow: 0 1px 5px rgba(11,31,58,0.05); margin-bottom: 18px;
    }
    .risk-high {
        background: #fff5f5; border: 1px solid #f5c6c6; border-left: 7px solid #b8001f;
        border-radius: 4px; padding: 24px; text-align: center;
    }
    .risk-low {
        background: #f0faf4; border: 1px solid #b6e4c8; border-left: 7px solid #0a6640;
        border-radius: 4px; padding: 24px; text-align: center;
    }
    .stButton > button {
        background: #b8001f !important; color: white !important; border: none !important;
        border-radius: 3px !important; padding: 12px 32px !important; font-weight: 600 !important;
        font-size: 0.9rem !important; width: 100% !important; letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }
    .stButton > button:hover { background: #900018 !important; }
    .dl-btn {
        display: inline-block; background: #0b1f3a; color: white !important;
        padding: 10px 26px; border-radius: 3px; font-weight: 600; font-size: 0.82rem;
        text-decoration: none !important; letter-spacing: 0.06em; text-transform: uppercase; margin-top: 14px;
    }
    .dl-btn:hover { background: #142d50; }
    .stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
        color: #2c3e55 !important; font-weight: 500; font-size: 0.88rem;
    }
    .rpt-header {
        background: #0b1f3a; border-bottom: 4px solid #b8001f;
        padding: 20px 28px; border-radius: 4px 4px 0 0;
    }
    .rpt-body {
        background: #ffffff; border: 1px solid #dce3ec; border-top: none;
        padding: 24px 28px; border-radius: 0 0 4px 4px; margin-bottom: 20px;
    }
    .rpt-row {
        display: flex; justify-content: space-between; padding: 7px 0;
        border-bottom: 1px solid #f0f3f7; font-size: 0.88rem;
    }
    .rpt-row:last-child { border-bottom: none; }
    .rpt-key { color: #7a8fa6; font-weight: 500; }
    .rpt-val { color: #0b1f3a; font-weight: 600; }
    div[data-testid="stDataFrame"] { border-radius: 4px; border: 1px solid #dce3ec !important; }
    .stTabs [data-baseweb="tab"] { color: #0b1f3a; font-weight: 500; font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { border-bottom: 3px solid #b8001f !important; color: #b8001f !important; }
    hr { border-color: #dce3ec !important; }
    </style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_clinical_records_dataset.csv")

dataset = load_data()
STANDARDIZE = ['age','creatinine_phosphokinase','ejection_fraction',
               'platelets','serum_creatinine','serum_sodium']

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 14px 0;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:14px;'>
        <div style='font-size:2.2rem;'>🫀</div>
        <div style='font-family:Lora,serif;font-size:1.05rem;font-weight:700;color:#e8f0f8;margin-top:6px;'>Heart Failure</div>
        <div style='font-size:0.7rem;color:#6a85a0;letter-spacing:0.12em;text-transform:uppercase;margin-top:3px;'>Clinical Prediction System</div>
    </div>
    """, unsafe_allow_html=True)
    section = st.radio("Navigate", [
        "🏠 Overview", "📊 EDA", "🔮 Live Prediction",
        "🤖 Logistic Regression", "🌲 Random Forest", "⚡ XGBoost",
        "📊 Model Comparison", "🧠 SHAP Explainability", "📈 Survival Analysis"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Dataset:** Heart Failure Clinical Records")
    st.markdown(f"**Patients:** {len(dataset)}")
    st.markdown(f"**Features:** {dataset.shape[1]-1}")
    st.markdown("**Models:** LR · RF · XGBoost")

# ── HELPERS ──────────────────────────────────────────────
@st.cache_data
def get_train_test():
    X = dataset.drop(['DEATH_EVENT','time'], axis=1)
    y = dataset['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr = X_train.copy(); X_te = X_test.copy()
    X_tr[STANDARDIZE] = sc.fit_transform(X_train[STANDARDIZE])
    X_te[STANDARDIZE] = sc.transform(X_test[STANDARDIZE])
    return X_tr, X_te, y_train, y_test, sc, list(X_train.columns)

@st.cache_resource
def train_all_models():
    X_tr, X_te, y_tr, y_te, sc, cols = get_train_test()
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_tr, y_tr)
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            use_label_encoder=False, eval_metric='logloss',
                            random_state=42, verbosity=0)
        xgb.fit(X_tr, y_tr)
    except ImportError:
        xgb = None
    return lr, rf, xgb

def get_metrics(model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:,1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    return {'accuracy': accuracy_score(y_te, y_pred), 'f1': f1_score(y_te, y_pred),
            'precision': precision_score(y_te, y_pred), 'recall': recall_score(y_te, y_pred),
            'auc': auc(fpr, tpr), 'y_pred': y_pred, 'y_prob': y_prob, 'fpr': fpr, 'tpr': tpr}

def medical_metrics(items):
    cols = st.columns(len(items))
    for col, (label, val) in zip(cols, items):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                        f'<div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

def ax_style(ax, fig):
    ax.set_facecolor('#fafbfc'); fig.patch.set_facecolor('#ffffff')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#dce3ec')

# ── SHAP FIX: always returns clean 2D array (samples × features) ──
def get_sv_2d(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]          # old SHAP RF: list[class0, class1]
    sv = np.array(shap_values)
    if sv.ndim == 3:
        return sv[:, :, 1]             # new SHAP RF: (samples, features, classes)
    return sv                          # XGBoost: already 2D

# ── PDF / HTML report ─────────────────────────────────────
def build_html_report(patient_vals, prob, pred, model_name, now):
    rc = "#b8001f" if pred==1 else "#0a6640"
    rl = "HIGH RISK" if pred==1 else "LOW RISK"
    ri = "⚠️" if pred==1 else "✅"
    rec = ("Immediate clinical attention recommended. Cardiology referral advised."
           if pred==1 else "Continue standard monitoring. Schedule follow-up in 90 days.")
    rows = "".join(f'<tr><td class="rk">{k}</td><td class="rv">{v}</td></tr>'
                   for k, v in patient_vals.items())
    bg = '#fff5f5' if pred==1 else '#f0faf4'
    bc = '#f5c6c6' if pred==1 else '#b6e4c8'
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
body{{font-family:Georgia,serif;background:#f4f6f9;margin:0;padding:30px;color:#0b1f3a;}}
.page{{max-width:800px;margin:auto;background:white;border:1px solid #dce3ec;box-shadow:0 4px 20px rgba(0,0,0,0.1);}}
.hdr{{background:#0b1f3a;color:white;padding:28px 36px;border-bottom:5px solid #b8001f;}}
.hdr h1{{margin:0;font-size:1.5rem;}} .meta{{font-size:0.78rem;color:#7a9abf;margin-top:6px;font-family:sans-serif;}}
.body{{padding:30px 36px;}}
.risk-box{{background:{bg};border:1px solid {bc};border-left:8px solid {rc};border-radius:4px;padding:22px;text-align:center;margin-bottom:28px;}}
.rl{{font-size:1.4rem;font-weight:700;color:{rc};}} .rp{{font-size:2.8rem;font-weight:800;color:{rc};line-height:1.1;}}
.rs{{font-size:0.82rem;color:#6a7f95;font-family:sans-serif;margin-top:4px;}}
.st{{font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#7a8fa6;font-family:sans-serif;border-bottom:2px solid #b8001f;padding-bottom:5px;margin:22px 0 12px;}}
table{{width:100%;border-collapse:collapse;font-family:sans-serif;font-size:0.88rem;}}
td{{padding:8px 10px;border-bottom:1px solid #f0f3f7;}} .rk{{color:#7a8fa6;width:55%;}} .rv{{color:#0b1f3a;font-weight:600;}}
.rec{{background:#f4f6f9;border-left:4px solid #0b1f3a;padding:14px 18px;border-radius:3px;font-family:sans-serif;font-size:0.88rem;color:#2c3e55;margin-top:20px;}}
.foot{{background:#f4f6f9;border-top:1px solid #dce3ec;padding:14px 36px;font-size:0.72rem;color:#9aaabf;font-family:sans-serif;text-align:center;}}
</style></head><body>
<div class="page">
<div class="hdr"><h1>🫀 Heart Failure Risk Assessment Report</h1>
<div class="meta">Generated: {now} &nbsp;|&nbsp; Model: {model_name} &nbsp;|&nbsp; Report ID: HF-{datetime.now().strftime('%Y%m%d%H%M%S')}</div></div>
<div class="body">
<div class="risk-box"><div class="rl">{ri} {rl}</div><div class="rp">{prob:.1%}</div>
<div class="rs">Estimated 30-Day Mortality Probability</div></div>
<div class="st">Patient Clinical Parameters</div>
<table>{rows}</table>
<div class="rec"><strong>Clinical Recommendation:</strong> {rec}</div>
</div>
<div class="foot">This report is generated by an AI-based decision support tool and should not replace professional clinical judgement. Always consult a qualified physician.</div>
</div></body></html>"""

def dl_link(html_str, fname="patient_report.html"):
    b64 = base64.b64encode(html_str.encode()).decode()
    return f'<a class="dl-btn" href="data:text/html;base64,{b64}" download="{fname}">⬇ Download Report</a>'

# ══════════════════════════════════════════════════════════
# 🏠 OVERVIEW
# ══════════════════════════════════════════════════════════
if section == "🏠 Overview":
    st.title("Heart Failure Prediction System")
    st.markdown("*AI-assisted clinical decision support — 30-day mortality risk prediction*")
    medical_metrics([("Total Patients", len(dataset)),
                     ("Mortality Rate", f"{dataset['DEATH_EVENT'].mean():.1%}"),
                     ("Clinical Features", dataset.shape[1]-1), ("ML Models","3")])
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.dataframe(dataset.head(8), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Feature Reference")
        desc = {"age":"Patient age (years)","anaemia":"Red blood cell decrease (0/1)",
                "creatinine_phosphokinase":"CPK enzyme level (mcg/L)","diabetes":"Diabetes (0/1)",
                "ejection_fraction":"Heart pump efficiency (%)","high_blood_pressure":"Hypertension (0/1)",
                "platelets":"Platelet count (kiloplatelets/mL)","serum_creatinine":"Blood creatinine (mg/dL)",
                "serum_sodium":"Blood sodium (mEq/L)","sex":"Gender (0=Female,1=Male)",
                "smoking":"Smoking status (0/1)","time":"Follow-up period (days)"}
        st.dataframe(pd.DataFrame(list(desc.items()), columns=["Feature","Description"]),
                     use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Descriptive Statistics")
    st.dataframe(dataset.describe().round(2), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 📊 EDA
# ══════════════════════════════════════════════════════════
elif section == "📊 EDA":
    st.title("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Death Event Distribution")
        fig, ax = plt.subplots(figsize=(5,4))
        counts = dataset['DEATH_EVENT'].value_counts()
        bars = ax.bar(['Survived','Died'], counts.values, color=['#0a6640','#b8001f'], edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, str(val), ha='center', fontweight='bold', color='#0b1f3a')
        ax_style(ax, fig); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Age Distribution by Outcome")
        fig, ax = plt.subplots(figsize=(5,4))
        dataset[dataset.DEATH_EVENT==0]['age'].hist(ax=ax, alpha=0.65, color='#0a6640', label='Survived', bins=15)
        dataset[dataset.DEATH_EVENT==1]['age'].hist(ax=ax, alpha=0.65, color='#b8001f', label='Died', bins=15)
        ax.set_xlabel("Age"); ax.legend(); ax_style(ax, fig); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,5))
    sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap='RdYlBu_r', ax=ax, linewidths=0.5, square=True)
    fig.patch.set_facecolor('#ffffff'); st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Feature Distributions")
    gcols = st.columns(3)
    for i, feat in enumerate(['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium']):
        with gcols[i%3]:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.hist(dataset[feat], bins=20, color='#0b1f3a', alpha=0.8, edgecolor='white')
            ax.set_title(feat, fontweight='bold', fontsize=9, color='#0b1f3a')
            ax_style(ax, fig); st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 🔮 LIVE PREDICTION + REPORT DOWNLOAD
# ══════════════════════════════════════════════════════════
elif section == "🔮 Live Prediction":
    st.title("Live Patient Risk Prediction")
    st.markdown("*Enter clinical values to generate a downloadable risk assessment report.*")
    _, _, _, _, sc, feature_cols = get_train_test()
    lr, rf, xgb = train_all_models()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Patient Clinical Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age (years)", 20, 95, 60)
        ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 38)
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.1, step=0.1)
        serum_sodium = st.slider("Serum Sodium (mEq/L)", 110, 150, 136)
    with c2:
        creatinine_phosphokinase = st.number_input("CPK Enzyme (mcg/L)", 20, 8000, 250)
        platelets = st.number_input("Platelets (kiloplatelets/mL)", 25000, 850000, 265000, step=5000)
        anaemia = st.selectbox("Anaemia", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        diabetes = st.selectbox("Diabetes", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    with c3:
        high_blood_pressure = st.selectbox("High Blood Pressure", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
        smoking = st.selectbox("Smoking", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        model_choice = st.selectbox("Prediction Model", ["Random Forest","Logistic Regression","XGBoost"])
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Generate Risk Assessment Report"):
        input_data = pd.DataFrame([[age, anaemia, creatinine_phosphokinase, diabetes,
                                    ejection_fraction, high_blood_pressure, platelets,
                                    serum_creatinine, serum_sodium, sex, smoking]], columns=feature_cols)
        input_std = input_data.copy()
        input_std[STANDARDIZE] = sc.transform(input_data[STANDARDIZE])
        model_map = {"Random Forest": rf, "Logistic Regression": lr, "XGBoost": xgb}
        chosen_model = model_map[model_choice]

        if chosen_model is None:
            st.error("XGBoost not installed. Run: pip install xgboost")
        else:
            prob = chosen_model.predict_proba(input_std)[0][1]
            pred = chosen_model.predict(input_std)[0]
            rc = "#b8001f" if pred==1 else "#0a6640"
            rl = "HIGH RISK" if pred==1 else "LOW RISK"
            ri = "⚠️" if pred==1 else "✅"
            now = datetime.now().strftime("%B %d, %Y — %H:%M")
            rec = ("Immediate clinical attention recommended. Cardiology referral advised."
                   if pred==1 else "Continue standard monitoring. Schedule follow-up in 90 days.")
            bg = '#fff5f5' if pred==1 else '#f0faf4'
            bc = '#f5c6c6' if pred==1 else '#b6e4c8'

            # Report header
            st.markdown(f"""
            <div class="rpt-header">
                <div style='font-family:Lora,serif;font-size:1.2rem;font-weight:700;color:#ffffff;'>🫀 Heart Failure Risk Assessment Report</div>
                <div style='font-size:0.75rem;color:#7a9abf;margin-top:5px;font-family:sans-serif;'>
                    Generated: {now} &nbsp;|&nbsp; Model: {model_choice} &nbsp;|&nbsp; Report ID: HF-{datetime.now().strftime('%Y%m%d%H%M%S')}
                </div>
            </div>
            <div class="rpt-body">
            """, unsafe_allow_html=True)

            col_r1, col_r2 = st.columns([1,2])
            with col_r1:
                st.markdown(f"""
                <div style='background:{bg};border:1px solid {bc};border-left:7px solid {rc};
                            border-radius:4px;padding:22px;text-align:center;'>
                    <div style='font-size:0.7rem;color:#7a8fa6;font-weight:700;text-transform:uppercase;
                                letter-spacing:0.1em;font-family:sans-serif;'>Risk Classification</div>
                    <div style='font-size:1.4rem;font-weight:700;color:{rc};margin:8px 0;'>{ri} {rl}</div>
                    <div style='font-size:2.8rem;font-weight:800;color:{rc};line-height:1;'>{prob:.1%}</div>
                    <div style='font-size:0.78rem;color:#6a7f95;font-family:sans-serif;margin-top:6px;'>30-Day Mortality Probability</div>
                </div>
                """, unsafe_allow_html=True)
            with col_r2:
                st.markdown(f"""
                <div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
                            color:#7a8fa6;font-family:sans-serif;margin-bottom:10px;'>Patient Parameters</div>
                <div class="rpt-row"><span class="rpt-key">Age</span><span class="rpt-val">{age} years</span></div>
                <div class="rpt-row"><span class="rpt-key">Sex</span><span class="rpt-val">{'Female' if sex==0 else 'Male'}</span></div>
                <div class="rpt-row"><span class="rpt-key">Ejection Fraction</span><span class="rpt-val">{ejection_fraction}%</span></div>
                <div class="rpt-row"><span class="rpt-key">Serum Creatinine</span><span class="rpt-val">{serum_creatinine} mg/dL</span></div>
                <div class="rpt-row"><span class="rpt-key">Serum Sodium</span><span class="rpt-val">{serum_sodium} mEq/L</span></div>
                <div class="rpt-row"><span class="rpt-key">CPK Enzyme</span><span class="rpt-val">{creatinine_phosphokinase} mcg/L</span></div>
                <div class="rpt-row"><span class="rpt-key">Platelets</span><span class="rpt-val">{platelets:,}</span></div>
                <div class="rpt-row"><span class="rpt-key">Diabetes</span><span class="rpt-val">{'Yes' if diabetes else 'No'}</span></div>
                <div class="rpt-row"><span class="rpt-key">Anaemia</span><span class="rpt-val">{'Yes' if anaemia else 'No'}</span></div>
                <div class="rpt-row"><span class="rpt-key">High Blood Pressure</span><span class="rpt-val">{'Yes' if high_blood_pressure else 'No'}</span></div>
                <div class="rpt-row"><span class="rpt-key">Smoking</span><span class="rpt-val">{'Yes' if smoking else 'No'}</span></div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style='background:#f4f6f9;border-left:4px solid #0b1f3a;padding:12px 16px;
                            border-radius:3px;font-family:sans-serif;font-size:0.88rem;color:#2c3e55;margin-top:16px;'>
                    <strong>Clinical Recommendation:</strong> {rec}
                </div>
            </div>""", unsafe_allow_html=True)

            # Risk gauge
            st.markdown('<div class="section-card" style="margin-top:16px;">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7,2))
            ax.barh(['Mortality Risk'], [prob], color=rc, height=0.45)
            ax.barh(['Mortality Risk'], [1-prob], left=[prob], color='#e8ecf1', height=0.45)
            ax.set_xlim(0,1); ax.axvline(x=0.5, color='#7a8fa6', linestyle='--', alpha=0.6, linewidth=1)
            ax.set_xlabel("Probability (0 = No Risk  →  1 = Certain)")
            ax_style(ax, fig)
            ax.text(prob/2, 0, f"{prob:.1%}", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
            st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

            # Download button
            pv = {"Age":f"{age} years","Sex":"Female" if sex==0 else "Male",
                  "Ejection Fraction":f"{ejection_fraction}%","Serum Creatinine":f"{serum_creatinine} mg/dL",
                  "Serum Sodium":f"{serum_sodium} mEq/L","CPK Enzyme":f"{creatinine_phosphokinase} mcg/L",
                  "Platelets":f"{platelets:,}","Diabetes":"Yes" if diabetes else "No",
                  "Anaemia":"Yes" if anaemia else "No","High Blood Pressure":"Yes" if high_blood_pressure else "No",
                  "Smoking":"Yes" if smoking else "No"}
            html_report = build_html_report(pv, prob, pred, model_choice, now)
            st.markdown(dl_link(html_report), unsafe_allow_html=True)
            st.caption("📄 Click to download → open in browser → Ctrl+P / Cmd+P to save as PDF")

# ══════════════════════════════════════════════════════════
# 🤖 LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════
elif section == "🤖 Logistic Regression":
    st.title("Logistic Regression")
    X_tr, X_te, y_tr, y_te, _, _ = get_train_test()
    lr, _, _ = train_all_models()
    m = get_metrics(lr, X_te, y_te)
    medical_metrics([("Accuracy",f"{m['accuracy']:.2%}"),("F1 Score",f"{m['f1']:.2%}"),
                     ("Precision",f"{m['precision']:.2%}"),("Recall",f"{m['recall']:.2%}")])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(confusion_matrix(y_te, m['y_pred']),
            display_labels=['Survived','Died']).plot(ax=ax, colorbar=False, cmap='Blues')
        fig.patch.set_facecolor('#ffffff'); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Feature Coefficients")
        coeffs = pd.Series(lr.coef_[0], index=X_tr.columns).sort_values()
        fig, ax = plt.subplots(figsize=(5,4))
        colors = ['#b8001f' if v > 0 else '#0a6640' for v in coeffs.values]
        coeffs.plot.barh(ax=ax, color=colors); ax.axvline(0, color='#0b1f3a', linewidth=0.8)
        ax_style(ax, fig); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Classification Report")
    st.text(classification_report(y_te, m['y_pred'], target_names=['Survived','Died']))
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 🌲 RANDOM FOREST
# ══════════════════════════════════════════════════════════
elif section == "🌲 Random Forest":
    st.title("Random Forest")
    X_tr, X_te, y_tr, y_te, _, _ = get_train_test()
    _, rf, _ = train_all_models()
    m = get_metrics(rf, X_te, y_te)
    medical_metrics([("Accuracy",f"{m['accuracy']:.2%}"),("F1 Score",f"{m['f1']:.2%}"),
                     ("AUC-ROC",f"{m['auc']:.2%}"),("Recall",f"{m['recall']:.2%}")])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(confusion_matrix(y_te, m['y_pred']),
            display_labels=['Survived','Died']).plot(ax=ax, colorbar=False, cmap='Greens')
        fig.patch.set_facecolor('#ffffff'); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(m['fpr'], m['tpr'], color='#0b1f3a', lw=2, label=f"AUC = {m['auc']:.3f}")
        ax.fill_between(m['fpr'], m['tpr'], alpha=0.08, color='#0b1f3a')
        ax.plot([0,1],[0,1],'--', color='#aab8c8'); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
        ax_style(ax, fig); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Feature Importance")
    fi = pd.Series(rf.feature_importances_, index=X_tr.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10,5))
    bc = ['#b8001f' if i >= len(fi)-2 else '#0b1f3a' for i in range(len(fi))]
    ax.barh(fi.index, fi.values, color=bc, alpha=0.85); ax.set_xlabel("Importance")
    ax_style(ax, fig); st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ⚡ XGBOOST
# ══════════════════════════════════════════════════════════
elif section == "⚡ XGBoost":
    st.title("XGBoost Classifier")
    X_tr, X_te, y_tr, y_te, _, _ = get_train_test()
    _, _, xgb = train_all_models()
    if xgb is None:
        st.error("XGBoost not installed. Run: `pip install xgboost`")
    else:
        m = get_metrics(xgb, X_te, y_te)
        medical_metrics([("Accuracy",f"{m['accuracy']:.2%}"),("F1 Score",f"{m['f1']:.2%}"),
                         ("AUC-ROC",f"{m['auc']:.2%}"),("Precision",f"{m['precision']:.2%}")])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5,4))
            ConfusionMatrixDisplay(confusion_matrix(y_te, m['y_pred']),
                display_labels=['Survived','Died']).plot(ax=ax, colorbar=False, cmap='Oranges')
            fig.patch.set_facecolor('#ffffff'); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("ROC Curve")
            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(m['fpr'], m['tpr'], color='#c45000', lw=2, label=f"AUC = {m['auc']:.3f}")
            ax.fill_between(m['fpr'], m['tpr'], alpha=0.08, color='#c45000')
            ax.plot([0,1],[0,1],'--', color='#aab8c8'); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
            ax_style(ax, fig); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Feature Importance")
        fi = pd.Series(xgb.feature_importances_, index=X_tr.columns).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(fi.index, fi.values, color='#c45000', alpha=0.85); ax.set_xlabel("Importance")
        ax_style(ax, fig); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Classification Report")
        st.text(classification_report(y_te, m['y_pred'], target_names=['Survived','Died']))
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 📊 MODEL COMPARISON
# ══════════════════════════════════════════════════════════
elif section == "📊 Model Comparison":
    st.title("Model Comparison")
    X_tr, X_te, y_tr, y_te, _, _ = get_train_test()
    lr, rf, xgb = train_all_models()
    models = {"Logistic Regression": lr, "Random Forest": rf}
    if xgb: models["XGBoost"] = xgb
    results = []
    for name, model in models.items():
        m = get_metrics(model, X_te, y_te)
        results.append({"Model":name,"Accuracy":f"{m['accuracy']:.2%}","F1 Score":f"{m['f1']:.2%}",
                        "AUC-ROC":f"{m['auc']:.2%}","Precision":f"{m['precision']:.2%}","Recall":f"{m['recall']:.2%}"})
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Performance Table")
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("ROC Curves — All Models")
    fig, ax = plt.subplots(figsize=(8,5))
    palette = {'Logistic Regression':'#0b1f3a','Random Forest':'#0a6640','XGBoost':'#c45000'}
    for name, model in models.items():
        m = get_metrics(model, X_te, y_te)
        ax.plot(m['fpr'], m['tpr'], lw=2, color=palette[name], label=f"{name} (AUC={m['auc']:.3f})")
    ax.plot([0,1],[0,1],'--', color='#aab8c8')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend(loc='lower right')
    ax_style(ax, fig); st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Metric Comparison Chart")
    metrics_raw = [{"Model":n,"Accuracy":get_metrics(m,X_te,y_te)['accuracy'],
                    "F1":get_metrics(m,X_te,y_te)['f1'],"AUC":get_metrics(m,X_te,y_te)['auc']}
                   for n, m in models.items()]
    df_m = pd.DataFrame(metrics_raw).set_index("Model")
    fig, ax = plt.subplots(figsize=(9,4))
    x = np.arange(len(df_m.columns)); width = 0.25
    mc = ['#0b1f3a','#0a6640','#c45000']
    for i, (idx, row) in enumerate(df_m.iterrows()):
        ax.bar(x+i*width, row.values, width, label=idx, color=mc[i], alpha=0.85)
    ax.set_xticks(x+width); ax.set_xticklabels(df_m.columns); ax.set_ylim(0,1); ax.legend()
    ax_style(ax, fig); st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 🧠 SHAP — FULLY FIXED
# ══════════════════════════════════════════════════════════
elif section == "🧠 SHAP Explainability":
    st.title("SHAP Explainability")
    st.markdown("*SHapley Additive exPlanations — understand exactly why the model predicts each outcome.*")
    try:
        import shap
        X_tr, X_te, y_tr, y_te, _, _ = get_train_test()
        _, rf, xgb = train_all_models()
        model_choice = st.radio("Select Model", ["Random Forest","XGBoost"], horizontal=True)
        chosen = rf if model_choice == "Random Forest" else xgb
        if chosen is None:
            st.error("XGBoost not installed. Run: pip install xgboost")
        else:
            with st.spinner("Computing SHAP values..."):
                explainer   = shap.TreeExplainer(chosen)
                shap_values = explainer.shap_values(X_te)
                sv = get_sv_2d(shap_values)   # ← FIXED: always clean 2D

            # Mean |SHAP| bar
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Feature Importance — Mean |SHAP|")
            mean_shap = np.abs(sv).mean(axis=0)
            shap_df = pd.Series(mean_shap, index=X_te.columns).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(9,5))
            bar_c = ['#b8001f' if v == shap_df.max() else '#0b1f3a' for v in shap_df.values]
            shap_df.plot.barh(ax=ax, color=bar_c, alpha=0.85)
            ax.set_xlabel("Mean |SHAP Value|"); ax_style(ax, fig); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

            # Beeswarm
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("SHAP Beeswarm Summary Plot")
            shap.summary_plot(sv, X_te, show=False)
            fig = plt.gcf(); fig.patch.set_facecolor('#ffffff')
            st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

            # Individual patient
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Individual Patient Explanation")
            patient_idx = st.slider("Select Patient Index", 0, len(X_te)-1, 0)
            patient_shap = sv[patient_idx]   # guaranteed 1-D
            patient_df = pd.DataFrame({
                "Feature": list(X_te.columns),
                "Value":   X_te.iloc[patient_idx].values.round(3),
                "SHAP Impact": patient_shap.round(4)
            }).sort_values("SHAP Impact", ascending=True)
            fig, ax = plt.subplots(figsize=(9,5))
            colors_p = ['#b8001f' if v > 0 else '#0a6640' for v in patient_df['SHAP Impact']]
            ax.barh(patient_df['Feature'], patient_df['SHAP Impact'], color=colors_p, alpha=0.85)
            ax.axvline(0, color='#0b1f3a', linewidth=0.8)
            ax.set_xlabel("SHAP Value  (Red = increases risk  |  Green = reduces risk)")
            ax_style(ax, fig); st.pyplot(fig); plt.close()
            st.dataframe(patient_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
    except ImportError:
        st.error("SHAP not installed. Run: `pip install shap`")

# ══════════════════════════════════════════════════════════
# 📈 SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════
elif section == "📈 Survival Analysis":
    st.title("Cox Proportional Hazards — Survival Analysis")
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        df_dev, df_test = train_test_split(dataset, test_size=0.2, random_state=42)
        df_train, df_val = train_test_split(df_dev, test_size=0.25, random_state=42)
        sc2 = StandardScaler()
        df_train = df_train.copy(); df_val = df_val.copy(); df_test = df_test.copy()
        df_train[STANDARDIZE] = sc2.fit_transform(df_train[STANDARDIZE])
        df_val[STANDARDIZE]   = sc2.transform(df_val[STANDARDIZE])
        df_test[STANDARDIZE]  = sc2.transform(df_test[STANDARDIZE])
        with st.spinner("Fitting Cox PH Model..."):
            cph = CoxPHFitter()
            cph.fit(df_train, duration_col='time', event_col='DEATH_EVENT', show_progress=False)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Cox Model Coefficients")
            fig, ax = plt.subplots(figsize=(6,5))
            cph.plot(ax=ax); ax_style(ax, fig); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Kaplan-Meier Survival Curve")
            kmf = KaplanMeierFitter()
            fig, ax = plt.subplots(figsize=(6,5))
            for val, label, color in [(0,"Survived",'#0a6640'),(1,"Died",'#b8001f')]:
                mask = dataset['DEATH_EVENT'] == val
                kmf.fit(dataset[mask]['time'], dataset[mask]['DEATH_EVENT'], label=label)
                kmf.plot_survival_function(ax=ax, color=color, ci_show=True)
            ax.set_xlabel("Days"); ax.set_ylabel("Survival Probability")
            ax_style(ax, fig); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Partial Effects on Outcome")
        feature_pe = st.selectbox("Select Feature", ['smoking','diabetes','anaemia','high_blood_pressure','sex'])
        fig, ax = plt.subplots(figsize=(8,4))
        cph.plot_partial_effects_on_outcome(feature_pe, values=[0,1], ax=ax)
        ax_style(ax, fig); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Cox Model Summary Table")
        st.dataframe(cph.summary.round(4), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    except ImportError:
        st.error("lifelines not installed. Run: `pip install lifelines`")
