import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             classification_report, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve, auc)

st.set_page_config(page_title="Heart Failure Prediction", layout="wide")
st.title("❤️ Heart Failure Prediction & Survival Model")
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #7c3aed;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #7c3aed;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #7c3aed !important;
    }
    
    /* Normal text — dark so it shows on light bg */
    p, label, .stText {
        color: #1a1a1a !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 2px solid #7c3aed;
        border-radius: 8px;
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)
# ── Load Data ──────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_clinical_records_dataset.csv")

dataset = load_data()

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", [
    "📋 Dataset Overview",
    "📊 EDA",
    "🤖 Logistic Regression",
    "🌲 Random Forest",
    "📈 Survival Analysis (Cox PH)"
])

standardize_features = ['age','creatinine_phosphokinase','ejection_fraction',
                        'platelets','serum_creatinine','serum_sodium']

# ── Helpers ────────────────────────────────────────────────
def get_train_test():
    X = dataset.drop(['DEATH_EVENT', 'time'], axis=1)
    y = dataset['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, stratify=y, random_state=42)
    sc = StandardScaler()
    X_train_std = X_train.copy()
    X_test_std  = X_test.copy()
    X_train_std[standardize_features] = sc.fit_transform(X_train[standardize_features])
    X_test_std[standardize_features]  = sc.transform(X_test[standardize_features])
    return X_train_std, X_test_std, y_train, y_test

# ══════════════════════════════════════════════════════════
if section == "📋 Dataset Overview":
    st.header("Dataset Overview")
    st.write(f"Shape: {dataset.shape}")
    st.dataframe(dataset.head(10))
    st.subheader("Missing Values")
    st.write(dataset.isnull().sum())
    st.subheader("Statistics")
    st.dataframe(dataset.describe())

# ══════════════════════════════════════════════════════════
elif section == "📊 EDA":
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Death Event Distribution")
        fig, ax = plt.subplots()
        dataset['DEATH_EVENT'].value_counts().plot.bar(ax=ax, color=['steelblue','tomato'])
        ax.set_xticklabels(['Survived','Died'], rotation=0)
        st.pyplot(fig)
    with col2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader("Feature Histograms")
    cols = st.columns(3)
    for i, col in enumerate(dataset.columns):
        with cols[i % 3]:
            fig, ax = plt.subplots()
            ax.hist(dataset[col], bins='auto', color='steelblue')
            ax.set_title(col)
            st.pyplot(fig)

# ══════════════════════════════════════════════════════════
elif section == "🤖 Logistic Regression":
    st.header("Logistic Regression Model")
    X_train_std, X_test_std, y_train, y_test = get_train_test()
    model = LogisticRegression()
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)

    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Feature Coefficients")
        coeffs = pd.Series(model.coef_[0], index=X_train_std.columns)
        fig, ax = plt.subplots()
        coeffs.sort_values().plot.barh(ax=ax)
        st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

# ══════════════════════════════════════════════════════════
elif section == "🌲 Random Forest":
    st.header("Random Forest Model")
    X_train_std, X_test_std, y_train, y_test = get_train_test()
    rf = RandomForestClassifier(n_estimators=100, random_state=10)
    rf.fit(X_train_std, y_train)
    y_pred_rf = rf.predict(X_test_std)

    acc = accuracy_score(y_test, y_pred_rf)
    st.metric("Accuracy", f"{acc:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf)).plot(ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("ROC Curve")
        probs = rf.predict_proba(X_test_std)[:,1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr,tpr):.2f}")
        ax.plot([0,1],[0,1],'--', color='grey')
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(); st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred_rf))

# ══════════════════════════════════════════════════════════
elif section == "📈 Survival Analysis (Cox PH)":
    st.header("Cox Proportional Hazards Model")
    try:
        from lifelines import CoxPHFitter
        df_dev, df_test = train_test_split(dataset, test_size=0.2, random_state=42)
        df_train, df_val = train_test_split(df_dev, test_size=0.25, random_state=42)
        sc = StandardScaler()
        df_train = df_train.copy(); df_val = df_val.copy(); df_test = df_test.copy()
        df_train[standardize_features] = sc.fit_transform(df_train[standardize_features])
        df_val[standardize_features]   = sc.transform(df_val[standardize_features])
        df_test[standardize_features]  = sc.transform(df_test[standardize_features])
        cph = CoxPHFitter()
        # NAYI (fix)
        cph.fit(df_train, duration_col='time', event_col='DEATH_EVENT',
        show_progress=False)
        st.subheader("Model Summary")
        fig, ax = plt.subplots(figsize=(8,6))
        cph.plot(ax=ax)
        st.pyplot(fig)
        st.subheader("Smoking Effect on Survival")
        fig, ax = plt.subplots()
        cph.plot_partial_effects_on_outcome('smoking', values=[0,1], ax=ax)
        st.pyplot(fig)
    except ImportError:
        st.error("lifelines library not found. Add it to requirements.txt")