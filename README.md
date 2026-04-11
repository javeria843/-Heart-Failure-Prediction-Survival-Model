DEPLOYED APP: https://juofarvcqn4pz52eqh3t2v.streamlit.app/


# 🫀 Heart Failure Clinical Prediction System

An end-to-end **Machine Learning + Streamlit web application** designed to predict **30-day mortality risk in heart failure patients** using clinical data.
This system integrates multiple ML models, explainability (SHAP), and survival analysis to support data-driven clinical insights.

---

## 📌 Overview

Heart failure is a critical medical condition requiring timely risk assessment. This project provides:

* 📊 Exploratory Data Analysis (EDA)
* 🤖 Multiple ML models for prediction
* 🧠 Model explainability using SHAP
* 📈 Survival analysis using Cox Proportional Hazards
* 📄 Automated patient report generation

---

## 🚀 Features

* **Live Prediction System**

  * Input patient clinical parameters
  * Get instant mortality risk prediction

* **Machine Learning Models**

  * Logistic Regression
  * Random Forest
  * XGBoost

* **Explainable AI (XAI)**

  * SHAP feature importance
  * Individual patient-level explanations

* **Model Evaluation**

  * Accuracy, F1-score, Precision, Recall
  * ROC curves & Confusion Matrix

* **Survival Analysis**

  * Kaplan-Meier curves
  * Cox Proportional Hazards model

* **Report Generation**

  * Downloadable clinical risk report (HTML → PDF)

---

## 🗂️ Dataset

* **Name:** Heart Failure Clinical Records Dataset
* **Features:** 12 clinical attributes
* **Target:** `DEATH_EVENT` (0 = Survived, 1 = Died)

### Key Features:

* Age
* Ejection Fraction
* Serum Creatinine
* Serum Sodium
* Platelets
* Diabetes, Anaemia, Smoking, etc.

---

## ⚙️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Libraries:**

  * pandas, numpy
  * scikit-learn
  * matplotlib, seaborn
  * xgboost
  * shap
  * lifelines

---

## 🧠 Model Workflow

1. Data preprocessing (scaling + splitting)
2. Model training
3. Evaluation using multiple metrics
4. Prediction on user input
5. Explainability using SHAP
6. Survival modeling using Cox PH

---

## 📊 Results

* Achieved strong predictive performance across models
* Random Forest and XGBoost showed better generalization
* SHAP analysis highlights:

  * Ejection Fraction
  * Serum Creatinine
  * Age
    as key mortality indicators

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

Deployed using **Streamlit Cloud**.

👉 Note: Ensure all dependencies (especially `shap`, `xgboost`, `lifelines`) are included in `requirements.txt`.

---

## 📁 Project Structure

```
├── app.py
├── requirements.txt
├── heart_failure_clinical_records_dataset.csv
└── README.md
```

---

## ⚠️ Disclaimer

This application is intended for **educational and research purposes only**.
It should **not be used as a substitute for professional medical advice or diagnosis**.

---

## 📜 License (MIT)

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## ✨ Future Improvements

* Deep learning models (ANN)
* Real-time hospital integration
* API-based deployment
* Improved UI/UX
* Larger medical datasets

---

## 👩‍💻 Author

**Javeria**
Engineering Student | AI & ML Enthusiast

---

## ⭐ Acknowledgements

* UCI Machine Learning Repository
* Open-source ML community
* Streamlit platform
