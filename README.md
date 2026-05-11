# 🔍 Fraud Detection System

Sistem deteksi transaksi kartu kredit mencurigakan menggunakan Machine Learning (XGBoost) dengan SHAP Explainability, dilengkapi web app interaktif berbasis Streamlit.

## 📊 Performa Model

| Metrik | Skor |
|--------|------|
| Accuracy | 99.92% |
| F1-Score | 78.30% |
| AUC-ROC | 98.00% |

## 🛠️ Tech Stack

- **Python** — bahasa pemrograman utama
- **XGBoost** — algoritma machine learning utama
- **SHAP** — explainability model
- **Scikit-learn** — preprocessing dan evaluasi
- **Imbalanced-learn** — menangani class imbalance (SMOTE)
- **Streamlit** — web app interaktif
- **Pandas & NumPy** — pengolahan data
- **Matplotlib & Seaborn** — visualisasi data

## 📁 Struktur Project
fraud-detection/
├── app/
│   └── app.py              # Streamlit web app
├── data/
│   └── creditcard.csv      # Dataset (tidak diupload, lihat sumber)
├── models/
│   ├── xgboost_model.json  # Model XGBoost terlatih
│   └── scaler.pkl          # StandardScaler
├── notebooks/
│   ├── 01_eda.ipynb        # Eksplorasi data
│   ├── 02_modeling.ipynb   # Training model
│   └── 03_shap_explainability.ipynb  # SHAP analysis
└── README.md

## 🚀 Cara Menjalankan

**1. Clone repository**
```bash
git clone https://github.com/fahrihamd1/Fraud-Detection-System.git
cd Fraud-Detection-System
```

**2. Install library**
```bash
pip install -r requirements.txt
```

**3. Download dataset**

Download dataset dari [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dan simpan di folder `data/creditcard.csv`

**4. Jalankan web app**
```bash
streamlit run app/app.py
```

## 📈 Alur Project

1. **Eksplorasi Data (EDA)** — memahami distribusi data dan class imbalance (hanya 0.17% fraud)
2. **Preprocessing** — StandardScaler untuk normalisasi, SMOTE untuk menangani class imbalance
3. **Modeling** — melatih dan membandingkan Logistic Regression, Random Forest, dan XGBoost
4. **Explainability** — SHAP untuk menjelaskan keputusan model per transaksi
5. **Deployment** — Streamlit web app untuk demo interaktif

## 📦 Dataset

- **Sumber**: [Kaggle — Credit Card Fraud Detection by ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Jumlah transaksi**: 284.807
- **Jumlah fraud**: 492 (0.17%)
- **Fitur**: 30 kolom (Time, V1-V28, Amount)

## 👤 Author

**Muhammad Fahri Hamdi**  
[GitHub](https://github.com/fahrihamd1)
