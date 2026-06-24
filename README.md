# Credit Card Fraud Detection

End-to-end ML pipeline for detecting fraudulent transactions — covering EDA, class imbalance handling, multi-model training, hyperparameter tuning, and a live Gradio deployment.

---

## Overview

Financial fraud datasets are highly imbalanced (fraud < 0.2% of transactions). This project tackles that head-on using SMOTE oversampling, then benchmarks three classifiers to find the best performer.

---

## Pipeline

```
Raw Data → EDA → Preprocessing → SMOTE → Model Training → Tuning → Evaluation → Gradio App
```

---

## Key Steps

**1. Exploratory Data Analysis**
- Distribution analysis of all 30 features (Time, V1–V28, Amount)
- Boxplots, violin plots, and correlation heatmap
- Class imbalance visualization (~0.17% fraud rate)

**2. Preprocessing**
- Removed duplicates
- StandardScaler on `Time` and `Amount`
- Stratified train-test split (80/20)

**3. Imbalance Handling**
- Applied **SMOTE** (Synthetic Minority Oversampling) on training set only
- Balanced classes before model training to avoid data leakage

**4. Models Trained**
| Model | Notes |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | `class_weight` balanced |
| XGBoost | Tuned via RandomizedSearchCV |

**5. Hyperparameter Tuning**
- RandomizedSearchCV on XGBoost (30 iterations, 3-fold CV)
- Optimized: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `gamma`, `reg_alpha`, `reg_lambda`

**6. Evaluation Metrics**
- Classification Report (Precision, Recall, F1)
- ROC-AUC Score
- ROC Curve comparison across all models

**7. Deployment**
- Best model saved via `joblib`
- Real-time inference via **Gradio** interface

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 99.95% | 0.972 | 0.737 | 0.838 | 0.9626 |
| Random Forest | 99.95% | 0.912 | 0.768 | 0.834 | 0.9694 |
| XGBoost | 99.92% | 0.728 | 0.789 | 0.758 | 0.9700 |
| **XGBoost Tuned** | **99.95%** | **0.972** | **0.737** | **0.838** | **0.9758** |

> Best model: **XGBoost (Tuned)** with ROC-AUC of **0.9758**

---

## Tech Stack

`Python` `scikit-learn` `XGBoost` `imbalanced-learn` `Gradio` `pandas` `seaborn` `matplotlib` `joblib`

---

## Setup

```bash
pip install scikit-learn xgboost imbalanced-learn gradio pandas seaborn matplotlib
```

Run the notebook end-to-end, then launch the Gradio app:

```python
iface.launch(share=True)  # share=True for public URL in Colab
```

---

## Dataset

[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
284,807 transactions | 492 frauds | Features are PCA-transformed (V1–V28)
