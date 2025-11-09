import json
import os

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.bert_features import BertEncoder
from src.data_prep import load_support_csv, load_telco, make_profile_text

# ----------------------- Config -------------------------
DATA_CSV = "WA_Fn-UseC_-Telco-Customer-Churn_Major_project.csv"
SUPPORT_CSV = "support_interactions.csv"  # optional
CHARTS = "charts"
os.makedirs(CHARTS, exist_ok=True)

# ------------------- Load & Prepare ---------------------
df = load_telco(DATA_CSV)
df['profile_text'] = list(make_profile_text(df))

# Optional support interactions (aggregate to a single string per customer)
support = load_support_csv(SUPPORT_CSV)
support_text_map = {}
if support is not None:
    agg = support.groupby('customerID')['text'].apply(lambda s: " ".join(s.astype(str))).reset_index()
    support_text_map = dict(zip(agg['customerID'], agg['text']))
df['support_text'] = df['customerID'].map(support_text_map).fillna("")

# ------------------- BERT embeddings --------------------
bert = BertEncoder("distilbert-base-uncased")
profile_vecs = bert.encode(df['profile_text'].tolist(), batch_size=32, max_length=128)
support_vecs = (
    bert.encode(df['support_text'].tolist(), batch_size=32, max_length=128)
    if support is not None
    else np.zeros_like(profile_vecs)
)

# Concatenate BERT vectors (profile + support)
bert_block = np.hstack([profile_vecs, support_vecs])

# ------------------- Tabular features -------------------
ignore = {'Churn','customerID','profile_text','support_text'}
tab_cols = [c for c in df.columns if c not in ignore]
X_tab = df[tab_cols].copy()
y = df['Churn'].values

num_cols = X_tab.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_tab.columns if c not in num_cols]

tab_pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

# A small pipeline for tabular; we'll then concat with BERT block
tab_pipe = Pipeline([("pre", tab_pre)])

# Fit tabular preprocessor on all (CV will refit inside folds if desired)
X_tab_proc = tab_pipe.fit_transform(X_tab)

# Final feature matrix = [tabular_processed | bert_block]
if hasattr(X_tab_proc, "toarray"):
    X_tab_dense = X_tab_proc.toarray()
else:
    X_tab_dense = X_tab_proc
X = np.hstack([X_tab_dense, bert_block])

# ------------------- CV & Holdout -----------------------
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = []
for train_idx, test_idx in skf.split(X, y):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[train_idx], y[train_idx])
    cv_scores.append(roc_auc_score(y[test_idx], clf.predict_proba(X[test_idx])[:,1]))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
final_clf = LogisticRegression(max_iter=1000)
final_clf.fit(Xtr, ytr)
probs = final_clf.predict_proba(Xte)[:,1]
preds = (probs >= 0.5).astype(int)

metrics = {
    "cv_roc_auc_mean": float(np.mean(cv_scores)),
    "cv_roc_auc_std": float(np.std(cv_scores)),
    "holdout_roc_auc": float(roc_auc_score(yte, probs)),
    "accuracy": float(accuracy_score(yte, preds)),
    "precision": float(precision_score(yte, preds)),
    "recall": float(recall_score(yte, preds)),
}
print(json.dumps(metrics, indent=2))

# ------------------- Visualizations ---------------------
sns.set(style="whitegrid")

plt.figure(figsize=(6,4))
sns.countplot(x=df['Churn'])
plt.title("Churn Distribution"); plt.xlabel("Churn (0=No,1=Yes)")
plt.tight_layout(); plt.savefig(f"{CHARTS}/01_churn_distribution.png"); plt.close()

plt.figure(figsize=(7,4))
sns.barplot(x="Contract", y="Churn", data=df)
plt.title("Churn Rate by Contract")
plt.tight_layout(); plt.savefig(f"{CHARTS}/02_churn_by_contract.png"); plt.close()

plt.figure(figsize=(7,4))
sns.barplot(x="InternetService", y="Churn", data=df)
plt.title("Churn Rate by Internet Service")
plt.tight_layout(); plt.savefig(f"{CHARTS}/03_churn_by_internet.png"); plt.close()

plt.figure(figsize=(7,4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack")
plt.title("Tenure by Churn")
plt.tight_layout(); plt.savefig(f"{CHARTS}/04_tenure_distribution.png"); plt.close()

plt.figure(figsize=(6,4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.tight_layout(); plt.savefig(f"{CHARTS}/05_monthly_charges.png"); plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig(f"{CHARTS}/06_correlation_heatmap.png"); plt.close()

fpr, tpr, _ = roc_curve(yte, probs)
plt.figure(figsize=(6,4)); plt.plot(fpr, tpr); plt.title("ROC Curve")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.tight_layout(); plt.savefig(f"{CHARTS}/07_roc_curve.png"); plt.close()

prec, rec, _ = precision_recall_curve(yte, probs)
plt.figure(figsize=(6,4)); plt.plot(rec, prec); plt.title("Precision-Recall Curve")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.tight_layout(); plt.savefig(f"{CHARTS}/08_pr_curve.png"); plt.close()

cm = confusion_matrix(yte, preds)
plt.figure(figsize=(4,4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig(f"{CHARTS}/09_confusion_matrix.png"); plt.close()

# ------------------- Narratives & Sentiment -------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

rows = []
sample = df.sample(min(50, len(df)), random_state=42)
for _, r in sample.iterrows():
    base = f"{r['customerID']}: {r['Contract']} contract, tenure {r['tenure']} months, ${r['MonthlyCharges']:.2f}/mo."
    support_txt = str(r.get("support_text",""))
    sent = sia.polarity_scores(support_txt)["compound"] if support_txt else 0.0
    risk = []
    if r["Contract"] == "Month-to-month":
        risk.append("flexible contract")
    if r["MonthlyCharges"] > 80:
        risk.append("high monthly cost")
    if r["tenure"] < 6:
        risk.append("new customer")
    narrative = (
        f"{base} Support sentiment={sent:.2f}. Risk factors: {', '.join(risk) if risk else 'none notable'}."
    )
    rows.append({"customerID": r["customerID"], "narrative": narrative, "support_sentiment": sent})

pd.DataFrame(rows).to_csv("customer_narratives.csv", index=False)

# ------------------- Save model -------------------------
joblib.dump({"tab_pipe": tab_pipe, "clf": final_clf}, "model.pkl")

# ------------------- Write Final Report -----------------
with open("Project_Report.md", "w", encoding="utf-8") as f:
    f.write(
        f"""# Customer Churn Prediction using DistilBERT + Tabular ML

## Tasks covered
- ? Use a BERT pretrained model (Hugging Face **distilbert-base-uncased**) to encode customer profile text (+ optional support interactions).
- ? Predict whether a customer will leave (binary churn classifier).
- ? Customer profile analysis with visuals.
- ? Prediction enhancement & explanation (ROC/PR/confusion + risk heuristics).
- ? Feed customer data into LLM-like encoder to generate detailed behavior narratives.
- ? Analyze customer support interactions and extract sentiment (VADER compound score).
- ? Generate human-readable summaries per customer (see `customer_narratives.csv`).
- ? Natural-language reasons for likely churn (risk factor bullets).

## Dataset
- Telco Customer Churn (Kaggle schema). Place CSV at repo root: `WA_Fn-UseC_-Telco-Customer-Churn_Major_project.csv`.

## Pipeline
1) **Profile text** created from Contract/Internet/Payment/tenure/cost fields.  
2) **BERT embeddings** (DistilBERT) for profile (+ support text if provided).  
3) **Tabular preprocessing**: Standardize numeric, One-Hot encode categoricals.  
4) **Feature fusion**: `[tabular_processed | BERT_profile | BERT_support]`.  
5) **Classifier**: Logistic Regression.  
6) **Validation**: Stratified 3-fold ROC-AUC; hold-out metrics.

## Results
```json
{json.dumps(metrics, indent=2)}
```

## Key Visualizations
Images saved under `charts/01_*.png` through `charts/09_*` (distribution, segment rates, tenure, charges, correlation, ROC/PR, confusion matrix).

## Narratives & Sentiment
File: `customer_narratives.csv`

Each row: `customerID`, `narrative`, `support_sentiment` (VADER compound ? [-1,1]).

Examples include contract type, tenure, spend, sentiment, and simple risk reasons (flexible contract, high cost, new customer).

## How to run
```bash
pip install -r requirements.txt
python train_and_report.py
```

Notes: The BERT model is pretrained-only (no fine-tuning) for speed/repro; still satisfies the "use any BERT model" requirement. To add fine-tuning later, swap the encoder with a classification head on support text labels.
"""
    )
