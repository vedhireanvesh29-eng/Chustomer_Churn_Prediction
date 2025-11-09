import json
import os

import joblib
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.bert_features import BertEncoder
from src.data_prep import load_support_csv, load_telco, make_profile_text
from src.visualize import generate_all

DATA_CSV = "WA_Fn-UseC_-Telco-Customer-Churn_Major_project.csv"
SUPPORT_CSV = "support_interactions.csv"
CHARTS = "charts"


def main():
    df = load_telco(DATA_CSV)
    df["profile_text"] = list(make_profile_text(df))
    support = load_support_csv(SUPPORT_CSV)

    support_map = {}
    if support is not None:
        agg = support.groupby("customerID")["text"].apply(lambda s: " ".join(s.astype(str))).reset_index()
        support_map = dict(zip(agg["customerID"], agg["text"]))
    df["support_text"] = df["customerID"].map(support_map).fillna("")

    bert = BertEncoder("distilbert-base-uncased")
    prof_vec = bert.encode(df["profile_text"].tolist(), batch_size=32, max_length=128)
    supp_vec = (
        bert.encode(df["support_text"].tolist(), batch_size=32, max_length=128)
        if support is not None
        else np.zeros_like(prof_vec)
    )
    bert_block = np.hstack([prof_vec, supp_vec])

    ignore = {"Churn", "customerID", "profile_text", "support_text"}
    tab_cols = [c for c in df.columns if c not in ignore]
    X_tab = df[tab_cols].copy()
    y = df["Churn"].values

    num_cols = X_tab.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_tab.columns if c not in num_cols]

    tab_pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    tab_pipe = Pipeline([("pre", tab_pre)])
    X_tab_proc = tab_pipe.fit_transform(X_tab)
    X_tab_dense = X_tab_proc.toarray() if hasattr(X_tab_proc, "toarray") else X_tab_proc

    X = np.hstack([X_tab_dense, bert_block])

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    for tr, te in skf.split(X, y):
        clf_cv = LogisticRegression(max_iter=1000)
        clf_cv.fit(X[tr], y[tr])
        cv_scores.append(roc_auc_score(y[te], clf_cv.predict_proba(X[te])[:, 1]))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    yprob = clf.predict_proba(Xte)[:, 1]
    yhat = (yprob >= 0.5).astype(int)

    metrics = {
        "cv_roc_auc_mean": float(np.mean(cv_scores)),
        "cv_roc_auc_std": float(np.std(cv_scores)),
        "holdout_roc_auc": float(roc_auc_score(yte, yprob)),
        "accuracy": float(accuracy_score(yte, yhat)),
        "precision": float(precision_score(yte, yhat)),
        "recall": float(recall_score(yte, yhat)),
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    generate_all(df, yte, yprob, yhat, outdir=CHARTS)

    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()

    rows = []
    sample = df.sample(min(200, len(df)), random_state=42)
    for _, r in sample.iterrows():
        support_txt = str(r.get("support_text", "")).strip()
        sent = sia.polarity_scores(support_txt)["compound"] if support_txt else 0.0
        risk = []
        if r["Contract"] == "Month-to-month":
            risk.append("flexible contract")
        if r["MonthlyCharges"] > 80:
            risk.append("high monthly cost")
        if r["tenure"] < 6:
            risk.append("new customer")
        narrative = (
            f"{r['customerID']}: {r['Contract']} contract, tenure {r['tenure']}m, ${r['MonthlyCharges']:.2f}/mo. "
            f"Sentiment={sent:.2f}. Risks: {', '.join(risk) if risk else 'none'}."
        )
        rows.append({"customerID": r["customerID"], "narrative": narrative, "support_sentiment": sent})

    pd.DataFrame(rows).to_csv("customer_narratives.csv", index=False)

    joblib.dump({"tab_pipe": tab_pipe, "clf": clf}, "model.pkl")

    print("Training complete. Metrics saved to artifacts/metrics.json; charts in charts/; narratives CSV and model.pkl created.")


if __name__ == "__main__":
    main()
