import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def _ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def churn_distribution(df, outdir="charts"):
    _ensure(outdir)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["Churn"])
    plt.title("Churn Distribution")
    plt.xlabel("Churn (0=No,1=Yes)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/01_churn_distribution.png")
    plt.close()


def churn_by_contract(df, outdir="charts"):
    _ensure(outdir)
    plt.figure(figsize=(7, 4))
    sns.barplot(x="Contract", y="Churn", data=df)
    plt.title("Churn Rate by Contract")
    plt.tight_layout()
    plt.savefig(f"{outdir}/02_churn_by_contract.png")
    plt.close()


def churn_by_internet(df, outdir="charts"):
    _ensure(outdir)
    plt.figure(figsize=(7, 4))
    sns.barplot(x="InternetService", y="Churn", data=df)
    plt.title("Churn Rate by Internet Service")
    plt.tight_layout()
    plt.savefig(f"{outdir}/03_churn_by_internet.png")
    plt.close()


def tenure_distribution(df, outdir="charts"):
    _ensure(outdir)
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30)
    plt.title("Tenure by Churn")
    plt.tight_layout()
    plt.savefig(f"{outdir}/04_tenure_distribution.png")
    plt.close()


def monthly_box(df, outdir="charts"):
    _ensure(outdir)
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
    plt.title("Monthly Charges vs Churn")
    plt.tight_layout()
    plt.savefig(f"{outdir}/05_monthly_charges.png")
    plt.close()


def corr_heatmap(df, outdir="charts"):
    _ensure(outdir)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{outdir}/06_correlation_heatmap.png")
    plt.close()


def roc_plot(y_true, y_prob, outdir="charts"):
    _ensure(outdir)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(f"{outdir}/07_roc_curve.png")
    plt.close()


def pr_plot(y_true, y_prob, outdir="charts"):
    _ensure(outdir)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(f"{outdir}/08_pr_curve.png")
    plt.close()


def confusion(y_true, y_pred, outdir="charts"):
    _ensure(outdir)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{outdir}/09_confusion_matrix.png")
    plt.close()


def generate_all(df, y_true, y_prob, y_pred, outdir="charts"):
    churn_distribution(df, outdir)
    churn_by_contract(df, outdir)
    churn_by_internet(df, outdir)
    tenure_distribution(df, outdir)
    monthly_box(df, outdir)
    corr_heatmap(df, outdir)
    roc_plot(y_true, y_prob, outdir)
    pr_plot(y_true, y_prob, outdir)
    confusion(y_true, y_pred, outdir)
