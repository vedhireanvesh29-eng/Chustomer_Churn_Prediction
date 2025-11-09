import json
import os
from pathlib import Path

METRICS = Path("artifacts/metrics.json")
CHARTS = Path("charts")
OUT = Path("Project_Report.md")


def img(name: str) -> str:
    return f"![{name}]({CHARTS.as_posix()}/{name})"


def main() -> None:
    if not METRICS.exists():
        raise SystemExit("Run `python train.py` first; metrics file not found.")

    with METRICS.open() as fh:
        metrics = json.load(fh)

    md = f"""# Customer Churn Prediction — DistilBERT + Tabular ML (Final Report)

## ? Tasks & Requirements Covered
- Use a BERT pretrained model from Hugging Face (**distilbert-base-uncased**)
- Predict whether a customer will leave (binary churn)
- Customer profile analysis with visualizations
- Prediction enhancement & explanation (ROC/PR/Confusion, correlation)
- Feed customer data into LLM-like encoder to generate narratives
- Analyze support interactions for sentiment, pain points, satisfaction
- Generate human-readable summaries (journey + risk factors)
- Natural-language explanations for why a customer is likely to churn

## ?? Dataset
Telco Customer Churn (Kaggle schema). Place the CSV in repo root:
`WA_Fn-UseC_-Telco-Customer-Churn_Major_project.csv`.

## ?? Pipeline Overview
1. **Profile text**: compose a short behavior string from contract, internet, payment, tenure, charges.
2. **BERT embeddings**: encode profile (and optional support text) using DistilBERT.
3. **Tabular preprocessing**: scale numeric + one-hot categorical.
4. **Feature fusion**: concatenate `[tabular_processed | BERT_profile | BERT_support]`.
5. **Classifier**: Logistic Regression.
6. **Validation**: Stratified 3-fold ROC-AUC + holdout metrics.

## ?? Results
```json
{json.dumps(metrics, indent=2)}
```

## ?? Visual Analysis (with images)
1. Churn Distribution
{img("01_churn_distribution.png")}

2. Churn by Contract
{img("02_churn_by_contract.png")}

3. Churn by Internet Service
{img("03_churn_by_internet.png")}

4. Tenure Distribution by Churn
{img("04_tenure_distribution.png")}

5. Monthly Charges by Churn
{img("05_monthly_charges.png")}

6. Correlation Heatmap
{img("06_correlation_heatmap.png")}

7. ROC Curve
{img("07_roc_curve.png")}

8. Precision–Recall Curve
{img("08_pr_curve.png")}

9. Confusion Matrix
{img("09_confusion_matrix.png")}

## ??? Narratives, Sentiment & Explanations
`customer_narratives.csv` : per-customer summary (contract, tenure, spend) + VADER sentiment over support text.

Risk factors highlighted in natural language: flexible month-to-month contract, high monthly cost, very new customer.

## ?? How to Reproduce
```bash
pip install -r requirements.txt
python train.py          # trains, saves metrics/charts/model/narratives
python make_report.py    # builds this report with embedded charts + metrics
```

## ?? Next Enhancements
- Add SHAP/Permutation importance for per-feature attributions.
- Try XGBoost/LightGBM alongside logistic regression.
- Calibrate threshold for business precision/recall targets.
"""

    OUT.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
