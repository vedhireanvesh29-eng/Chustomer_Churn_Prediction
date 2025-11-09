# Customer Churn Prediction — BERT + Tabular

This repo predicts telecom churn by combining:

- DistilBERT embeddings of a short customer profile text (+ optional support-interaction text)
- Tabular features (tenure, cost, contract, etc.)
- A Logistic Regression classifier

## What this project demonstrates
- ? Use a Hugging Face BERT model (distilbert-base-uncased)
- ? Predict churn (binary classification)
- ? Profile analysis with visualizations
- ? Narratives per customer + sentiment over support interactions
- ? Clear report with embedded charts (`Project_Report.md`)

## Quick start
```bash
pip install -r requirements.txt
# Put Telco CSV at repo root with this exact name:
#   WA_Fn-UseC_-Telco-Customer-Churn_Major_project.csv
python train_and_report.py
```

Outputs:

- `charts/01_...png` … `charts/09_confusion_matrix.png`
- `model.pkl`
- `customer_narratives.csv`
- `Project_Report.md` (includes metrics + embedded charts)

## Optional: support interactions
If you have a file `support_interactions.csv` with:

```
customerID,text
7590-VHVEG,"agent chat text ... "
5575-GNVDE,"email/notes ..."
```

the pipeline will:

1. aggregate text per customer,
2. encode with BERT,
3. compute VADER sentiment, and
4. incorporate that signal in narratives + features.
