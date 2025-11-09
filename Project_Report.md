# Customer Churn Prediction -- DistilBERT + Tabular ML (Final Report)

This report documents the complete churn prediction pipeline, results, and visual analysis. It uses a **Hugging Face DistilBERT** encoder for compact text features and classic **tabular ML**.

##  Tasks & Requirements Covered
- Use a BERT pretrained model from Hugging Face (**distilbert-base-uncased**)
- Predict whether a customer will leave (binary churn)
- Customer profile analysis with visualizations
- Prediction enhancement & explanation (**ROC**, **PR**, **Confusion Matrix**, correlations)
- Feed customer data into LLM-like encoder to generate detailed behavior narratives
- Analyze customer support interactions for sentiment, pain points, satisfaction levels
- Generate human-readable summaries of each customer's journey and risk factors
- Generate natural language explanations for why a customer is likely to churn

##  Dataset
Telco Customer Churn (Kaggle schema). Place the CSV in repo root:
`WA_Fn-UseC_-Telco-Customer-Churn_Major_project.csv`

##  Pipeline Overview
1. **Profile text**: compose a short behavior string from contract, internet, payment, tenure, charges.
2. **BERT embeddings**: encode profile (and optional support text) using DistilBERT.
3. **Tabular preprocessing**: scale numeric + one-hot categorical.
4. **Feature fusion**: concatenate `[tabular_processed | BERT_profile | BERT_support]`.
5. **Classifier**: Logistic Regression.
6. **Validation**: Stratified 3-fold ROC-AUC + holdout metrics.

##  Results
_Run `python train.py` to generate metrics (`artifacts/metrics.json`)._

##  Visual Analysis
## Narratives, Sentiment & Explanations
- `customer_narratives.csv` contains a per-customer summary (contract, tenure, spend) and **VADER** sentiment over support interactions (if provided).
- **Risk factors** surfaced in plain English:
  - flexible month-to-month contract
  - high monthly cost
  - very new customer

_Run `python train.py` to generate `customer_narratives.csv` and re-open this report._

##  Reproducibility
```bash
pip install -r requirements.txt
python train.py          # trains, saves metrics/charts/model/narratives
```

## Next Enhancements
- SHAP/Permutation importance for per-feature attributions
- Try XGBoost/LightGBM alongside logistic regression
- Calibrate threshold for business precision/recall targets
