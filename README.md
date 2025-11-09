# Customer Churn Prediction -- DistilBERT + Tabular

Predicts **telecom churn** by combining DistilBERT embeddings of short profile text (plus optional support transcripts) with classic tabular features and a Logistic Regression classifier.

## Structure
- `train.py` -- trains the pipeline, runs CV/holdout evaluation, emits charts, metrics, narratives, and `model.pkl`.
- `src/`
  - `bert_features.py` -- lightweight DistilBERT encoder wrapper.
  - `data_prep.py` -- Telco CSV cleaning + optional support aggregation.
  - `visualize.py` -- shared plotting helpers writing to `charts/*.png`.
- `charts/` -- populated after training (ROC, PR, confusion matrix, etc.).
- `artifacts/metrics.json` -- metrics dump from the last training run.
- `customer_narratives.csv` -- per-customer summaries + VADER sentiment (created by `train.py`).
- `model.pkl` -- serialized tabular pipeline + classifier.
- `Project_Report.md` -- Markdown report assembled from metrics, charts, and sample narratives.

## Run locally
1. Place `WA_Fn-UseC_-Telco-Customer-Churn_Major_project.csv` at repo root (optional `support_interactions.csv` for transcripts).
2. Install deps and train:
   ```bash
   pip install -r requirements.txt
   python train.py
   ```
3. Rebuild the Markdown report any time by rerunning `python train.py` (it refreshes metrics/charts/narratives that power `Project_Report.md`).

## Outputs
- `charts/01_*.png` ... `charts/09_*.png`
- `artifacts/metrics.json`
- `customer_narratives.csv`
- `model.pkl`
- `Project_Report.md` (embedded images + metrics + top narratives)

## Notes
- DistilBERT runs in inference mode only (no fine-tuning) for quick iteration.
- If `artifacts/metrics.json` or `charts/` is missing, run `python train.py` to regenerate before opening the report.
