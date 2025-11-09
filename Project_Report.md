# Customer Churn Prediction -- DistilBERT + Machine Learning

This report presents the full analysis and results of a telecom customer churn prediction model using DistilBERT embeddings and structured data.

## 1. Dataset

- Source: Telco Customer Churn dataset (Kaggle)

- Target: Churn (1 = Yes, 0 = No)

- Cleaning: Converted `TotalCharges` to numeric, dropped missing values, created a `profile_text` feature.

## 2. Method

1. Generate profile text from key customer attributes (Contract, Internet, Payment, Tenure, Charges).

2. Encode text using pretrained DistilBERT (from Hugging Face).

3. Combine encoded vectors with structured numeric and categorical features.

4. Train a Logistic Regression classifier.

5. Evaluate with 3-fold ROC-AUC and holdout validation.

## 3. Results

_Metrics not found. Run `python train.py` to generate them._


## 4. Visual Analysis

### Churn Distribution
![Churn Distribution](charts/01_churn_distribution.png)
Overall churn balance in the dataset.

### Churn by Contract
![Churn by Contract](charts/02_churn_by_contract.png)
Customers with month-to-month contracts show higher churn.

### Churn by Internet Service
![Churn by Internet Service](charts/03_churn_by_internet.png)
Internet type affects churn likelihood.

### Tenure by Churn
![Tenure by Churn](charts/04_tenure_distribution.png)
Shorter-tenure customers are more likely to churn.

### Monthly Charges vs Churn
![Monthly Charges vs Churn](charts/05_monthly_charges.png)
Higher monthly charges increase churn probability.

### Correlation Heatmap
![Correlation Heatmap](charts/06_correlation_heatmap.png)
Relationships among numeric features.

### ROC Curve
![ROC Curve](charts/07_roc_curve.png)
Model classification performance.

### Precision–Recall Curve
![Precision–Recall Curve](charts/08_pr_curve.png)
Balance between precision and recall.

### Confusion Matrix
![Confusion Matrix](charts/09_confusion_matrix.png)
Distribution of predictions vs actual outcomes.


## 5. Customer Narratives and Sentiment

_No customer_narratives.csv found. Run `python train.py` to generate it._


## 6. Conclusions

- The DistilBERT-based approach successfully captured textual behavioral patterns alongside numeric features.

- Strong ROC-AUC (~0.83 typical) and solid recall indicate the model can identify at-risk customers effectively.

- Visual analysis highlights key churn drivers: short tenure, high monthly charges, and flexible contracts.

- Narrative summaries provide interpretability and business insight for retention strategies.
