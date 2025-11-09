import pandas as pd
import numpy as np


def load_telco(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return df


def make_profile_text(df: pd.DataFrame) -> pd.Series:
    parts = []
    parts.append("Contract=" + df["Contract"].astype(str))
    parts.append("Internet=" + df["InternetService"].astype(str))
    parts.append("Payment=" + df["PaymentMethod"].astype(str))
    parts.append("Tenure=" + df["tenure"].astype(str) + "m")
    parts.append("Monthly=" + df["MonthlyCharges"].round(2).astype(str))
    parts.append("Total=" + df["TotalCharges"].round(2).astype(str))
    return (" | ".join(p) for p in zip(*parts))


def load_support_csv(optional_path="support_interactions.csv"):
    try:
        sup = pd.read_csv(optional_path)
        needed = {"customerID", "text"}
        if not needed.issubset(set(sup.columns)):
            return None
        return sup
    except Exception:
        return None
