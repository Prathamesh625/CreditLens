"""Data loading and a synthetic data generator.

The generator lets the project run end to end without downloading anything.
To use a real dataset (Home Credit, German Credit, ...), put a CSV in data/raw/
with the columns listed in src/config.py and pass --data-path to src.train.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES, NUMERIC, SEED, TARGET


def generate_synthetic(n: int = 20000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    gender = rng.choice(["F", "M"], n)
    income = np.clip(rng.lognormal(10.8, 0.45, n), 12000, 400000).round(0)
    loan_amount = np.clip(income * rng.uniform(0.05, 1.4, n), 1000, 500000).round(0)
    employment_years = np.clip(rng.gamma(2.5, 3.0, n), 0, age - 18).round(1)
    credit_history_length = np.clip(rng.gamma(3.0, 3.0, n), 0, age - 18).round(1)
    num_open_accounts = rng.poisson(4, n)
    num_late_payments = rng.poisson(0.8, n)
    home_ownership = rng.choice(["RENT", "OWN", "MORTGAGE"], n, p=[0.45, 0.2, 0.35])
    debt_to_income = np.clip(loan_amount / income + rng.normal(0, 0.05, n), 0, 3).round(3)

    logit = (
        -3.9
        + 2.2 * debt_to_income
        + 0.40 * num_late_payments
        - 0.05 * employment_years
        - 0.04 * credit_history_length
        + 0.03 * num_open_accounts
        + 0.40 * (home_ownership == "RENT")
        - 0.30 * (home_ownership == "OWN")
        # non-linear effects a linear model cannot capture
        + 1.1 * (debt_to_income > 0.8)
        + 0.30 * num_late_payments * (home_ownership == "RENT")
        - 0.9 * ((credit_history_length > 10) & (num_late_payments == 0))
        + 0.6 * ((income < 25000) & (loan_amount > 15000))
        + rng.normal(0, 0.5, n)
    )
    p = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, p)

    return pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "income": income,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
            "credit_history_length": credit_history_length,
            "num_open_accounts": num_open_accounts,
            "num_late_payments": num_late_payments,
            "debt_to_income": debt_to_income,
            "home_ownership": home_ownership,
            TARGET: default,
        }
    )


def load_data(path: str | None = None, n_rows: int = 20000, seed: int = SEED) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)
    else:
        df = generate_synthetic(n_rows, seed)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def basic_report(df: pd.DataFrame) -> dict:
    """Small EDA summary printed at train time (full EDA lives in notebooks/)."""
    return {
        "rows": int(len(df)),
        "default_rate": round(float(df[TARGET].mean()), 4),
        "missing_values": {c: int(v) for c, v in df[FEATURES].isna().sum().items() if v > 0},
        "numeric_summary": df[NUMERIC].describe().round(2).to_dict(),
    }
