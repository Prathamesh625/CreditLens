"""Shared constants for training and serving."""

SEED = 42
MODEL_VERSION = "1.0.0"

TARGET = "default"
NUMERIC = [
    "age",
    "income",
    "loan_amount",
    "employment_years",
    "credit_history_length",
    "num_open_accounts",
    "num_late_payments",
    "debt_to_income",
]
CATEGORICAL = ["home_ownership"]
FEATURES = NUMERIC + CATEGORICAL

# Used ONLY for the fairness audit, never as model features.
SENSITIVE = ["gender", "age"]

# Risk bands on default probability and the decision threshold used in the fairness audit.
BAND_LOW = 0.10
BAND_HIGH = 0.25
DECISION_THRESHOLD = 0.25
