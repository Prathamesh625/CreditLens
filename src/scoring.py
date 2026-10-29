"""Load the trained model and produce a score plus SHAP-based reasons."""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import shap

from .config import BAND_HIGH, BAND_LOW, CATEGORICAL, FEATURES

READABLE = {
    "age": "Age",
    "income": "Annual income",
    "loan_amount": "Loan amount",
    "employment_years": "Years employed",
    "credit_history_length": "Credit history length",
    "num_open_accounts": "Open accounts",
    "num_late_payments": "Late payments",
    "debt_to_income": "Debt-to-income ratio",
    "home_ownership": "Home ownership",
}


def risk_band(p: float) -> str:
    return "low" if p < BAND_LOW else "medium" if p < BAND_HIGH else "high"


class Scorer:
    def __init__(self, model_dir: str):
        art = joblib.load(os.path.join(model_dir, "model.joblib"))
        self.pre = art["preprocessor"]
        self.model = art["lightgbm"]
        self.version = art["version"]
        self.explainer = shap.TreeExplainer(self.model)
        self.out_names = list(self.pre.get_feature_names_out())
        with open(os.path.join(model_dir, "metrics.json")) as f:
            self.metrics = json.load(f)
        # Map every transformed column back to its original feature
        self.origin = [self._origin(n) for n in self.out_names]

    @staticmethod
    def _origin(name: str) -> str:
        kind, rest = name.split("__", 1)
        if kind == "num":
            return rest
        for c in CATEGORICAL:
            if rest.startswith(c):
                return c
        return rest

    def _shap_row(self, X: np.ndarray) -> np.ndarray:
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):  # older shap: [class0, class1]
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, features, classes)
            sv = sv[:, :, 1]
        return sv[0]

    def predict(self, row: dict, top_k: int = 3) -> dict:
        df = pd.DataFrame([row])[FEATURES]
        X = self.pre.transform(df)
        prob = float(self.model.predict_proba(X)[0, 1])
        sv = self._shap_row(X)

        by_feature: dict[str, float] = {}
        for impact, feat in zip(sv, self.origin):
            by_feature[feat] = by_feature.get(feat, 0.0) + float(impact)

        top = sorted(by_feature.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        reasons = [
            {
                "feature": f,
                "label": READABLE.get(f, f),
                "value": row[f],
                "impact": round(v, 4),
                "direction": "increases risk" if v > 0 else "decreases risk",
            }
            for f, v in top
        ]
        return {
            "default_probability": round(prob, 4),
            "risk_band": risk_band(prob),
            "top_reasons": reasons,
            "model_version": self.version,
        }
