"""Train baseline + LightGBM, evaluate, run the fairness audit, save artifacts.

Usage:
    python -m src.train                          # synthetic data
    python -m src.train --data-path data/raw/credit.csv --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, FEATURES, MODEL_VERSION, NUMERIC, SEED, TARGET
from .data import basic_report, load_data
from .fairness import audit
from .metrics import evaluate


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", num, NUMERIC), ("cat", cat, CATEGORICAL)], sparse_threshold=0
    )


def train(
    data_path: str | None = None,
    n_rows: int = 20000,
    seed: int = SEED,
    out_dir: str = "models",
    verbose: bool = True,
) -> dict:
    df = load_data(data_path, n_rows, seed)
    if verbose:
        rep = {k: v for k, v in basic_report(df).items() if k != "numeric_summary"}
        print("Data report:", json.dumps(rep))

    # 60 / 20 / 20 stratified split. Preprocessing is fit on TRAIN only (no leakage).
    train_df, tmp = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=seed)
    val_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp[TARGET], random_state=seed)

    pre = build_preprocessor().fit(train_df[FEATURES])
    X_tr, X_val, X_te = (pre.transform(d[FEATURES]) for d in (train_df, val_df, test_df))
    y_tr, y_val, y_te = (d[TARGET].values for d in (train_df, val_df, test_df))

    # Baseline: logistic regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed).fit(X_tr, y_tr)

    # Final: LightGBM with early stopping on the validation set
    gbm = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    gbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    metrics = {
        "validation": {
            "logistic_regression": evaluate(y_val, lr.predict_proba(X_val)[:, 1]),
            "lightgbm": evaluate(y_val, gbm.predict_proba(X_val)[:, 1]),
        },
        # Test set is touched once, for the final report.
        "test": {
            "logistic_regression": evaluate(y_te, lr.predict_proba(X_te)[:, 1]),
            "lightgbm": evaluate(y_te, gbm.predict_proba(X_te)[:, 1]),
        },
        "rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "default_rate": round(float(df[TARGET].mean()), 4),
        "best_iteration": int(gbm.best_iteration_ or gbm.n_estimators),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "seed": seed,
    }

    fairness = audit(test_df, gbm.predict_proba(X_te)[:, 1])

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(
        {"preprocessor": pre, "lightgbm": gbm, "baseline": lr, "version": MODEL_VERSION},
        os.path.join(out_dir, "model.joblib"),
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "fairness.json"), "w") as f:
        json.dump(fairness, f, indent=2)

    if verbose:
        print(json.dumps(metrics["test"], indent=2))
        print("Fairness (approval-rate gaps):",
              {k: v.get("_approval_rate_gap") for k, v in fairness["groups"].items()})
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default=None)
    p.add_argument("--n-rows", type=int, default=20000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-dir", default="models")
    a = p.parse_args()
    train(a.data_path, a.n_rows, a.seed, a.out_dir)
