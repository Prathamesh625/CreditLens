<div align="center">



\# 📉 explainable-credit-risk



\*\*Loan default prediction with a reason for every decision: a scikit-learn / LightGBM model served through FastAPI, with SHAP explanations and a live dashboard.\*\*



!\[Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python\&logoColor=white)

!\[FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi\&logoColor=white)

!\[scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn\&logoColor=white)

!\[PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql\&logoColor=white)

!\[Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker\&logoColor=white)

!\[CI](https://github.com/Prathamesh625/explainable-credit-risk/actions/workflows/ci.yml/badge.svg)

!\[License](https://img.shields.io/badge/license-MIT-green)



\[Overview](#-overview) · \[Results](#-results) · \[Architecture](#-architecture) · \[Quick start](#-quick-start) · \[API](#-api-overview) · \[Methodology](#-methodology) · \[Limitations](#-limitations)



</div>



> ⚠️ \*\*Learning project.\*\* Built on a public dataset to demonstrate an end-to-end ML workflow. It is not a real credit decisioning system and must not be used to make real lending decisions.



\---



\## 📌 Overview



Most loan-default projects stop at a notebook and a single accuracy number. This one covers the full path:



1\. \*\*Analyse\*\* the data (missing values, outliers, leakage risks)

2\. \*\*Compare\*\* a simple baseline against a stronger model

3\. \*\*Evaluate\*\* with metrics that suit imbalanced data

4\. \*\*Explain\*\* every prediction with SHAP, since lenders must justify decisions

5\. \*\*Check fairness\*\* across groups

6\. \*\*Serve\*\* the model through a validated API

7\. \*\*Record\*\* every prediction, model version and explanation for audit



\## ✨ Features



\- \*\*Risk score API:\*\* `/predict` returns a default probability, a risk band and the \*\*top 3 risk drivers\*\*

\- \*\*Baseline vs. final model:\*\* logistic regression compared with LightGBM, with numbers side by side

\- \*\*Imbalance-aware evaluation:\*\* ROC-AUC, precision-recall, KS statistic

\- \*\*SHAP explainability:\*\* per-prediction reasons and a global feature-importance view

\- \*\*Fairness check:\*\* approval and error rates compared across groups

\- \*\*Audit trail:\*\* inputs, score, reasons and model version stored in PostgreSQL

\- \*\*Dashboard:\*\* enter applicant details and see the score and a Chart.js bar chart of risk drivers

\- \*\*Reproducible:\*\* fixed random seeds, pinned requirements, Docker, pytest and CI



\## 📊 Results



> Fill these in only with numbers you measured on the held-out \*\*test set\*\*, and note the command or notebook for each.



\*\*Dataset:\*\* \_TODO: name (e.g. Home Credit Default Risk), number of rows, default rate\_



| Model | ROC-AUC | PR-AUC | KS | Notes |

|---|---|---|---|---|

| Logistic regression (baseline) | \_TODO\_ | \_TODO\_ | \_TODO\_ | Scaled, class-weighted |

| LightGBM (final) | \_TODO\_ | \_TODO\_ | \_TODO\_ | Tuned with cross-validation |



| Serving metric | Result | How measured |

|---|---|---|

| `/predict` p95 latency | \_TODO\_ | Locust, \_N\_ users |

| Test count / coverage | \_TODO\_ | `pytest --cov` |



\*\*Top global drivers (SHAP):\*\* \_TODO: paste your top 5 features and a one-line interpretation\_



\## 🧱 Architecture



```mermaid

flowchart LR

&#x20;   U\[Dashboard / Client] -->|JSON| A\[FastAPI]

&#x20;   A --> V\[Pydantic validation]

&#x20;   V --> M\[Model pipeline<br/>preprocess + LightGBM]

&#x20;   M --> X\[SHAP explainer<br/>top 3 drivers]

&#x20;   X --> A

&#x20;   A --> D\[(PostgreSQL<br/>predictions + audit log)]

&#x20;   T\[Training pipeline] -->|versioned artifact| M

```



\*\*Training flow:\*\* raw data → cleaning and feature engineering → train/validation/test split → baseline → tuned model → evaluation → saved artifact with a version tag.



\## 🧰 Tech stack



| Layer | Tools |

|---|---|

| Data and ML | Python, pandas, NumPy, scikit-learn, LightGBM, SHAP |

| Experiment tracking | MLflow (optional) |

| API | FastAPI, Pydantic |

| Storage | PostgreSQL, SQLAlchemy |

| Frontend | HTML, CSS, JavaScript, Chart.js |

| DevOps | Docker, GitHub Actions, pytest |



\## 🚀 Quick start



\*\*Prerequisites:\*\* Docker and Docker Compose (or Python 3.12 for local runs)



```bash

\# 1. Clone

git clone https://github.com/Prathamesh625/explainable-credit-risk.git

cd explainable-credit-risk



\# 2. Get the data (see data/README.md for the download link and licence)

\#    Place the files in data/raw/



\# 3. Train the model

python -m src.train --seed 42



\# 4. Start the API and database

docker compose up --build

```



\- API docs (Swagger): http://localhost:8000/docs

\- Dashboard: http://localhost:8000/dashboard



\## 📡 API overview



| Method | Endpoint | Description |

|---|---|---|

| `POST` | `/predict` | Score an applicant and return top-3 reasons |

| `GET` | `/predictions/{id}` | Retrieve a stored prediction and explanation |

| `GET` | `/model/info` | Active model version, training date, metrics |

| `GET` | `/health` | Liveness check |



\*\*Example request\*\*



```bash

curl -X POST http://localhost:8000/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{"age": 34, "income": 52000, "loan\_amount": 15000, "employment\_years": 6, "credit\_history\_length": 8}'

```



\*\*Example response\*\* \_(illustrative: replace with a real response from your model)\_



```json

{

&#x20; "prediction\_id": 1042,

&#x20; "default\_probability": 0.18,

&#x20; "risk\_band": "medium",

&#x20; "top\_reasons": \[

&#x20;   {"feature": "loan\_amount", "impact": "+0.07"},

&#x20;   {"feature": "credit\_history\_length", "impact": "-0.05"},

&#x20;   {"feature": "income", "impact": "-0.03"}

&#x20; ],

&#x20; "model\_version": "v1.0.0"

}

```



\## 🔬 Methodology



\*\*1. Data analysis (pandas)\*\*

Missing values, outliers, class balance, feature distributions and correlations. Charts are in `notebooks/01\_eda.ipynb`.



\*\*2. Avoiding data leakage\*\*

Preprocessing is fit only on the training split and wrapped in a scikit-learn `Pipeline`. Features that would not exist at application time are removed. \_TODO: list which columns you dropped and why.\_



\*\*3. Handling imbalance\*\*

Class weights were used, and metrics focus on PR-AUC and KS rather than accuracy. \_TODO: note anything else you tried (e.g. SMOTE) and what happened.\_



\*\*4. Model comparison\*\*

A logistic regression baseline was trained first, then LightGBM. The final model is only used if it beats the baseline on the validation set, and the comparison is reported on the untouched test set.



\*\*5. Explainability\*\*

SHAP values give each prediction its top drivers, and global SHAP plots show what the model relies on overall.



\*\*6. Fairness check\*\*

Approval rate and error rates (false positive and false negative) are compared across groups such as age band or gender, where the dataset provides them. \_TODO: summarise what you found.\_



\## 🧪 Testing



```bash

pytest --cov=src --cov=app

```



| Test | What it covers |

|---|---|

| Input validation | Bad or missing fields return clear 422 errors |

| Prediction contract | Score is between 0 and 1, and exactly 3 reasons are returned |

| Reproducibility | Same input and model version give the same output |

| Persistence | Every prediction is stored with its model version |



\## 🗂️ Project structure



```

explainable-credit-risk/

├── app/              # FastAPI app, routes, schemas, DB models

│   └── static/       # dashboard HTML/CSS/JS (Chart.js)

├── src/              # training, features, evaluation, explain

├── notebooks/        # EDA and model comparison

├── models/           # versioned model artifacts

├── data/             # raw/ and processed/ (not committed)

├── tests/

├── docker-compose.yml

├── Dockerfile

├── requirements.txt

└── README.md

```



\## ⚠️ Limitations



Be upfront about these. It shows engineering maturity.



\- Trained on a \*\*public dataset\*\*, so real-world performance would differ.

\- SHAP shows what the model used, \*\*not causation\*\*.

\- Fairness checks cover only the attributes available in the data.

\- No drift monitoring or automated retraining yet.

\- Not suitable for real lending decisions.



\## 🐞 Problems I faced



> Replace with your real notes.



\- \_Example: first model scored suspiciously high AUC because of a leaked column.\_

\- \_Example: probabilities were poorly calibrated until I checked a calibration curve.\_



\## 🗺️ Roadmap



\- \[ ] Probability calibration

\- \[ ] Drift detection (PSI) and a retraining trigger

\- \[ ] Model registry with MLflow

\- \[ ] Batch scoring endpoint



\## 📄 License



MIT. See \[LICENSE](LICENSE). Dataset licence: see `data/README.md`.



\---



<div align="center">



Built by \[Prathamesh Thorat](https://github.com/Prathamesh625) · \[LinkedIn](https://linkedin.com/in/thorat-prathamesh)



</div>

