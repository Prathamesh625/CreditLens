import pytest
from fastapi.testclient import TestClient

from src.train import train


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    train(n_rows=4000, seed=7, out_dir=str(d), verbose=False)
    return str(d)


@pytest.fixture()
def client(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", model_dir)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from app.main import app

    with TestClient(app) as c:
        yield c


GOOD = {
    "age": 40, "income": 90000, "loan_amount": 8000, "employment_years": 12,
    "credit_history_length": 15, "num_open_accounts": 3, "num_late_payments": 0,
    "debt_to_income": 0.09, "home_ownership": "OWN",
}
RISKY = {
    "age": 25, "income": 20000, "loan_amount": 25000, "employment_years": 0.5,
    "credit_history_length": 1, "num_open_accounts": 8, "num_late_payments": 5,
    "debt_to_income": 1.25, "home_ownership": "RENT",
}
