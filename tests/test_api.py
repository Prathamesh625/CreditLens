from .conftest import GOOD, RISKY


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_contract(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["risk_band"] in {"low", "medium", "high"}
    assert len(body["top_reasons"]) == 3
    assert body["model_version"]


def test_risky_scores_higher_than_good(client):
    good = client.post("/predict", json=GOOD).json()["default_probability"]
    risky = client.post("/predict", json=RISKY).json()["default_probability"]
    assert risky > good


def test_validation_errors(client):
    bad = {**GOOD, "age": 5}
    assert client.post("/predict", json=bad).status_code == 422
    missing = {k: v for k, v in GOOD.items() if k != "income"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**GOOD, "home_ownership": "CASTLE"}).status_code == 422


def test_prediction_is_persisted(client):
    pid = client.post("/predict", json=GOOD).json()["prediction_id"]
    stored = client.get(f"/predictions/{pid}").json()
    assert stored["inputs"]["income"] == GOOD["income"]
    assert stored["model_version"]
    assert len(stored["top_reasons"]) == 3
    assert client.get("/predictions/999999").status_code == 404


def test_same_input_same_output(client):
    a = client.post("/predict", json=GOOD).json()
    b = client.post("/predict", json=GOOD).json()
    assert a["default_probability"] == b["default_probability"]
    assert a["top_reasons"] == b["top_reasons"]


def test_model_info(client):
    info = client.get("/model/info").json()
    assert "lightgbm" in info["metrics"]["test"]
