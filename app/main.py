import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.scoring import Scorer

from .config import database_url, model_dir
from .db import Prediction, make_session_factory
from .schemas import Applicant, PredictionOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("credit-risk")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = Scorer(model_dir())
    app.state.sessions = make_session_factory(database_url())
    log.info("Loaded model v%s", app.state.scorer.version)
    yield


app = FastAPI(
    title="Explainable Credit Risk API",
    description="Loan default probability with SHAP-based reasons. Learning project, not for real lending decisions.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_db(request: Request):
    db = request.app.state.sessions()
    try:
        yield db
    finally:
        db.close()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/dashboard/")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info(request: Request):
    s = request.app.state.scorer
    return {"model_version": s.version, "metrics": s.metrics}


@app.post("/predict", response_model=PredictionOut)
def predict(applicant: Applicant, request: Request, db=Depends(get_db)):
    row = applicant.model_dump()
    result = request.app.state.scorer.predict(row)
    rec = Prediction(
        inputs=row,
        default_probability=result["default_probability"],
        risk_band=result["risk_band"],
        top_reasons=result["top_reasons"],
        model_version=result["model_version"],
    )
    db.add(rec)
    db.commit()
    log.info("prediction id=%s band=%s", rec.id, rec.risk_band)
    return {"prediction_id": rec.id, **result}


@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int, db=Depends(get_db)):
    rec = db.get(Prediction, prediction_id)
    if rec is None:
        raise HTTPException(404, "Prediction not found")
    return {
        "prediction_id": rec.id,
        "created_at": rec.created_at,
        "inputs": rec.inputs,
        "default_probability": rec.default_probability,
        "risk_band": rec.risk_band,
        "top_reasons": rec.top_reasons,
        "model_version": rec.model_version,
    }


app.mount(
    "/dashboard",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="dashboard",
)
