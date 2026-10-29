from typing import Literal

from pydantic import BaseModel, Field


class Applicant(BaseModel):
    age: int = Field(ge=18, le=100)
    income: float = Field(gt=0, description="Annual income")
    loan_amount: float = Field(gt=0)
    employment_years: float = Field(ge=0, le=60)
    credit_history_length: float = Field(ge=0, le=60)
    num_open_accounts: int = Field(ge=0, le=50)
    num_late_payments: int = Field(ge=0, le=50)
    debt_to_income: float = Field(ge=0, le=5)
    home_ownership: Literal["RENT", "OWN", "MORTGAGE"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 34, "income": 52000, "loan_amount": 15000,
                "employment_years": 6, "credit_history_length": 8,
                "num_open_accounts": 4, "num_late_payments": 1,
                "debt_to_income": 0.29, "home_ownership": "RENT",
            }
        }
    }


class Reason(BaseModel):
    feature: str
    label: str
    value: float | int | str
    impact: float
    direction: str


class PredictionOut(BaseModel):
    prediction_id: int
    default_probability: float
    risk_band: str
    top_reasons: list[Reason]
    model_version: str
