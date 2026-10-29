import os


def model_dir() -> str:
    return os.environ.get("MODEL_DIR", "models")


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///./credit_risk.db")
