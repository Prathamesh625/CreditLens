from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Audit record: every prediction with its inputs, reasons and model version."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    inputs: Mapped[dict] = mapped_column(JSON)
    default_probability: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(10))
    top_reasons: Mapped[list] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(20))


def make_session_factory(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
