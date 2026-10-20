import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("heart-disease-api")


MODEL_PATH = Path("models/heart_disease_pipeline.joblib")

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="Heart Disease Prediction API",
    description="AIMLCZG523 MLOps Assignment",
    version="1.0.0",
)


REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
)

ERROR_COUNT = Counter(
    "prediction_errors_total",
    "Total prediction errors",
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency",
)


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float | None = None
    thal: float | None = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    logger.info(
        "%s %s status=%s duration=%.4fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


@app.get("/")
def root():
    return {
        "service": "Heart Disease Prediction API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(patient: PatientData):
    REQUEST_COUNT.inc()

    start = time.perf_counter()

    try:
        input_df = pd.DataFrame(
            [patient.model_dump()]
        )

        prediction = int(
            model.predict(input_df)[0]
        )

        probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        return {
            "prediction": prediction,
            "label": (
                "heart_disease_present"
                if prediction == 1
                else "no_heart_disease"
            ),
            "confidence": round(probability, 4),
        }

    except Exception as exc:
        ERROR_COUNT.inc()
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc

    finally:
        REQUEST_LATENCY.observe(
            time.perf_counter() - start
        )