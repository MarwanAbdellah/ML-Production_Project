"""FastAPI application: HTTP/JSON -> Pydantic validation -> saved pipeline -> JSON."""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from imblearn.pipeline import Pipeline

from production_app.config import settings
from production_app.logging_config import configure_logging, get_logger
from production_app.model_service import load_model, predict
from production_app.schemas import (
    LABELS,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

configure_logging(settings.log_level)
logger = get_logger(__name__)

MODEL_UNAVAILABLE = {"description": "Model artifact unavailable"}

app = FastAPI(
    title="Breast Cancer Classification API",
    description=(
        "Serves the Assignment 01 fitted pipeline (SMOTETomek -> StandardScaler "
        "-> Logistic Regression). Inference only: the model is never retrained."
    ),
    version="0.1.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Echoing `input` back would resend NaN/inf (not JSON-serialisable, turns a
    # 422 into a 500) and reflect the whole payload; keep type/loc/msg only.
    errors = [
        {"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": errors}
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the detail server-side, return nothing that reveals internals to the client.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "internal server error"},
    )


def _get_model() -> Pipeline:
    try:
        return load_model()
    except Exception as exc:  # missing or corrupt artifact, never disclose which
        logger.error("Model artifact unavailable (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model artifact unavailable",
        ) from exc


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, **MODEL_UNAVAILABLE}},
    summary="Service and model-artifact health",
)
def health() -> JSONResponse:
    try:
        load_model()
    except Exception as exc:  # any load failure means unhealthy
        logger.error(
            "Health check failed: model artifact unavailable (%s)", type(exc).__name__
        )
        body = HealthResponse(
            status="unhealthy",
            model_loaded=False,
            detail="model artifact unavailable",
        )
        return JSONResponse(status_code=503, content=body.model_dump())
    return JSONResponse(
        status_code=200,
        content=HealthResponse(status="ok", model_loaded=True).model_dump(),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={503: MODEL_UNAVAILABLE},
    summary="Predict one observation with the saved pipeline",
)
def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    model = _get_model()
    frame = payload.to_frame()
    prediction = int(predict(model, frame)[0])
    probability = (
        float(model.predict_proba(frame)[0][1])
        if hasattr(model, "predict_proba")
        else None
    )
    logger.info("Prediction served: %s", prediction)
    return PredictResponse(
        prediction=prediction,
        label=LABELS[prediction],
        probability=probability,
    )
