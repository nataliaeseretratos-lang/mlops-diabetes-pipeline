from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import numpy as np
from pydantic import BaseModel, Field

MODEL_NAME = "diabetes_progression_regressor"
MODEL_ALIAS = "champion"
TRACKING_URI = "sqlite:///mlflow.db"

app = FastAPI(
    title="Diabetes Progression API",
    description="API para servir predicciones del modelo champion registrado en MLflow.",
    version="1.0.0",
)

model: Optional[Any] = None
model_metadata: Dict[str, Any] = {
    "model_name": MODEL_NAME,
    "alias": MODEL_ALIAS,
    "version": None,
    "run_id": None,
}


def load_champion_model() -> None:
    """Carga el modelo con alias 'champion' desde MLflow Registry."""
    global model, model_metadata
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    model_version_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    loaded_model = mlflow.sklearn.load_model(model_uri)

    model = loaded_model
    model_metadata = {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(model_version_info.version),
        "run_id": model_version_info.run_id,
    }


@app.on_event("startup")
def startup_event() -> None:
    try:
        load_champion_model()
    except Exception:
        model = None  # noqa: F841


class DiabetesInput(BaseModel):
    age: float = Field(..., description="Edad normalizada")
    sex: float = Field(..., description="Sexo normalizado")
    bmi: float = Field(..., description="Índice de masa corporal normalizado")
    bp: float = Field(..., description="Presión arterial media normalizada")
    s1: float = Field(..., description="Colesterol total sérico (tc)")
    s2: float = Field(..., description="Lipoproteínas de baja densidad (ldl)")
    s3: float = Field(..., description="Lipoproteínas de alta densidad (hdl)")
    s4: float = Field(..., description="Colesterol total / HDL (tch)")
    s5: float = Field(..., description="Posible logaritmo de triglicéridos (ltg)")
    s6: float = Field(..., description="Nivel de azúcar en sangre (glu)")


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    alias: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo champion no está cargado.",
        )
    return {
        "model_name": model_metadata["model_name"],
        "version": model_metadata["version"],
        "alias": model_metadata["alias"],
        "tracking_uri": TRACKING_URI,
        "run_id": model_metadata["run_id"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: DiabetesInput) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio no disponible: el modelo champion no está cargado.",
        )

    try:
        features = np.array(
            [[
                payload.age,
                payload.sex,
                payload.bmi,
                payload.bp,
                payload.s1,
                payload.s2,
                payload.s3,
                payload.s4,
                payload.s5,
                payload.s6,
            ]],
            dtype=float,
        )
        pred_value = float(model.predict(features)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar la predicción: {exc}",
        ) from exc

    return PredictionResponse(
        prediction=pred_value,
        model_name=model_metadata["model_name"],
        model_version=str(model_metadata["version"]),
        alias=model_metadata["alias"],
    )
