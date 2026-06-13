"""
API de predicción de churn con FastAPI.

La API carga un modelo serializado, valida los datos de entrada
y devuelve una predicción junto con su probabilidad.
"""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_churn_v1.joblib"
STATIC_DIR = PROJECT_ROOT / "static"

VERSION_MODELO = "modelo_churn_v1"
AUTOR = "David Salomon Aliaga Nina"

if not MODEL_PATH.exists():
    raise RuntimeError(
        "No se encontró el modelo serializado. "
        "Ejecute primero: python src/entrenar_modelo.py"
    )

modelo = joblib.load(MODEL_PATH)


class ClienteEntrada(BaseModel):
    antiguedad: int = Field(
        ...,
        ge=0,
        le=120,
        description="Antigüedad del cliente expresada en meses",
        examples=[24],
    )
    cargo_mensual: float = Field(
        ...,
        ge=0,
        le=150,
        description="Cargo mensual del cliente en USD",
        examples=[85.0],
    )
    reclamos: int = Field(
        ...,
        ge=0,
        le=7,
        description="Cantidad de reclamos recientes (últimos 12 meses)",
        examples=[2],
    )


class PrediccionSalida(BaseModel):
    churn_predicho: int
    probabilidad_churn: float
    prediccion: str
    version_modelo: str
    autor: str


app = FastAPI(
    title="API de predicción de churn",
    description="Servicio académico ML-Ops para estimar riesgo de abandono.",
    version="1.0.0",
)

# Sirve el frontend estático en /ui
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def raiz():
    """Redirige al frontend HTML si existe, sino devuelve estado."""
    html_path = STATIC_DIR / "churn_predictor.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "autor": AUTOR,
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "estado": "ok",
        "modelo": VERSION_MODELO,
    }


@app.post("/predict", response_model=PrediccionSalida)
def predict(datos: ClienteEntrada) -> PrediccionSalida:
    """
    Recibe los datos del cliente y devuelve la predicción de churn.

    - **antiguedad**: meses como cliente (0–120)
    - **cargo_mensual**: monto mensual en USD (0–150)
    - **reclamos**: cantidad de reclamos recientes (0–7)
    """
    try:
        X = [[
            datos.antiguedad,
            datos.cargo_mensual,
            datos.reclamos,
        ]]

        probabilidad = float(modelo.predict_proba(X)[0][1])
        churn_predicho = 1 if probabilidad >= 0.50 else 0
        etiqueta = "alto_riesgo" if churn_predicho == 1 else "bajo_riesgo"

        return PrediccionSalida(
            churn_predicho=churn_predicho,
            probabilidad_churn=round(probabilidad, 4),
            prediccion=etiqueta,
            version_modelo=VERSION_MODELO,
            autor=AUTOR,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar la predicción.",
        ) from exc