from pathlib import Path
 
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
 
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_FILE = BASE_DIR / "models" / "modelo_churn.pkl"
STATIC_DIR = BASE_DIR / "static"
 
app = FastAPI(
    title="API de Predicción de Churn",
    version="0.1.0",
    description="API básica para consumir un modelo de Machine Learning."
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
 
class Cliente(BaseModel):
    edad: int
    antiguedad_meses: int
    saldo_promedio: float
    reclamos: int
    usa_app: int
 
def cargar_modelo():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)
 
@app.get("/app")
def frontend():
    return FileResponse(str(STATIC_DIR / "churn_predictor.html"))
 
@app.get("/")
def inicio():
    return {"mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "autor": "David Salomon Aliaga Nina"}
 
@app.get("/health")
def health():
    return {"estado": "ok", "modelo_disponible": MODEL_FILE.exists()}
 
@app.post("/predict")
def predict(cliente: Cliente):
    modelo = cargar_modelo()
 
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo aún no está disponible. Primero se debe entrenar el modelo."
        )
 
    datos = pd.DataFrame([cliente.model_dump()])
    prediccion = int(modelo.predict(datos)[0])
 
    probabilidad = None
    if hasattr(modelo, "predict_proba"):
        probabilidad = float(modelo.predict_proba(datos)[0][1])
 
    return {
        "churn_predicho": prediccion,
        "probabilidad_churn": probabilidad
    }