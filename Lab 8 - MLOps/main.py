import pickle
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

try:
    with open("./models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

if hasattr(model, "get_xgb_params"):
    best_params = model.get_xgb_params()
else:
    best_params = {}

# Predecir
def make_prediction(features_list):
    prediction = model.predict([features_list]).item()
    return prediction

class WaterQualityInput(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Ruta GET 
@app.get("/", response_class=HTMLResponse)
async def home():   
    html_content = f"""
    <html>
        <head>
            <title>Modelo de Potabilidad del Agua 🚰</title>
        </head>
        <body style="font-family:Arial; line-height:1.6; margin:40px;">
            <h1>Sistema de Predicción de Agua Potable</h1>

            <p>
                Este sistema predice si el agua es potable o no, utilizando un modelo <strong>XGBoost</strong> optimizado con <strong>Optuna</strong>.
                El modelo fue entrenado con mediciones químicas captadas por sensores distribuidos en la red hídrica de la comuna.
            </p>

            <p>
                <strong>Entrada:</strong> pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity.<br>
                <strong>Salida:</strong> Etiqueta que indica si el agua es potable o no potable.
            </p>

            <p>
                Este sistema permite alertar a tiempo ante eventuales riesgos en la calidad del agua.
            </p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Ruta POST
@app.post("/potabilidad/")
async def predict(data: WaterQualityInput):
    features = [
        data.ph,
        data.Hardness,
        data.Solids,
        data.Chloramines,
        data.Sulfate,
        data.Conductivity,
        data.Organic_carbon,
        data.Trihalomethanes,
        data.Turbidity
    ]
    prediction = make_prediction(features)
    return {"potabilidad": prediction}

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
