
import pickle
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np


#Guardar el código de esta sección en el archivo `main.py`. Note que ejecutar `python main.py` debería levantar el servidor en el puerto por defecto.

try:
    with open("./models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    try:
        best_params = model.get_xgb_params()
    except:
        best_params = {}
except Exception as e:
    raise RuntimeError(f"{e}")







labels_dict = {0:'agua no potable', 1: 'agua potable'}

def make_prediction(ph,Hardness, Solids, Chloramines, Sulfate,
                  Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    features = [
        [ph,Hardness, Solids, Chloramines, Sulfate,
                  Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
        ]
    prediction = model.predict(features).item()
    label = labels_dict[prediction]
    return label


#Defina `GET` con ruta tipo *home* que describa brevemente su modelo, el problema que intenta resolver, su entrada y salida.
app = FastAPI()
@app.get("/", response_class=HTMLResponse)
async def home():
    hyperparams_html = ""
    if best_params:
        hyperparams_html = "<h2>Hiperparámetros del modelo XGBoost optimizados con Optuna:</h2><ul>"
        for k, v in best_params.items():
            hyperparams_html += f"<li><strong>{k}</strong>: {v}</li>"
        hyperparams_html += "</ul>"

    html_content = f"""
    <html>
        <head>
            <title>Modelo de Potabilidad del Agua </title>
        </head>
        <body style="font-family:Arial; line-height:1.6; margin:40px;">
            <h1>Predicción de Agua Potable</h1>
            <p>
                Este sistema predice si el agua es potable o no, utilizando un modelo <strong>XGBoost</strong> optimizado con <strong>Optuna</strong>.
                El modelo fue entrenado con mediciones químicas captadas por sensores distribuidos en la red hídrica de la comuna.
            </p>
            <p>
                <strong>Entrada:</strong> 9 variables numéricas (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity).<br>
                <strong>Salida:</strong> Etiqueta que indica si el agua es <em>potable</em> o <em>no potable</em>.
            </p>
            <p>
                Este sistema permite alertar a tiempo ante eventuales riesgos en la calidad del agua.
            </p>
            {hyperparams_html}
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)



#Defina un `POST` a la ruta `/potabilidad/` donde utilice su mejor optimizado para predecir si una medición de agua es o no potable.
@app.post("/potabilidad")
async def predict(ph:float,Hardness:float, Solids:float, Chloramines:float, Sulfate:float,
                  Conductivity:float, Organic_carbon:float, Trihalomethanes:float, Turbidity:float):
    label = make_prediction(ph,Hardness, Solids, Chloramines, Sulfate,
                  Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
    return {"label": label}

if __name__ == "__main__":
    uvicorn.run('main:app', port=8000)
    
