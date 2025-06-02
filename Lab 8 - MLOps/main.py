
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

try:
    with open("./models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"{e}")