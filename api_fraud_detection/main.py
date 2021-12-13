from typing import List
from model import fraud_model

from fastapi import FastAPI
from pydantic import BaseModel


class Features(BaseModel):
    device_id: float
    balance: float 
    processed_at: float
    age_range: float
    number_of_selfies_sent: float
    time_client: float
    cash_out_type_1: float
    cash_out_type_2: float
    cash_out_type_3: float
    cash_out_type_6: float

class Samples(BaseModel):
    samples: List[Features]


app = FastAPI()

@app.get("/")
def home():
    return "Fraud Predict APi"

@app.post("/predict/")
async def create_item(features: Samples):

    features_dict = features.samples

    predict = fraud_model.predict(features_dict)

    return predict