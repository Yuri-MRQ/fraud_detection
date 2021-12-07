from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel


class Features(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None    

app = FastAPI()

@app.post("/predict/")
async def create_item(features: Features):
    features_dict = features.dict()
    return features_dict