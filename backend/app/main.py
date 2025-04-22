from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from ml.predict import predict_transaction

app = FastAPI(title="Système de Détection de Fraudes Financières")

# 🎯 Modèle de requête attendu
class Transaction(BaseModel):
    amount: float
    transaction_type: Literal["payment", "transfer", "withdrawal"]
    transaction_channel: Literal["web", "mobile", "ATM", "POS"]
    account_age_days: int
    num_transactions_last_24h: int
    avg_transaction_amount_last_7d: float
    is_weekend: int
    device_change_flag: int



# 🚀 Endpoint de prédiction (PREDICT COMMAND)
@app.post("/predict")
def predict(tx: Transaction):
    tx_dict = tx.dict()
    is_fraud = predict_transaction(tx_dict)
    return {"fraud": is_fraud}

# ✅ Endpoint de test de vie (PING COMMAND)
@app.get("/ping")
def ping():
    return {"message": "API opérationnelle 🚀"}