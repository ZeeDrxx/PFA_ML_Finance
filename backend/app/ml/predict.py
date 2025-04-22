import joblib
from .preprocess import preprocess_single
import os

# Charger le modèle une seule fois
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_fraud_model.joblib")
#MODEL_PATH = "../../../models/xgboost_fraud_model.joblib"

model = joblib.load(MODEL_PATH)

def predict_transaction(transaction: dict) -> bool:
    """
    Prédit si une transaction est frauduleuse (True) ou non (False).
    """
    tx_processed = preprocess_single(transaction)
    prediction = model.predict(tx_processed)[0]
    return bool(prediction)
