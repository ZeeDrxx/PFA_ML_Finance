import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Charger le dataset simulé
df = pd.read_csv("data/simulated_fraud_dataset.csv")

# Sélection des features utiles
features = [
    "amount",
    "transaction_type",
    "transaction_channel",
    "account_age_days",
    "num_transactions_last_24h",
    "avg_transaction_amount_last_7d",
    "is_weekend",
    "device_change_flag"
]

# Encodage one-hot des variables catégorielles
df_encoded = pd.get_dummies(df[features], columns=["transaction_type", "transaction_channel"], drop_first=True)
X = df_encoded
y = df["is_fraud"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialisation du modèle XGBoost
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # équilibrage des classes
    random_state=42
)

# Entraînement
model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = model.predict(X_test)

print("🎯 Rapport de performance :")
print(classification_report(y_test, y_pred))

print("📊 Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# Sauvegarde du modèle
joblib.dump(model, "models/xgboost_fraud_model.joblib")
print("✅ Modèle sauvegardé sous models/xgboost_fraud_model.joblib")
