import os
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_dataset

# 📁 Résoudre le chemin du CSV
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
CSV_PATH = os.path.join(BASE_DIR, "data", "simulated_fraud_dataset.csv")

# 📦 Charger et prétraiter les données
X, y = preprocess_dataset(CSV_PATH)

# 🔀 Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ⚙️ Initialisation du modèle XGBoost
model = xgb.XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42
)

# 🎯 Entraînement
print("⚙️ Entraînement du modèle en cours...")
model.fit(X_train, y_train)
print("✅ Modèle entraîné avec succès")

# 📊 Évaluation
y_pred = model.predict(X_test)
print("\n🎯 Rapport de performance :")
print(classification_report(y_test, y_pred))
print("\n📊 Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# 💾 Sauvegarde du modèle
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "xgboost_fraud_model.joblib")
joblib.dump(model, model_path)
print(f"✅ Modèle sauvegardé : {model_path}")
