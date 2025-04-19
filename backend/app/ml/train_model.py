import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_dataset

# Charger et prétraiter les données
X, y = preprocess_dataset("data/simulated_fraud_dataset.csv")

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Initialiser XGBoost avec gestion des classes déséquilibrées
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42
)

# Entraînement
print("⚙️ Entraînement du modèle en cours...")
model.fit(X_train, y_train)
print("✅ Modèle entraîné avec succès")

# Évaluation
y_pred = model.predict(X_test)
print("\n🎯 Rapport de performance :")
print(classification_report(y_test, y_pred))
print("\n📊 Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# Sauvegarde du modèle
model_path = "models/xgboost_fraud_model.joblib"
joblib.dump(model, model_path)
print(f"💾 Modèle sauvegardé : {model_path}")
