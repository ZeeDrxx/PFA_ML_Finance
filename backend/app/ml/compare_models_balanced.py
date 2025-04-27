import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess_dataset
from colorama import Fore, Style, init

# 🎨 Initialiser colorama
init(autoreset=True)

# 📁 Charger le dataset
csv_path = "../../../data/simulated_fraud_dataset.csv"
X, y = preprocess_dataset(csv_path)

# 🔀 Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 📦 Modèles avec balancing
models = {
    "Decision Tree (balanced)": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest (balanced)": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42),
    "XGBoost (scale_pos_weight)": xgb.XGBClassifier(
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

# 📊 Résultats
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

# 🔥 Afficher tableau
df_results = pd.DataFrame(results)
print(f"\n{Fore.YELLOW}📊 Résultats comparatifs des modèles (équilibrés) :{Style.RESET_ALL}")
print(df_results.sort_values(by="Precision", ascending=False))

# 🏆 Meilleur modèle
best_model = df_results.sort_values(by="Precision", ascending=False).iloc[0]
print(f"\n{Fore.GREEN}🏆 Meilleur modèle basé sur la précision : {best_model['Model']}{Style.RESET_ALL}")
