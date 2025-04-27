import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

# Charger les données
csv_path = "../../../data/simulated_fraud_dataset.csv"
X, y = preprocess_dataset(csv_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Définir les modèles
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

# Entraîner et évaluer
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

df_results = pd.DataFrame(results)

# Visualisation
plt.figure(figsize=(12, 6))
df_results_melted = df_results.melt(id_vars=["Model"], var_name="Metric", value_name="Score")

sns.barplot(x="Model", y="Score", hue="Metric", data=df_results_melted)
plt.title("Comparaison des modèles - Precision, Recall, F1-Score")
plt.ylabel("Score")
plt.xlabel("Modèles")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
