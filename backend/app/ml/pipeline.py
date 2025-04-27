import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess_dataset
from colorama import Fore, Style, init
import os

# 🎨 Initialiser colorama
init(autoreset=True)

# 📁 Charger et afficher le dataset
csv_path = "../../../data/simulated_fraud_dataset.csv"
X, y = preprocess_dataset(csv_path)

print(f"\n{Fore.CYAN}🔍 Aperçu du dataset :{Style.RESET_ALL}")
print(pd.read_csv(csv_path).head())

print(f"\n{Fore.CYAN}📊 Infos générales :{Style.RESET_ALL}")
print(pd.read_csv(csv_path).info())

print(f"\n{Fore.CYAN}📈 Statistiques descriptives :{Style.RESET_ALL}")
print(pd.read_csv(csv_path).describe())

# 🔀 Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 📦 Modèles équilibrés
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

# 🔥 Afficher tableau comparatif
df_results = pd.DataFrame(results)
print(f"\n{Fore.YELLOW}📊 Résultats comparatifs des modèles :{Style.RESET_ALL}")
print(df_results.sort_values(by="Precision", ascending=False))

# 💾 Sauvegarder les résultats en CSV
os.makedirs("../../../outputs", exist_ok=True)
csv_output_path = "../../../outputs/model_comparison_results.csv"
df_results.to_csv(csv_output_path, index=False)
print(f"\n📄 Résultats sauvegardés dans : {csv_output_path}")

# 🏆 Meilleur modèle
best_model = df_results.sort_values(by="Precision", ascending=False).iloc[0]
print(f"\n{Fore.GREEN}🏆 Meilleur modèle basé sur la précision : {best_model['Model']}{Style.RESET_ALL}")

# 📈 Visualisation
plt.figure(figsize=(12, 6))
df_melted = df_results.melt(id_vars=["Model"], var_name="Metric", value_name="Score")

sns.barplot(x="Model", y="Score", hue="Metric", data=df_melted)
plt.title("Comparaison des modèles - Precision, Recall, F1-Score")
plt.ylabel("Score")
plt.xlabel("Modèles")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.tight_layout()

# 💾 Sauvegarder le graphe
plot_output_path = "../../../outputs/model_comparison_graph.png"
plt.savefig(plot_output_path)
plt.show()

print(f"\n🖼️ Graphe sauvegardé sous : {plot_output_path}")