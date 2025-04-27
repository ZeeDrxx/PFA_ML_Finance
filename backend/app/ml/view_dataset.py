import pandas as pd

# Remonter trois niveaux depuis backend/app/ml/ vers la racine du projet
csv_path = "../../../data/simulated_fraud_dataset.csv"

# Charger le dataset
df = pd.read_csv(csv_path)

# 🔍 Aperçu des premières lignes
print("\n🔍 Aperçu du dataset :")
print(df.head())

# 📊 Informations sur les colonnes et types
print("\n📊 Informations :")
print(df.info())

# 📈 Statistiques générales
print("\n📈 Statistiques descriptives :")
print(df.describe())

# 🚨 Répartition des fraudes
print("\n🚨 Répartition de la variable cible (is_fraud) :")
print(df['is_fraud'].value_counts(normalize=True))
