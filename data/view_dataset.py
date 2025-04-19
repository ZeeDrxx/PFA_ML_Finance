import pandas as pd

# Charger le dataset
df = pd.read_csv("data/simulated_fraud_dataset.csv")

# Afficher les premières lignes
print("\n🔍 Aperçu du dataset :")
print(df.head())

# Afficher infos générales
print("\n📊 Informations :")
print(df.info())

# Statistiques générales
print("\n📈 Statistiques descriptives :")
print(df.describe())

# Répartition des fraudes
print("\n🚨 Répartition de la variable cible (is_fraud) :")
print(df['is_fraud'].value_counts(normalize=True))
