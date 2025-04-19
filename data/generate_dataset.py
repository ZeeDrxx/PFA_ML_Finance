import pandas as pd 
import numpy as np 
from faker import Faker 
import random

# Initialiser Faker et seed
fake = Faker()
np.random.seed(42)

# Nombre de transactions à générer
n_samples = 10000
user_ids = [fake.uuid4() for _ in range(500)]

# Génération des données
data = {
    "transaction_id": [fake.uuid4() for _ in range(n_samples)],
    "user_id": [random.choice(user_ids) for _ in range(n_samples)],
    "amount": np.round(np.random.exponential(scale=200, size=n_samples), 2),
    "transaction_type": np.random.choice(["payment", "transfer", "withdrawal", "deposit"], size=n_samples),
    "timestamp": [fake.date_time_this_year() for _ in range(n_samples)],
    "transaction_channel": np.random.choice(["web", "mobile", "ATM", "POS"], size=n_samples),
    "location": [fake.city() for _ in range(n_samples)],
    "account_age_days": np.random.randint(10, 1500, size=n_samples),
    "num_transactions_last_24h": np.random.poisson(2, size=n_samples),
    "avg_transaction_amount_last_7d": np.round(np.random.normal(loc=150, scale=50, size=n_samples), 2),
    "is_weekend": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
    "device_change_flag": np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
}

# Règle simple pour simuler des fraudes
def is_fraud(row):
    score = 0
    if row["amount"] > 1000:
        score += 1
    if row["device_change_flag"] == 1:
        score += 1
    if row["num_transactions_last_24h"] > 5:
        score += 1
    return 1 if score >= 2 else 0

df = pd.DataFrame(data)
df["is_fraud"] = df.apply(is_fraud, axis=1)

# Export CSV
df.to_csv("simulated_fraud_dataset.csv", index=False)
print("✅ Dataset simulé généré : simulated_fraud_dataset.csv")
