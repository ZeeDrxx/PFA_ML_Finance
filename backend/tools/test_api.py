import requests

# Adresse de l’API locale
url = "http://127.0.0.1:8000/predict"

# Exemple de transaction
payload = {
    "amount": 890.55,
    "transaction_type": "transfer",
    "transaction_channel": "POS",
    "account_age_days": 310,
    "num_transactions_last_24h": 6,
    "avg_transaction_amount_last_7d": 140.00,
    "is_weekend": 1,
    "device_change_flag": 1
}

# Envoyer la requête POST
response = requests.post(url, json=payload)

# Afficher le résultat
if response.status_code == 200:
    result = response.json()
    print("✅ Résultat :", "🚨 FRAUDE détectée" if result["fraud"] else "✔️ Transaction normale")
else:
    print("❌ Erreur API :", response.status_code, response.text)
