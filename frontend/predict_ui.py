import streamlit as st
import requests

st.set_page_config(page_title="Détection de Fraude", page_icon="💸")

st.title("💸 Système de Détection de Fraude")
st.write("Remplis les informations de la transaction ci-dessous :")

# Formulaire utilisateur
amount = st.number_input("Montant de la transaction", min_value=0.0, step=0.01)
transaction_type = st.selectbox("Type de transaction", ["payment", "transfer", "withdrawal"])
transaction_channel = st.selectbox("Canal utilisé", ["web", "mobile", "ATM", "POS"])
account_age_days = st.number_input("Âge du compte (jours)", min_value=0)
num_tx_24h = st.number_input("Nombre de transactions dernières 24h", min_value=0)
avg_tx_7d = st.number_input("Montant moyen des 7 derniers jours", min_value=0.0, step=0.01)
is_weekend = st.selectbox("Est-ce le weekend ?", [0, 1])
device_change_flag = st.selectbox("Changement de device détecté ?", [0, 1])

# Bouton d'envoi
if st.button("Prédire"):
    tx = {
        "amount": amount,
        "transaction_type": transaction_type,
        "transaction_channel": transaction_channel,
        "account_age_days": account_age_days,
        "num_transactions_last_24h": num_tx_24h,
        "avg_transaction_amount_last_7d": avg_tx_7d,
        "is_weekend": is_weekend,
        "device_change_flag": device_change_flag
    }

    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=tx)
        if res.status_code == 200:
            result = res.json()
            if result["fraud"]:
                st.error("🚨 FRAUDE DÉTECTÉE !")
            else:
                st.success("✅ Transaction NORMALE.")
        else:
            st.warning(f"Erreur API : {res.status_code}")
    except Exception as e:
        st.warning(f"Erreur de connexion à l'API : {e}")
