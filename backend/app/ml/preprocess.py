import pandas as pd

def preprocess_dataset(csv_path: str):
    # Charger le dataset
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # ➤ 1. Supprimer les doublons
    df.drop_duplicates(inplace=True)
    print(f"🔁 Après suppression des doublons : {df.shape[0]} lignes")

    # ➤ 2. Nettoyage des valeurs manquantes
    missing = df.isnull().sum().sum()
    if missing > 0:
        df.dropna(inplace=True)
        print(f"⚠️ Valeurs manquantes supprimées : {missing}")
    else:
        print("✅ Aucune valeur manquante")

    # ➤ 3. Suppression des données aberrantes (valeurs extrêmes)
    outlier_threshold = 100000  # montant maximal arbitraire
    before = df.shape[0]
    df = df[df["amount"] < outlier_threshold]
    print(f"🧹 Données aberrantes supprimées : {before - df.shape[0]} lignes")

    # ➤ 4. Sélection des variables pertinentes
    features = [
        "amount",
        "transaction_type",
        "transaction_channel",
        "account_age_days",
        "num_transactions_last_24h",
        "avg_transaction_amount_last_7d",
        "is_weekend",
        "device_change_flag"
    ]

    # ➤ 5. Encodage one-hot
    df_encoded = pd.get_dummies(df[features], columns=["transaction_type", "transaction_channel"], drop_first=True)

    # ➤ 6. Variable cible
    y = df["is_fraud"]
    X = df_encoded

    print(f"✅ Prétraitement terminé : {X.shape[1]} features prêtes")

    return X, y

def preprocess_single(tx: dict) -> pd.DataFrame:
    """
    Prépare une transaction pour prédiction.
    """
    df = pd.DataFrame([tx])

    df = pd.get_dummies(df, columns=["transaction_type", "transaction_channel"], drop_first=True)

    # Liste exacte des colonnes que le modèle attend
    expected_cols = [
        'amount',
        'account_age_days',
        'num_transactions_last_24h',
        'avg_transaction_amount_last_7d',
        'is_weekend',
        'device_change_flag',
        'transaction_type_payment',
        'transaction_type_transfer',
        'transaction_type_withdrawal',
        'transaction_channel_POS',
        'transaction_channel_mobile',
        'transaction_channel_web'
    ]

    # Ajout des colonnes manquantes avec 0
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Supprimer les colonnes en trop
    df = df[[col for col in expected_cols]]

    return df