@echo off
title 🚀 Initialisation et Lancement Détection de Fraude - ML_Finance

:: Si .venv n'existe pas, créer l'environnement virtuel
if not exist ".venv" (
    echo 🔵 Création de l'environnement virtuel...
    python -m venv .venv
)

:: Activer l'environnement virtuel
call .venv\Scripts\activate

:: Installer les dépendances depuis requirements.txt
echo 🛠️ Installation des dépendances...
pip install -r requirements.txt

:: Lancer FastAPI (backend)
start cmd /k "cd backend\app && uvicorn main:app --reload"

:: Attendre quelques secondes que FastAPI démarre
timeout /t 3 > nul

:: Lancer Streamlit (frontend)
cd frontend
streamlit run predict_ui.py
