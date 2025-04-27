@echo off
title 🚀 Lancement Détection de Fraude - ML_Finance

:: Activer l'environnement virtuel
call .venv\Scripts\activate

:: Lancer FastAPI (backend)
start cmd /k "cd backend\app && uvicorn main:app --reload"

:: Attendre quelques secondes que le serveur démarre
timeout /t 3 > nul

:: Lancer Streamlit (frontend)
cd frontend
streamlit run predict_ui.py
