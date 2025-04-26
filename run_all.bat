@echo off
title 🚀 Lancement Système de Détection de Fraude

:: Activer l'environnement virtuel
call .venv\Scripts\activate

:: Lancer FastAPI en arrière-plan
start cmd /k "cd backend\app && uvicorn main:app --reload"

:: Attendre 3 secondes pour laisser FastAPI démarrer
timeout /t 3 > nul

:: Lancer l'interface Streamlit
cd frontend
streamlit run predict_ui.py
