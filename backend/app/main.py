from fastapi import FastAPI # type: ignore

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Système de détection de fraude actif"}
