from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os

from model_inference import predict_image

app = FastAPI(
    title="ODIR-5K API",
    description="API de prédiction de pathologies oculaires (ODIR-5K + ResNet18).",
    version="1.0.0"
)

# CORS pour autoriser ton futur front (localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en prod, mets ton vrai domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "API ODIR-5K opérationnelle 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Générer un nom temporaire
    ext = os.path.splitext(file.filename)[1] or ".png"
    temp_name = f"temp_{uuid.uuid4()}{ext}"

    # 2. Sauvegarder le fichier
    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 3. Prédire
        result = predict_image(temp_name, threshold=0.5)

    finally:
        # 4. Nettoyer le fichier temporaire
        if os.path.exists(temp_name):
            os.remove(temp_name)

    # 5. Retourner le résultat
    return {
        "success": True,
        "prediction": result["labels"],          # labels activés
        "all_probabilities": result["all_probabilities"]  # toutes les classes
    }
