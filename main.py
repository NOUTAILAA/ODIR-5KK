from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os

from model_inference import predict_image

app = FastAPI(
    title="ODIR-5K API",
    description="API de prédiction de pathologies oculaires (ODIR-5K + ResNet18 multimodal : image + âge + sexe).",
    version="1.0.0"
)

# CORS pour autoriser ton futur front (localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en prod : mets ton vrai domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "API ODIR-5K multimodale opérationnelle 🚀"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    age: float = Form(...),
    sex: str = Form(...)
):
    """
    Endpoint de prédiction :
    - file : image fundus
    - age  : âge du patient (en années)
    - sex  : 'M' ou 'F'
    """

    # 1. Générer un nom temporaire
    ext = os.path.splitext(file.filename)[1] or ".png"
    temp_name = f"temp_{uuid.uuid4()}{ext}"

    # 2. Sauvegarder le fichier
    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 3. Prédire (image + âge + sexe)
        result = predict_image(
            image_path=temp_name,
            age_raw=age,
            sex_raw=sex,
            threshold=0.5
        )

    finally:
        # 4. Nettoyer le fichier temporaire
        if os.path.exists(temp_name):
            os.remove(temp_name)

    # 5. Retourner le résultat JSON
    return {
        "success": True,
        "prediction": result["labels"],            # labels activés (code, name, prob)
        "all_probabilities": result["all_probabilities"]  # toutes les classes
    }
