import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ====== Config ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ordre des labels comme dans ton dataset
LABEL_CODES = ["N", "D", "G", "C", "A", "H", "M", "O"]
LABEL_NAMES = {
    "N": "Normal",
    "D": "Diabétique (DR)",
    "G": "Glaucome",
    "C": "Cataracte",
    "A": "Dégénérescence maculaire liée à l'âge (ARMD)",
    "H": "Hypertension",
    "M": "Myopie",
    "O": "Autre maladie"
}

# mêmes normalisations que dans ton training (ImageNet)
IMG_SIZE = 224
inference_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ====== Construction du modèle ======
def build_model(weights_path: str):
    # même archi que dans ton notebook
    model = models.resnet18(weights=None)  # pas besoin des poids ImageNet ici
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(LABEL_CODES))

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


# Charger le modèle une seule fois au démarrage de l'API
MODEL_PATH = "odir_resnet18_multilabel.pth"
model = build_model(MODEL_PATH)


# ====== Fonction de prédiction ======
@torch.no_grad()
def predict_image(image_path: str, threshold: float = 0.5):
    """Retourne les probabilités + labels activés pour une image."""

    # 1. Charger l'image
    img = Image.open(image_path).convert("RGB")

    # 2. Transform
    x = inference_tfms(img).unsqueeze(0).to(DEVICE)  # shape (1,3,H,W)

    # 3. Forward
    logits = model(x)
    probs = torch.sigmoid(logits)[0]  # shape (8,)

    # 4. Convertir en Python
    probs_list = probs.cpu().tolist()

    # 5. Multi-label : labels où prob >= threshold
    active_labels = []
    for i, p in enumerate(probs_list):
        if p >= threshold:
            code = LABEL_CODES[i]
            active_labels.append({
                "code": code,
                "name": LABEL_NAMES[code],
                "probability": float(p)
            })

    # Si aucun label ne dépasse le threshold, on prend le plus probable
    if not active_labels:
        idx_max = int(torch.argmax(probs).item())
        code = LABEL_CODES[idx_max]
        active_labels.append({
            "code": code,
            "name": LABEL_NAMES[code],
            "probability": float(probs_list[idx_max])
        })

    # Retour structuré
    return {
        "labels": active_labels,
        "all_probabilities": [
            {
                "code": LABEL_CODES[i],
                "name": LABEL_NAMES[LABEL_CODES[i]],
                "probability": float(p)
            }
            for i, p in enumerate(probs_list)
        ]
    }
