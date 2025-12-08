import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================
# Config générale
# ==========================
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

# ⚠️ IMPORTANT : même normalisation que dans le notebook
# Remplace 90.0 par la vraie valeur de df_img["age"].max() AVANT normalisation.
AGE_MAX = 90.0

IMG_SIZE = 224
inference_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==========================
# Modèle multimodal
# ==========================
class MultimodalResNet(nn.Module):
    """
    CNN (ResNet18) pour l'image + MLP pour (âge, sexe).
    """
    def __init__(self, num_meta: int = 2, num_classes: int = 8):
        super().__init__()

        self.cnn = models.resnet18(weights=None)  # même archi que pendant l'entraînement
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # on enlève la tête d'origine

        self.meta_fc = nn.Sequential(
            nn.Linear(num_meta, 16),
            nn.ReLU()
        )

        self.classifier = nn.Linear(num_ftrs + 16, num_classes)

    def forward(self, image, meta):
        img_feat = self.cnn(image)           # (B, num_ftrs)
        meta_feat = self.meta_fc(meta)       # (B, 16)
        x = torch.cat([img_feat, meta_feat], dim=1)
        logits = self.classifier(x)          # (B, 8)
        return logits


def build_model(weights_path: str):
    """
    Construit le modèle multimodal et charge les poids .pth
    """
    model = MultimodalResNet(num_meta=2, num_classes=len(LABEL_CODES))
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# Charger le modèle une seule fois au démarrage
MODEL_PATH = "odir_resnet18_multimodal_age_sex.pth"
model = build_model(MODEL_PATH)


def encode_sex(sex_raw):
    """
    sex_raw peut être :
      - "M" / "F"
      - 0 / 1 (ou "0" / "1")
    On renvoie 0.0 (M) ou 1.0 (F).
    """
    if isinstance(sex_raw, str):
        s = sex_raw.strip().upper()
        if s in ["M", "H", "0"]:
            return 0.0
        elif s in ["F", "1"]:
            return 1.0
        else:
            raise ValueError("Sexe invalide, attendu M ou F (ou 0/1).")
    else:
        # numérique
        val = float(sex_raw)
        if val not in [0.0, 1.0]:
            raise ValueError("Sexe numérique invalide, attendu 0 ou 1.")
        return val


@torch.no_grad()
def predict_image(image_path: str, age_raw: float, sex_raw, threshold: float = 0.5):
    """
    Retourne les probabilités + labels activés pour une image,
    en utilisant l'âge et le sexe du patient.
    - age_raw : âge réel en années
    - sex_raw : 'M' / 'F' ou 0 / 1
    """

    # 1. Charger l'image
    img = Image.open(image_path).convert("RGB")

    # 2. Transform
    x = inference_tfms(img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    # 3. Préparer les métadonnées (âge normalisé + sexe encodé)
    sex_enc = encode_sex(sex_raw)
    age_norm = age_raw / AGE_MAX

    meta = torch.tensor([[age_norm, sex_enc]], dtype=torch.float32, device=DEVICE)  # (1,2)

    # 4. Forward
    logits = model(x, meta)
    probs = torch.sigmoid(logits)[0]  # (8,)

    # 5. Convertir en Python
    probs_list = probs.cpu().tolist()

    # 6. Multi-label : labels où prob >= threshold
    active_labels = []
    for i, p in enumerate(probs_list):
        if p >= threshold:
            code = LABEL_CODES[i]
            active_labels.append({
                "code": code,
                "name": LABEL_NAMES[code],
                "probability": float(p)
            })

    # Si aucun label ne dépasse le threshold, on prend la plus probable
    if not active_labels:
        idx_max = int(torch.argmax(probs).item())
        code = LABEL_CODES[idx_max]
        active_labels.append({
            "code": code,
            "name": LABEL_NAMES[code],
            "probability": float(probs_list[idx_max])
        })

    # 7. Retour structuré
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
