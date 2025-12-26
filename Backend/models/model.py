import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import cv2
import numpy as np
import os
import io
from typing import Optional

from models.tamper_model import HybridModel
from models.restore_model import load_restore_model

# ==============================
# DEVICE
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAMPER_MODEL_PATH = os.path.join(BASE_DIR, "weights", "hybrid_vgg16_quantum_ela.pt")

# ==============================
# TRANSFORM (MUST MATCH TRAINING)
# ==============================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

_tamper_model = None
_restore_model = None

# ==============================
# LOAD MODELS
# ==============================
def load_tamper_model():
    global _tamper_model
    if _tamper_model is None:
        state = torch.load(TAMPER_MODEL_PATH, map_location=DEVICE)
        _tamper_model = HybridModel().to(DEVICE)
        _tamper_model.load_state_dict(state, strict=False)
        _tamper_model.eval()
    return _tamper_model


def load_restoration_model():
    global _restore_model
    if _restore_model is None:
        _restore_model = load_restore_model()
    return _restore_model


# ==============================
# CLASSIFICATION (FM SCORE)
# ==============================
def predict_tampering(
    suspected_img: Image.Image,
    original_img: Optional[Image.Image] = None
):
    """
    Returns:
        label (Tampered / Authentic),
        fm_confidence (float %)
    """

    model = load_tamper_model()

    if suspected_img.mode != "RGB":
        suspected_img = suspected_img.convert("RGB")

    x = transform(suspected_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    authentic_conf = probs[0][0].item()
    tampered_conf = probs[0][1].item()

    if tampered_conf >= authentic_conf:
        return "Tampered", round(tampered_conf * 100, 2)
    else:
        return "Authentic", round(authentic_conf * 100, 2)


# ==============================
# FORGERY MASK GENERATION (LIKE PAPER)
# ==============================
def generate_forgery_mask(suspected_img: Image.Image):
    """
    Produces a binary white-on-black forgery map
    similar to research paper outputs.
    """

    if suspected_img.mode != "RGB":
        suspected_img = suspected_img.convert("RGB")

    img_np = np.array(suspected_img)

    # ---- ELA STEP ----
    buf = io.BytesIO()
    suspected_img.save(buf, "JPEG", quality=90)
    buf.seek(0)
    compressed = Image.open(buf)

    ela = np.abs(
        np.array(suspected_img).astype(np.int16) -
        np.array(compressed).astype(np.int16)
    ).astype(np.uint8)

    ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)
    ela_gray = cv2.normalize(ela_gray, None, 0, 255, cv2.NORM_MINMAX)

    # ---- THRESHOLD ----
    _, mask = cv2.threshold(ela_gray, 30, 255, cv2.THRESH_BINARY)

    # ---- MORPHOLOGY (SILHOUETTE CLEANING) ----
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask[mask > 0] = 255
    return mask
