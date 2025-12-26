# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import numpy as np
import cv2

from models.model import (
    load_restoration_model,
    predict_tampering,
    generate_forgery_mask
)

app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "AI Image Security Backend is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Backend is operational"
    }

# ================= UTILS =================
def img_to_base64(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def mask_to_base64(mask: np.ndarray):
    _, buffer = cv2.imencode(".png", mask)
    return base64.b64encode(buffer).decode()

# ================= RESTORATION API =================
@app.post("/restore")
async def restore_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        restorer = load_restoration_model()
        img_array = np.array(img)

        if img_array.size == 0:
            return {"error": "Invalid image provided"}

        restored_np = restorer.restore(img_array)

        if restored_np.dtype != np.uint8:
            restored_np = np.clip(restored_np, 0, 255).astype(np.uint8)

        if len(restored_np.shape) == 2:
            restored_np = np.stack([restored_np] * 3, axis=-1)

        restored_img = Image.fromarray(restored_np, mode="RGB")

        return {
            "original_image": img_to_base64(img),
            "restored_image": img_to_base64(restored_img)
        }

    except Exception as e:
        return {"error": f"Restoration failed: {str(e)}"}

# ================= TAMPERING API =================
@app.post("/detect")
async def detect_tampering(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        if not img_bytes:
            return {"error": "Empty file provided"}

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if img.size[0] == 0 or img.size[1] == 0:
            return {"error": "Invalid image dimensions"}

        # ---------- 1️⃣ CLASSIFICATION ----------
        label, fm_confidence = predict_tampering(img)

        # ---------- 2️⃣ FORGERY MASK ----------
        forgery_mask = generate_forgery_mask(img)

        return {
            "label": label,
            "fm_confidence": float(fm_confidence),
            "suspect_image": img_to_base64(img),
            "forgery_mask": mask_to_base64(forgery_mask)
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Detection failed: {str(e)}",
                "label": "Error",
                "fm_confidence": 0
            }
        )
