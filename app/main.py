"""
Face APIs — Passport Validation + Face Verification
Endpoints:
  POST /validate-passport       (MediaPipe — crop & enhance passport photo)
  POST /verify-face-fast-file   (InsightFace ArcFace — 1v1 face matching)
"""

import base64
import logging
import time

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.face_model import face_model

# ─── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Face APIs",
    version="2.0.0",
    description="Passport validation + ArcFace 1v1 face verification.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MediaPipe init (for /validate-passport) ─────────────────────────────────

mp_fd = mp.solutions.face_detection
face_detector = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ─── InsightFace init on startup (for /verify-face-fast-file) ────────────────

@app.on_event("startup")
async def startup():
    logger.info("Loading InsightFace model (CPU)...")
    face_model.initialize(model_name="buffalo_l", ctx_id=-1)
    logger.info("InsightFace model ready.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PASSPORT VALIDATION — /validate-passport
# ═══════════════════════════════════════════════════════════════════════════════

class ImageModel(BaseModel):
    image_base64: str


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def enhance_clarity(img_bgr):
    den = cv2.fastNlMeansDenoisingColored(img_bgr, None, 5, 5, 7, 21)
    blur = cv2.GaussianBlur(den, (0, 0), 1.2)
    sharp = cv2.addWeighted(den, 1.25, blur, -0.25, 0)
    return sharp


def encode_jpeg_under_size(img_bgr, max_kb=200):
    max_bytes = max_kb * 1024
    for q in range(92, 49, -3):
        ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok and len(enc) <= max_bytes:
            return enc.tobytes()
    return None


def crop_square_with_margin(img_bgr, detection, margin=0.55):
    h, w = img_bgr.shape[:2]
    bb = detection.location_data.relative_bounding_box
    bx = int(bb.xmin * w)
    by = int(bb.ymin * h)
    bw = int(bb.width * w)
    bh = int(bb.height * h)
    cx = bx + bw / 2
    cy = by + bh / 2
    size = max(bw, bh) * (1 + margin)
    x1 = clamp(int(cx - size / 2), 0, w - 1)
    y1 = clamp(int(cy - size / 2), 0, h - 1)
    x2 = clamp(int(cx + size / 2), 0, w - 1)
    y2 = clamp(int(cy + size / 2), 0, h - 1)
    return img_bgr[y1:y2, x1:x2]


@app.post("/validate-passport")
async def validate_passport(data: ImageModel):
    try:
        encoded = data.image_base64.split(",", 1)[-1] if "," in data.image_base64 else data.image_base64
        decoded_data = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"status": False, "message": "Invalid image data"}

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_detector.process(img_rgb)

        if not result.detections:
            return {"status": False, "message": "No face detected"}

        largest_det = max(result.detections, key=lambda d: d.location_data.relative_bounding_box.width)

        crop = crop_square_with_margin(image, largest_det, margin=0.55)
        crop = cv2.resize(crop, (600, 600), interpolation=cv2.INTER_AREA)
        crop = enhance_clarity(crop)

        final_bytes = encode_jpeg_under_size(crop, max_kb=200)
        if final_bytes is None:
            return {"status": False, "message": "Compression failed"}

        cropped_base64 = base64.b64encode(final_bytes).decode("utf-8")
        return {
            "status": True,
            "message": "Passport face validated and enhanced",
            "cropped_image": f"data:image/jpeg;base64,{cropped_base64}",
            "meta": {
                "size_kb": len(final_bytes) // 1024,
                "face_detected": len(result.detections),
            },
        }
    except Exception as e:
        return {"status": False, "message": f"Server error: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
#  FACE VERIFICATION — /verify-face-fast-file
# ═══════════════════════════════════════════════════════════════════════════════

MATCH_THRESHOLD = 0.45


def preprocess_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    original = f"{w}x{h}"

    MIN_DIM = 400
    if w < MIN_DIM or h < MIN_DIM:
        scale = max(MIN_DIM / w, MIN_DIM / h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        logger.info(f"Upscaled {original} -> {w}x{h}")

    pad_pct = 0.30
    pad_top = int(h * pad_pct)
    pad_bottom = int(h * pad_pct)
    pad_left = int(w * pad_pct)
    pad_right = int(w * pad_pct)
    img = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(128, 128, 128),
    )
    new_h, new_w = img.shape[:2]
    logger.info(f"Padded -> {new_w}x{new_h}")

    MAX_DIM = 1280
    if new_w > MAX_DIM or new_h > MAX_DIM:
        scale = min(MAX_DIM / new_w, MAX_DIM / new_h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return img


def get_embedding(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    original_size = f"{w}x{h}"
    processed = preprocess_image(img)
    result = face_model.get_embedding_with_metadata(processed)
    result["image_size"] = original_size
    result["processed_size"] = f"{processed.shape[1]}x{processed.shape[0]}"
    return result


def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-10)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-10)
    return float(np.dot(emb1, emb2))


def build_response(result1: dict, result2: dict, similarity: float, elapsed: float) -> dict:
    is_match = similarity >= MATCH_THRESHOLD
    logger.info(
        f"VERIFY | similarity={similarity:.4f} | threshold={MATCH_THRESHOLD} | "
        f"match={is_match} | "
        f"face1_size={result1.get('face_size')} det={result1.get('det_score', 0):.3f} | "
        f"face2_size={result2.get('face_size')} det={result2.get('det_score', 0):.3f} | "
        f"time={elapsed:.0f}ms"
    )
    return {
        "match": is_match,
        "similarity": round(similarity, 4),
        "threshold": MATCH_THRESHOLD,
        "det_score_1": round(result1.get("det_score", 0), 4),
        "det_score_2": round(result2.get("det_score", 0), 4),
        "face_size_1": result1.get("face_size"),
        "face_size_2": result2.get("face_size"),
        "image_size_1": result1.get("image_size"),
        "image_size_2": result2.get("image_size"),
        "processed_size_1": result1.get("processed_size"),
        "processed_size_2": result2.get("processed_size"),
        "processing_time_ms": round(elapsed, 1),
    }


@app.post("/verify-face-fast-file")
async def verify_face_file(
    captured_image: UploadFile = File(...),
    existing_image: UploadFile = File(...),
):
    """Verify two faces via file upload."""
    start = time.perf_counter()

    bytes1 = await captured_image.read()
    bytes2 = await existing_image.read()

    img1 = cv2.imdecode(np.frombuffer(bytes1, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(bytes2, np.uint8), cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        return {"match": False, "reason": "Invalid image file"}

    result1 = get_embedding(img1)
    if not result1["success"]:
        return {"match": False, "reason": f"captured_image: {result1.get('reason', 'no_face')}", "face_size": result1.get("face_size"), "image_size": result1.get("image_size")}

    result2 = get_embedding(img2)
    if not result2["success"]:
        return {"match": False, "reason": f"existing_image: {result2.get('reason', 'no_face')}", "face_size": result2.get("face_size"), "image_size": result2.get("image_size")}

    similarity = compare_embeddings(result1["embedding"], result2["embedding"])
    elapsed = (time.perf_counter() - start) * 1000

    return build_response(result1, result2, similarity, elapsed)


# ─── Health Check ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": True,
        "message": "Face APIs running",
        "endpoints": ["/validate-passport", "/verify-face-fast-file"],
    }


# ─── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7654)
