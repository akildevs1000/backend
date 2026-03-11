"""
Face Verification API — File Upload Only
Uses InsightFace (ArcFace) for high-accuracy 1v1 face matching.
Single endpoint: POST /verify-face-fast-file
"""

import logging
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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
    title="Face Verification API",
    version="1.0.0",
    description="ArcFace-powered 1v1 face verification via file upload.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Init model on startup ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("Loading InsightFace model (CPU)...")
    face_model.initialize(model_name="buffalo_l", ctx_id=-1)
    logger.info("Model ready.")


# ─── Config ────────────────────────────────────────────────────────────────────

MATCH_THRESHOLD = 0.45  # ArcFace cosine similarity threshold (0.45 = relaxed, 0.6 = strict)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Prepare image for face detection.
    - Passport/small photos: upscale + add padding so RetinaFace can detect the face.
    - Large images: scale down to save processing time.
    """
    h, w = img.shape[:2]
    original = f"{w}x{h}"

    # Step 1: Upscale tiny images (passport photos are often < 400px)
    MIN_DIM = 400
    if w < MIN_DIM or h < MIN_DIM:
        scale = max(MIN_DIM / w, MIN_DIM / h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        logger.info(f"Upscaled {original} -> {w}x{h}")

    # Step 2: Add padding around the image (helps RetinaFace with tightly-cropped faces)
    pad_pct = 0.30  # 30% padding on each side
    pad_top = int(h * pad_pct)
    pad_bottom = int(h * pad_pct)
    pad_left = int(w * pad_pct)
    pad_right = int(w * pad_pct)
    img = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(128, 128, 128)  # neutral gray padding
    )
    new_h, new_w = img.shape[:2]
    logger.info(f"Padded -> {new_w}x{new_h}")

    # Step 3: Scale down very large images to save time
    MAX_DIM = 1280
    if new_w > MAX_DIM or new_h > MAX_DIM:
        scale = min(MAX_DIM / new_w, MAX_DIM / new_h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return img


def get_embedding(img: np.ndarray) -> dict:
    """Extract ArcFace embedding from image. Returns dict with embedding or error."""
    h, w = img.shape[:2]
    original_size = f"{w}x{h}"

    # Preprocess for passport/small photos
    processed = preprocess_image(img)

    result = face_model.get_embedding_with_metadata(processed)
    result["image_size"] = original_size
    result["processed_size"] = f"{processed.shape[1]}x{processed.shape[0]}"
    return result


def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings."""
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-10)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-10)
    return float(np.dot(emb1, emb2))


def build_response(result1: dict, result2: dict, similarity: float, elapsed: float) -> dict:
    """Build detailed response with debug info."""
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


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": True,
        "message": "Face Verification API is running",
        "model": "InsightFace ArcFace (buffalo_l)",
        "endpoints": ["/verify-face-fast-file"],
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


# ─── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
