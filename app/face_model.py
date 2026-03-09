"""
FILE: backend/app/face_model.py
InsightFace model loading and management (RetinaFace + ArcFace)
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceModel:
    """
    Singleton wrapper around InsightFace's buffalo_l model pack.
    - Detection: RetinaFace
    - Recognition: ArcFace (512-dim embeddings)
    - Anti-Spoofing: passive liveness via reflection / texture analysis
    """

    _instance: Optional["FaceModel"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def initialize(self, model_name: str = "buffalo_l", ctx_id: int = 0):
        """Load InsightFace model pack. ctx_id=0 for GPU, -1 for CPU."""
        if self._initialized:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis

            logger.info(f"Loading InsightFace model pack: {model_name} (ctx_id={ctx_id})")
            start = time.time()

            self.app = FaceAnalysis(
                name=model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id >= 0
                          else ["CPUExecutionProvider"]
            )
            # det_size: minimum face size the detector will look for
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

            self.embedding_size = 512
            self.model_name = model_name
            self._initialized = True

            elapsed = time.time() - start
            logger.info(f"InsightFace loaded successfully in {elapsed:.2f}s")

        except ImportError:
            logger.warning("InsightFace not installed. Running in MOCK mode.")
            self._mock_mode = True
            self._initialized = True
            self.embedding_size = 512
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    # ─── Core Methods ───────────────────────────────────────────────────────────

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw image bytes to BGR numpy array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image. Invalid or corrupted file.")
        return img

    def detect_faces(self, img: np.ndarray) -> list:
        """
        Run RetinaFace detection on image.
        Returns list of face objects with .bbox, .kps, .det_score, .embedding
        """
        if getattr(self, "_mock_mode", False):
            return self._mock_detect(img)

        faces = self.app.get(img)
        return faces

    def get_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ArcFace 512-dim embedding from a single-face image.
        Returns normalized embedding or None if no face found.
        """
        faces = self.detect_faces(img)

        if len(faces) == 0:
            logger.debug("No face detected in image")
            return None

        if len(faces) > 1:
            logger.warning(f"Multiple faces ({len(faces)}) detected — rejecting")
            raise ValueError(f"Multiple faces detected ({len(faces)}). Please submit an image with exactly one face.")

        face = faces[0]

        if face.det_score < 0.5:
            logger.debug(f"Face detection confidence too low: {face.det_score:.3f}")
            return None

        embedding = face.normed_embedding  # Already L2-normalized by InsightFace
        return embedding.astype(np.float32)

    def get_embedding_with_metadata(self, img: np.ndarray) -> dict:
        """
        Full analysis: embedding + bounding box + landmarks + quality.
        """
        faces = self.detect_faces(img)

        if len(faces) == 0:
            return {"success": False, "reason": "no_face_detected", "face_count": 0}

        if len(faces) > 1:
            return {"success": False, "reason": "multiple_faces", "face_count": len(faces)}

        face = faces[0]

        if face.det_score < 0.5:
            return {"success": False, "reason": "low_confidence", "det_score": float(face.det_score)}

        # Check minimum face size
        bbox = face.bbox.astype(int)
        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]

        if face_w < 40 or face_h < 40:
            return {
                "success": False,
                "reason": "face_too_small",
                "face_size": f"{face_w}x{face_h}"
            }

        embedding = face.normed_embedding.astype(np.float32)

        return {
            "success": True,
            "embedding": embedding,
            "det_score": float(face.det_score),
            "bbox": bbox.tolist(),
            "landmarks": face.kps.tolist() if face.kps is not None else None,
            "face_size": f"{face_w}x{face_h}",
            "face_count": 1,
        }

    # ─── Liveness Detection ─────────────────────────────────────────────────────

    def estimate_liveness(self, img: np.ndarray, face) -> float:
        """
        Basic passive liveness estimation using texture analysis.
        Returns score 0.0–1.0 (higher = more likely real face).

        For production, integrate a dedicated anti-spoofing model
        (e.g., Silent-Face-Anti-Spoofing or FAS-SGTD).
        """
        try:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Crop face region
            face_crop = img[max(0, y1):y2, max(0, x1):x2]
            if face_crop.size == 0:
                return 0.5

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

            # 1. Laplacian variance (sharpness) — printed photos are often blurry
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(lap_var / 500.0, 1.0)

            # 2. LBP texture analysis — screens have repetitive texture patterns
            lbp_score = self._compute_lbp_score(gray)

            # 3. Reflection analysis — screens often have uniform lighting
            reflection_score = self._analyze_specular_reflection(face_crop)

            # Weighted combination
            liveness_score = (
                0.4 * sharpness_score +
                0.35 * lbp_score +
                0.25 * reflection_score
            )

            return float(np.clip(liveness_score, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"Liveness estimation failed: {e}")
            return 0.5  # Neutral score on error

    def _compute_lbp_score(self, gray: np.ndarray) -> float:
        """Local Binary Pattern uniformity score."""
        try:
            resized = cv2.resize(gray, (64, 64))
            # Simplified LBP: compare each pixel to its neighbors
            neighbors = [
                np.roll(resized, (dy, dx), axis=(0, 1))
                for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                if not (dy == 0 and dx == 0)
            ]
            lbp = sum((resized > n).astype(np.uint8) for n in neighbors)
            hist, _ = np.histogram(lbp.ravel(), bins=8, range=(0, 8))
            hist = hist.astype(float) / hist.sum()
            # High entropy = more natural texture
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return min(entropy / 3.0, 1.0)
        except Exception:
            return 0.5

    def _analyze_specular_reflection(self, face_crop: np.ndarray) -> float:
        """Check for screen-like uniform high-intensity regions."""
        try:
            hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            # High percentage of very bright pixels = possible screen
            bright_ratio = np.sum(v_channel > 240) / v_channel.size
            # Natural faces have < 5% extremely bright pixels
            return float(1.0 - min(bright_ratio * 20, 1.0))
        except Exception:
            return 0.5

    # ─── Mock Mode ──────────────────────────────────────────────────────────────

    def _mock_detect(self, img: np.ndarray) -> list:
        """Return a mock face detection result for testing without GPU."""
        class MockFace:
            det_score = 0.99
            normed_embedding = np.random.randn(512).astype(np.float32)
            normed_embedding /= np.linalg.norm(normed_embedding)
            bbox = np.array([100, 100, 300, 300], dtype=np.float32)
            kps = np.zeros((5, 2), dtype=np.float32)

        logger.debug("MOCK: Returning synthetic face detection result")
        return [MockFace()]

    # ─── Utility ────────────────────────────────────────────────────────────────

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(emb1, emb2))

    @property
    def is_ready(self) -> bool:
        return self._initialized


# Module-level singleton
face_model = FaceModel()
