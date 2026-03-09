"""
FILE: backend/app/face_matching.py
Face verification and matching logic using FAISS + ArcFace
"""

import logging
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import Employee
from app.embedding_service import embedding_service
from app.face_model import face_model

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a face matching operation."""
    matched: bool
    employee_id: Optional[str]
    employee_name: Optional[str]
    employee_code: Optional[str]
    department: Optional[str]
    similarity_score: float
    liveness_score: float
    det_score: float
    reason: Optional[str]          # Why it failed if matched=False
    processing_time_ms: float


class FaceMatchingService:
    """
    Orchestrates the complete face verification pipeline:
    1. Decode image
    2. Detect face (RetinaFace)
    3. Liveness check
    4. Extract embedding (ArcFace)
    5. FAISS nearest-neighbor search
    6. Threshold decision
    7. Load employee from DB
    """

    async def verify_face(
        self,
        image_bytes: bytes,
        company_id: str,
        db: AsyncSession,
        device_id: Optional[str] = None,
    ) -> MatchResult:
        """
        Main entry point: verify a face image against all registered employees.
        """
        import time
        start = time.perf_counter()

        # ── Step 1: Extract embedding + liveness ────────────────────────────────
        extraction_result = await embedding_service.extract_embedding(image_bytes)

        if not extraction_result["success"]:
            elapsed = (time.perf_counter() - start) * 1000
            reason = extraction_result.get("reason", "extraction_failed")
            return MatchResult(
                matched=False,
                employee_id=None,
                employee_name=None,
                employee_code=None,
                department=None,
                similarity_score=0.0,
                liveness_score=extraction_result.get("liveness_score", 0.0),
                det_score=extraction_result.get("det_score", 0.0),
                reason=self._translate_reason(reason),
                processing_time_ms=elapsed,
            )

        embedding: np.ndarray = extraction_result["embedding"]
        det_score: float = extraction_result.get("det_score", 0.0)
        liveness_score: float = extraction_result.get("liveness_score", 1.0)

        # ── Step 2: FAISS search ─────────────────────────────────────────────────
        candidates = await embedding_service.search(embedding, top_k=5)

        if not candidates:
            elapsed = (time.perf_counter() - start) * 1000
            return MatchResult(
                matched=False,
                employee_id=None,
                employee_name=None,
                employee_code=None,
                department=None,
                similarity_score=0.0,
                liveness_score=liveness_score,
                det_score=det_score,
                reason="No employees registered in the system",
                processing_time_ms=elapsed,
            )

        best = candidates[0]
        similarity = best["similarity_score"]

        # ── Step 3: Threshold decision ───────────────────────────────────────────
        if similarity < settings.FACE_MATCH_THRESHOLD:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                f"Face not recognized. Best similarity: {similarity:.4f} "
                f"(threshold: {settings.FACE_MATCH_THRESHOLD})"
            )
            return MatchResult(
                matched=False,
                employee_id=None,
                employee_name=None,
                employee_code=None,
                department=None,
                similarity_score=similarity,
                liveness_score=liveness_score,
                det_score=det_score,
                reason="Face not recognized",
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        # ── Step 4: Load matched employee from DB ────────────────────────────────
        employee_uuid = best["employee_id"]
        employee = await self._load_employee(employee_uuid, company_id, db)

        elapsed = (time.perf_counter() - start) * 1000

        if employee is None:
            return MatchResult(
                matched=False,
                employee_id=None,
                employee_name=None,
                employee_code=None,
                department=None,
                similarity_score=similarity,
                liveness_score=liveness_score,
                det_score=det_score,
                reason="Employee account not found or inactive",
                processing_time_ms=elapsed,
            )

        # ── Step 5: Device binding check ─────────────────────────────────────────
        if device_id and employee.registered_device_id:
            if employee.registered_device_id != device_id:
                logger.warning(
                    f"Device mismatch for employee {employee_uuid}: "
                    f"expected {employee.registered_device_id}, got {device_id}"
                )
                return MatchResult(
                    matched=False,
                    employee_id=str(employee.employee_id),
                    employee_name=employee.name,
                    employee_code=employee.employee_code,
                    department=employee.department,
                    similarity_score=similarity,
                    liveness_score=liveness_score,
                    det_score=det_score,
                    reason="Device not authorized for this employee",
                    processing_time_ms=elapsed,
                )

        logger.info(
            f"Face matched: {employee.name} ({employee_uuid}) "
            f"similarity={similarity:.4f} liveness={liveness_score:.4f} "
            f"time={elapsed:.1f}ms"
        )

        return MatchResult(
            matched=True,
            employee_id=str(employee.employee_id),
            employee_name=employee.name,
            employee_code=employee.employee_code,
            department=employee.department,
            similarity_score=similarity,
            liveness_score=liveness_score,
            det_score=det_score,
            reason=None,
            processing_time_ms=elapsed,
        )

    async def verify_1v1(
        self,
        image_bytes: bytes,
        stored_embedding: list[float],
    ) -> dict:
        """
        1-vs-1 verification: compare image against a specific stored embedding.
        Used for employee re-registration confirmation.
        """
        extraction_result = await embedding_service.extract_embedding(image_bytes)

        if not extraction_result["success"]:
            return {"matched": False, "similarity": 0.0, "reason": extraction_result.get("reason")}

        query_emb = extraction_result["embedding"]
        stored_emb = np.array(stored_embedding, dtype=np.float32)

        # Normalize both
        query_emb /= np.linalg.norm(query_emb) + 1e-10
        stored_emb /= np.linalg.norm(stored_emb) + 1e-10

        similarity = float(np.dot(query_emb, stored_emb))

        return {
            "matched": similarity >= settings.FACE_MATCH_THRESHOLD,
            "similarity": similarity,
            "reason": None if similarity >= settings.FACE_MATCH_THRESHOLD else "Faces do not match",
        }

    # ─── Helpers ────────────────────────────────────────────────────────────────

    async def _load_employee(
        self, employee_id: str, company_id: str, db: AsyncSession
    ) -> Optional[Employee]:
        """Load employee from DB, ensuring they belong to the correct company."""
        result = await db.execute(
            select(Employee).where(
                Employee.employee_id == employee_id,
                Employee.company_id == company_id,
                Employee.is_active == True,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _translate_reason(reason: str) -> str:
        translations = {
            "no_face_detected": "No face detected in the image. Please ensure your face is clearly visible.",
            "multiple_faces": "Multiple faces detected. Please ensure only your face is in the frame.",
            "low_confidence": "Face detection confidence too low. Please improve lighting and try again.",
            "face_too_small": "Face is too small in the image. Please move closer to the camera.",
            "liveness_check_failed": "Liveness check failed. Please use a live camera (no photos or screens).",
            "internal_error": "An internal error occurred during face processing.",
        }
        return translations.get(reason, reason)


# Module-level singleton
face_matching_service = FaceMatchingService()
