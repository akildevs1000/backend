"""
FILE: backend/app/embedding_service.py
Manages employee face embeddings: storage, FAISS index building, persistence
"""

import asyncio
import logging
import os
import pickle
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import Employee
from app.face_model import face_model

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Manages the full lifecycle of face embeddings:
    1. Extracting embeddings from images via InsightFace
    2. Storing embeddings in PostgreSQL
    3. Building and maintaining a FAISS index for fast search
    4. Mapping FAISS index IDs back to employee UUIDs
    """

    def __init__(self):
        self.faiss_index = None
        self.faiss_index_path = Path(settings.FAISS_INDEX_PATH)
        self.id_map_path = self.faiss_index_path.with_suffix(".map.pkl")
        self.id_map: list[str] = []  # faiss_id -> employee_uuid
        self._lock = asyncio.Lock()
        self._dirty = False  # True when in-memory index differs from persisted

    # ─── Embedding Extraction ───────────────────────────────────────────────────

    async def extract_embedding(self, image_bytes: bytes) -> dict:
        """
        Extract face embedding from raw image bytes.
        Returns dict with embedding and metadata, or error info.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._sync_extract_embedding, image_bytes
        )
        return result

    def _sync_extract_embedding(self, image_bytes: bytes) -> dict:
        """Synchronous embedding extraction (runs in thread pool)."""
        try:
            img = face_model.decode_image(image_bytes)
            result = face_model.get_embedding_with_metadata(img)

            if not result["success"]:
                return result

            # Run liveness check
            if settings.LIVENESS_ENABLED and result.get("success"):
                faces = face_model.detect_faces(img)
                if faces:
                    liveness_score = face_model.estimate_liveness(img, faces[0])
                    result["liveness_score"] = liveness_score
                    if liveness_score < settings.LIVENESS_THRESHOLD:
                        result["success"] = False
                        result["reason"] = "liveness_check_failed"
                        result["liveness_score"] = liveness_score
                        return result

            return result

        except ValueError as e:
            return {"success": False, "reason": str(e)}
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}", exc_info=True)
            return {"success": False, "reason": "internal_error"}

    # ─── FAISS Index Management ─────────────────────────────────────────────────

    async def build_index(self, db: AsyncSession) -> int:
        """
        (Re)build FAISS index from all employee embeddings in the database.
        Returns number of employees indexed.
        """
        import faiss

        async with self._lock:
            logger.info("Building FAISS index from database embeddings...")

            result = await db.execute(
                select(Employee).where(
                    Employee.face_embedding.isnot(None),
                    Employee.is_active == True
                )
            )
            employees = result.scalars().all()

            if not employees:
                logger.warning("No employee embeddings found. FAISS index will be empty.")
                self.faiss_index = faiss.IndexFlatIP(settings.EMBEDDING_SIZE)
                self.id_map = []
                return 0

            embeddings = []
            id_map = []

            for emp in employees:
                emb = np.array(emp.face_embedding, dtype=np.float32)
                if emb.shape[0] == settings.EMBEDDING_SIZE:
                    # Ensure L2 normalization
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    embeddings.append(emb)
                    id_map.append(str(emp.employee_id))

            if not embeddings:
                self.faiss_index = faiss.IndexFlatIP(settings.EMBEDDING_SIZE)
                self.id_map = []
                return 0

            matrix = np.vstack(embeddings).astype(np.float32)
            n = matrix.shape[0]

            # Choose index type based on dataset size
            if n < 1000 or settings.FAISS_INDEX_TYPE == "Flat":
                # Exact search for small datasets
                index = faiss.IndexFlatIP(settings.EMBEDDING_SIZE)
            else:
                # IVFFlat for large datasets (100K+ employees)
                nlist = min(settings.FAISS_NLIST, n // 10)
                quantizer = faiss.IndexFlatIP(settings.EMBEDDING_SIZE)
                index = faiss.IndexIVFFlat(quantizer, settings.EMBEDDING_SIZE, nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(matrix)
                index.nprobe = settings.FAISS_NPROBE

            index.add(matrix)
            self.faiss_index = index
            self.id_map = id_map
            self._dirty = True

            logger.info(f"FAISS index built: {n} employees indexed")
            await self._save_index()
            return n

    async def add_to_index(self, employee_id: str, embedding: np.ndarray, db: AsyncSession) -> int:
        """
        Add a single employee embedding to the FAISS index.
        Returns the assigned FAISS index ID.
        """
        import faiss

        async with self._lock:
            if self.faiss_index is None:
                await self.build_index(db)

            emb = embedding.copy().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            # If using IVFFlat, we need to check if it's trained
            if hasattr(self.faiss_index, "is_trained") and not self.faiss_index.is_trained:
                await self.build_index(db)
                return len(self.id_map) - 1

            self.faiss_index.add(emb.reshape(1, -1))
            faiss_id = len(self.id_map)
            self.id_map.append(employee_id)
            self._dirty = True

            await self._save_index()
            return faiss_id

    async def remove_from_index(self, employee_id: str, db: AsyncSession):
        """Remove employee from FAISS index (requires full rebuild)."""
        async with self._lock:
            # FAISS doesn't support efficient single-element removal
            # Rebuild index excluding this employee
            await self.build_index(db)

    async def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search FAISS index for nearest neighbors.
        Returns list of {employee_id, similarity_score} sorted by score desc.
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            logger.warning("FAISS index is empty or not loaded")
            return []

        emb = query_embedding.copy().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        loop = asyncio.get_event_loop()
        distances, indices = await loop.run_in_executor(
            None,
            lambda: self.faiss_index.search(emb.reshape(1, -1), min(top_k, self.faiss_index.ntotal))
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            results.append({
                "employee_id": self.id_map[idx],
                "similarity_score": float(dist),
                "faiss_id": int(idx),
            })

        return results

    # ─── Persistence ────────────────────────────────────────────────────────────

    async def load_index(self):
        """Load FAISS index from disk if it exists."""
        import faiss

        if self.faiss_index_path.exists() and self.id_map_path.exists():
            try:
                loop = asyncio.get_event_loop()
                index = await loop.run_in_executor(
                    None, faiss.read_index, str(self.faiss_index_path)
                )
                with open(self.id_map_path, "rb") as f:
                    id_map = pickle.load(f)

                async with self._lock:
                    self.faiss_index = index
                    self.id_map = id_map

                logger.info(f"FAISS index loaded: {index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. Will rebuild from DB.")
        else:
            logger.info("No existing FAISS index found. Will build on first request.")

    async def _save_index(self):
        """Persist FAISS index to disk."""
        import faiss

        if self.faiss_index is None:
            return

        try:
            self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, faiss.write_index, self.faiss_index, str(self.faiss_index_path)
            )
            with open(self.id_map_path, "wb") as f:
                pickle.dump(self.id_map, f)
            self._dirty = False
            logger.debug(f"FAISS index saved: {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    @property
    def index_size(self) -> int:
        return self.faiss_index.ntotal if self.faiss_index else 0


# Module-level singleton
embedding_service = EmbeddingService()
