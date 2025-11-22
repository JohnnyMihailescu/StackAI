"""Chunk store with thread-safe CRUD operations and separate embedding storage.

Chunks are stored differently from other entities:
- Metadata (id, document_id, text, etc.) goes to JSON
- Embeddings go to a separate NumPy binary file for efficiency
"""

from pathlib import Path
from typing import Optional

import numpy as np

from app.models.chunk import Chunk
from app.storage.base_store import BaseStore
from app.storage.file_io_utils import (
    atomic_write_json,
    load_json,
    atomic_write_numpy,
    load_numpy,
)


class ChunkStore(BaseStore[Chunk]):
    """Thread-safe store for Chunk entities with separate embedding storage."""

    def __init__(self, data_dir: Path, persist: bool = True) -> None:
        super().__init__(persist=persist)
        self._data_path = data_dir / "chunks.json"
        self._embeddings_path = data_dir / "embeddings.npz"
        self._embeddings: dict[str, list[float]] = {}

    def _get_id(self, item: Chunk) -> str:
        return item.id

    def _save(self) -> None:
        # Save chunk metadata (without embeddings)
        data = {id_: chunk.model_dump(mode="json") for id_, chunk in self._data.items()}
        atomic_write_json(self._data_path, data)

    def _save_embeddings(self) -> None:
        ids = list(self._embeddings.keys())
        if ids:
            vectors = np.array([self._embeddings[cid] for cid in ids])
        else:
            vectors = np.array([])
        atomic_write_numpy(self._embeddings_path, ids, vectors)

    async def load(self) -> None:
        async with self._lock.write():
            # Load chunk metadata
            raw_data = load_json(self._data_path)
            self._data = {cid: Chunk(**data) for cid, data in raw_data.items()}

            # Load embeddings
            ids, vectors = load_numpy(self._embeddings_path)
            if len(ids) > 0:
                self._embeddings = {id_: vec.tolist() for id_, vec in zip(ids, vectors)}
            else:
                self._embeddings = {}

    async def create(self, chunk: Chunk) -> Chunk:
        """Create a new chunk, storing embedding separately."""
        async with self._lock.write():
            if chunk.id in self._data:
                raise ValueError(f"Chunk with id '{chunk.id}' already exists")

            embedding = chunk.embedding
            chunk_for_storage = chunk.model_copy()
            chunk_for_storage.embedding = None
            self._data[chunk.id] = chunk_for_storage

            if embedding:
                self._embeddings[chunk.id] = embedding

            if self._persist:
                self._save()
                if embedding:
                    self._save_embeddings()

            return chunk

    async def create_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """Create multiple chunks atomically."""
        async with self._lock.write():
            existing = [c.id for c in chunks if c.id in self._data]
            if existing:
                raise ValueError(f"Chunks with these IDs already exist: {existing}")

            has_embeddings = False
            for chunk in chunks:
                embedding = chunk.embedding
                chunk_for_storage = chunk.model_copy()
                chunk_for_storage.embedding = None
                self._data[chunk.id] = chunk_for_storage

                if embedding:
                    self._embeddings[chunk.id] = embedding
                    has_embeddings = True

            if self._persist:
                self._save()
                if has_embeddings:
                    self._save_embeddings()

            return chunks

    async def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID, with embedding attached."""
        async with self._lock.read():
            chunk = self._data.get(chunk_id)
            if chunk is None:
                return None
            result = chunk.model_copy()
            result.embedding = self._embeddings.get(chunk_id)
            return result

    async def list_all(self) -> list[Chunk]:
        """List all chunks with embeddings attached."""
        async with self._lock.read():
            return self._attach_embeddings(list(self._data.values()))

    async def list_by_document(self, document_id: str) -> list[Chunk]:
        """List all chunks in a specific document."""
        async with self._lock.read():
            chunks = [c for c in self._data.values() if c.document_id == document_id]
            return self._attach_embeddings(chunks)

    async def delete(self, chunk_id: str) -> bool:
        """Delete a chunk and its embedding."""
        async with self._lock.write():
            if chunk_id not in self._data:
                return False
            del self._data[chunk_id]
            self._embeddings.pop(chunk_id, None)
            if self._persist:
                self._save()
                self._save_embeddings()
            return True

    async def delete_by_document(self, document_id: str) -> list[str]:
        """Delete all chunks for a document. Returns deleted chunk IDs."""
        async with self._lock.write():
            to_delete = [cid for cid, c in self._data.items() if c.document_id == document_id]
            for chunk_id in to_delete:
                del self._data[chunk_id]
                self._embeddings.pop(chunk_id, None)
            if self._persist and to_delete:
                self._save()
                self._save_embeddings()
            return to_delete

    async def find_existing(self, chunk_ids: list[str]) -> list[str]:
        """Return which of the given IDs already exist."""
        async with self._lock.read():
            return [cid for cid in chunk_ids if cid in self._data]

    async def get_embedding(self, chunk_id: str) -> Optional[list[float]]:
        """Get just the embedding for a chunk."""
        async with self._lock.read():
            return self._embeddings.get(chunk_id)

    async def get_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Get all embeddings as (ids, vectors) for index building."""
        async with self._lock.read():
            if not self._embeddings:
                return [], np.array([])
            ids = list(self._embeddings.keys())
            vectors = np.array([self._embeddings[cid] for cid in ids])
            return ids, vectors

    def _attach_embeddings(self, chunks: list[Chunk]) -> list[Chunk]:
        """Attach embeddings to chunks. Must be called within lock."""
        result = []
        for chunk in chunks:
            c = chunk.model_copy()
            c.embedding = self._embeddings.get(chunk.id)
            result.append(c)
        return result

    async def clear(self) -> None:
        """Clear all data including embeddings."""
        async with self._lock.write():
            self._data.clear()
            self._embeddings.clear()
