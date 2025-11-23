"""Chunk store with thread-safe CRUD operations.

Chunks store text and metadata only. Embeddings are managed by SearchService.
"""

from pathlib import Path
from typing import Optional

from app.models.chunk import Chunk
from app.storage.base_store import BaseStore
from app.storage.file_io_utils import atomic_write_json, load_json


class ChunkStore(BaseStore[Chunk]):
    """Thread-safe store for Chunk entities.

    Note: Embeddings are not stored here. They are managed by SearchService
    and persisted in the index files.
    """

    def __init__(self, data_dir: Path, persist: bool = True) -> None:
        super().__init__(persist=persist)
        self._data_path = data_dir / "chunks.json"

    def _get_id(self, item: Chunk) -> str:
        return item.id

    def _save(self) -> None:
        data = {id_: chunk.model_dump(mode="json") for id_, chunk in self._data.items()}
        atomic_write_json(self._data_path, data)

    async def load(self) -> None:
        async with self._lock.write():
            raw_data = load_json(self._data_path)
            self._data = {cid: Chunk(**data) for cid, data in raw_data.items()}

    async def create(self, chunk: Chunk) -> Chunk:
        """Create a new chunk (without embedding - that's handled by SearchService)."""
        async with self._lock.write():
            if chunk.id in self._data:
                raise ValueError(f"Chunk with id '{chunk.id}' already exists")

            # Store chunk without embedding
            chunk_for_storage = chunk.model_copy()
            chunk_for_storage.embedding = None
            self._data[chunk.id] = chunk_for_storage

            if self._persist:
                self._save()

            return chunk

    async def create_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """Create multiple chunks atomically."""
        async with self._lock.write():
            existing = [c.id for c in chunks if c.id in self._data]
            if existing:
                raise ValueError(f"Chunks with these IDs already exist: {existing}")

            for chunk in chunks:
                chunk_for_storage = chunk.model_copy()
                chunk_for_storage.embedding = None
                self._data[chunk.id] = chunk_for_storage

            if self._persist:
                self._save()

            return chunks

    async def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID."""
        async with self._lock.read():
            chunk = self._data.get(chunk_id)
            if chunk is None:
                return None
            return chunk.model_copy()

    async def list_all(self) -> list[Chunk]:
        """List all chunks."""
        async with self._lock.read():
            return [c.model_copy() for c in self._data.values()]

    async def list_by_document(self, document_id: str) -> list[Chunk]:
        """List all chunks in a specific document."""
        async with self._lock.read():
            return [
                c.model_copy()
                for c in self._data.values()
                if c.document_id == document_id
            ]

    async def delete(self, chunk_id: str) -> bool:
        """Delete a chunk."""
        async with self._lock.write():
            if chunk_id not in self._data:
                return False
            del self._data[chunk_id]
            if self._persist:
                self._save()
            return True

    async def delete_by_document(self, document_id: str) -> list[str]:
        """Delete all chunks for a document. Returns deleted chunk IDs."""
        async with self._lock.write():
            to_delete = [cid for cid, c in self._data.items() if c.document_id == document_id]
            for chunk_id in to_delete:
                del self._data[chunk_id]
            if self._persist and to_delete:
                self._save()
            return to_delete

    async def find_existing(self, chunk_ids: list[str]) -> list[str]:
        """Return which of the given IDs already exist."""
        async with self._lock.read():
            return [cid for cid in chunk_ids if cid in self._data]

    async def clear(self) -> None:
        """Clear all data."""
        async with self._lock.write():
            self._data.clear()
