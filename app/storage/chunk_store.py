"""Chunk store with thread-safe CRUD operations.

Chunks store text and metadata only. Embeddings are managed by SearchService.
"""

from pathlib import Path
from typing import Optional

from app.models.chunk import Chunk, ChunkCreate
from app.storage.base_store import BaseStore
from app.storage.file_io_utils import atomic_write_json, load_json


class ChunkStore(BaseStore[Chunk]):
    """Thread-safe store for Chunk entities.

    Chunks have:
    - Auto-generated integer IDs
    - No name-based access (chunks don't have meaningful names)

    Note: Embeddings are not stored here. They are managed by SearchService
    and persisted in the index files.
    """

    def __init__(self, data_dir: Path, persist: bool = True) -> None:
        super().__init__(persist=persist)
        self._data_path = data_dir / "chunks.json"

    def _get_id(self, item: Chunk) -> int:
        return item.id

    def _save(self) -> None:
        data = {
            "_meta": {"next_id": self._next_id},
            "items": {str(id_): chunk.for_storage() for id_, chunk in self._data.items()}
        }
        atomic_write_json(self._data_path, data)

    async def load(self) -> None:
        async with self._lock.write():
            raw_data = load_json(self._data_path)
            self._next_id = raw_data.get("_meta", {}).get("next_id", 1)
            items = raw_data.get("items", {})
            self._data = {int(cid): Chunk(**data) for cid, data in items.items()}

    async def create(self, document_id: int, chunk_create: ChunkCreate) -> Chunk:
        """Create a new chunk (embedding handled separately by SearchService)."""
        async with self._lock.write():
            new_id = self._allocate_id()
            chunk = Chunk(
                id=new_id,
                document_id=document_id,
                text=chunk_create.text,
                metadata=chunk_create.metadata,
            )

            self._data[new_id] = chunk

            if self._persist:
                self._save()

            return chunk.model_copy()

    async def create_batch(self, document_id: int, chunk_creates: list[ChunkCreate]) -> list[Chunk]:
        """Create multiple chunks atomically."""
        async with self._lock.write():
            chunks = []
            for chunk_create in chunk_creates:
                new_id = self._allocate_id()
                chunk = Chunk(
                    id=new_id,
                    document_id=document_id,
                    text=chunk_create.text,
                    metadata=chunk_create.metadata,
                )
                self._data[new_id] = chunk
                chunks.append(chunk)

            if self._persist:
                self._save()

            return [c.model_copy() for c in chunks]

    async def get(self, chunk_id: int) -> Optional[Chunk]:
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

    async def list_by_document(self, document_id: int) -> list[Chunk]:
        """List all chunks in a specific document."""
        async with self._lock.read():
            return [
                c.model_copy()
                for c in self._data.values()
                if c.document_id == document_id
            ]

    async def delete(self, chunk_id: int) -> bool:
        """Delete a chunk."""
        async with self._lock.write():
            if chunk_id not in self._data:
                return False
            del self._data[chunk_id]
            if self._persist:
                self._save()
            return True

    async def delete_by_document(self, document_id: int) -> list[int]:
        """Delete all chunks for a document. Returns deleted chunk IDs."""
        async with self._lock.write():
            to_delete = [cid for cid, c in self._data.items() if c.document_id == document_id]
            for chunk_id in to_delete:
                del self._data[chunk_id]
            if self._persist and to_delete:
                self._save()
            return to_delete

    async def clear(self) -> None:
        """Clear all data."""
        async with self._lock.write():
            self._data.clear()
            self._next_id = 1
