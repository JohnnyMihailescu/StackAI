"""Document store with thread-safe CRUD operations."""

from pathlib import Path
from typing import Optional

from app.models.document import Document
from app.storage.base_store import BaseStore
from app.storage.file_io_utils import atomic_write_json, load_json


class DocumentStore(BaseStore[Document]):
    """Thread-safe store for Document entities."""

    def __init__(self, data_dir: Path, persist: bool = True) -> None:
        super().__init__(persist=persist)
        self._data_path = data_dir / "documents.json"

    def _get_id(self, item: Document) -> str:
        return item.id

    def _save(self) -> None:
        data = {id_: doc.model_dump(mode="json") for id_, doc in self._data.items()}
        atomic_write_json(self._data_path, data)

    async def load(self) -> None:
        async with self._lock.write():
            raw_data = load_json(self._data_path)
            self._data = {id_: Document(**doc_data) for id_, doc_data in raw_data.items()}

    async def list_by_library(self, library_id: str) -> list[Document]:
        """List all documents in a specific library."""
        async with self._lock.read():
            return [doc for doc in self._data.values() if doc.library_id == library_id]

    async def get_library_id(self, document_id: str) -> Optional[str]:
        """Get the library_id for a document. Returns None if not found."""
        async with self._lock.read():
            doc = self._data.get(document_id)
            return doc.library_id if doc else None
