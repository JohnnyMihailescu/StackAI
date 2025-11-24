"""Document store with thread-safe CRUD operations."""

from pathlib import Path
from typing import Optional

from app.models.document import Document, DocumentCreate
from app.storage.base_store import BaseStore
from app.storage.file_io_utils import atomic_write_json, load_json


class DocumentStore(BaseStore[Document]):
    """Thread-safe store for Document entities.

    Documents have:
    - Auto-generated integer IDs
    - Names unique within their parent library
    """

    def __init__(self, data_dir: Path, persist: bool = True) -> None:
        super().__init__(persist=persist)
        self._data_path = data_dir / "documents.json"
        # Scoped name index: (library_id, name) -> doc_id
        self._name_index: dict[tuple[int, str], int] = {}

    def _get_id(self, item: Document) -> int:
        return item.id

    def _save(self) -> None:
        data = {
            "_meta": {"next_id": self._next_id},
            "items": {str(id_): doc.model_dump(mode="json") for id_, doc in self._data.items()}
        }
        atomic_write_json(self._data_path, data)

    async def load(self) -> None:
        async with self._lock.write():
            raw_data = load_json(self._data_path)
            self._next_id = raw_data.get("_meta", {}).get("next_id", 1)
            items = raw_data.get("items", {})
            self._data = {int(id_): Document(**doc_data) for id_, doc_data in items.items()}
            self._name_index = {
                (doc.library_id, doc.name): doc.id for doc in self._data.values()
            }

    async def create(self, library_id: int, doc_create: DocumentCreate) -> Document:
        """Create a new document. Raises ValueError if name already exists in library."""
        async with self._lock.write():
            if (library_id, doc_create.name) in self._name_index:
                raise ValueError(
                    f"Document with name '{doc_create.name}' already exists in library {library_id}"
                )

            new_id = self._allocate_id()
            document = Document(
                id=new_id,
                library_id=library_id,
                name=doc_create.name,
                source=doc_create.source,
                metadata=doc_create.metadata,
            )

            self._data[new_id] = document
            self._name_index[(library_id, document.name)] = new_id

            if self._persist:
                self._save()

            return document

    async def list_by_library(self, library_id: int) -> list[Document]:
        """List all documents in a specific library."""
        async with self._lock.read():
            return [doc for doc in self._data.values() if doc.library_id == library_id]

    async def get_library_id(self, document_id: int) -> Optional[int]:
        """Get the library_id for a document. Returns None if not found."""
        async with self._lock.read():
            doc = self._data.get(document_id)
            return doc.library_id if doc else None

    async def delete(self, item_id: int) -> bool:
        """Delete a document and remove from name index."""
        async with self._lock.write():
            if item_id not in self._data:
                return False
            document = self._data[item_id]
            del self._name_index[(document.library_id, document.name)]
            del self._data[item_id]
            if self._persist:
                self._save()
            return True

    async def clear(self) -> None:
        """Clear all data and reset state."""
        async with self._lock.write():
            self._data.clear()
            self._name_index.clear()
            self._next_id = 1
