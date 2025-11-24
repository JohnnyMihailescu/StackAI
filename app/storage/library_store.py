"""Library store with thread-safe CRUD operations."""

from pathlib import Path
from typing import Optional

from app.models.library import Library, LibraryCreate
from app.storage.base_store import BaseStore
from app.storage.file_io_utils import atomic_write_json, load_json


class LibraryStore(BaseStore[Library]):
    """Thread-safe store for Library entities.

    Libraries have:
    - Auto-generated integer IDs
    - Globally unique names (for name-based lookups)
    """

    def __init__(self, data_dir: Path) -> None:
        super().__init__()
        self._data_path = data_dir / "libraries.json"
        self._name_index: dict[str, int] = {}  # name -> id

    def _get_id(self, item: Library) -> int:
        return item.id

    def _save(self) -> None:
        data = {
            "_meta": {"next_id": self._next_id},
            "items": {str(id_): lib.model_dump(mode="json") for id_, lib in self._data.items()}
        }
        atomic_write_json(self._data_path, data)

    async def load(self) -> None:
        async with self._lock.write():
            raw_data = load_json(self._data_path)
            self._next_id = raw_data.get("_meta", {}).get("next_id", 1)
            items = raw_data.get("items", {})
            self._data = {int(id_): Library(**lib_data) for id_, lib_data in items.items()}
            self._name_index = {lib.name: lib.id for lib in self._data.values()}

    async def create(self, library_create: LibraryCreate) -> Library:
        """Create a new library. Raises ValueError if name already exists."""
        async with self._lock.write():
            if library_create.name in self._name_index:
                raise ValueError(f"Library with name '{library_create.name}' already exists")

            new_id = self._allocate_id()
            library = Library(
                id=new_id,
                name=library_create.name,
                description=library_create.description,
                index_type=library_create.index_type,
                metric=library_create.metric,
            )

            self._data[new_id] = library
            self._name_index[library.name] = new_id
            self._save()
            return library

    async def delete(self, item_id: int) -> bool:
        """Delete a library and remove from name index."""
        async with self._lock.write():
            if item_id not in self._data:
                return False
            library = self._data[item_id]
            del self._name_index[library.name]
            del self._data[item_id]
            self._save()
            return True

    async def clear(self) -> None:
        """Clear all data and reset state."""
        async with self._lock.write():
            self._data.clear()
            self._name_index.clear()
            self._next_id = 1
