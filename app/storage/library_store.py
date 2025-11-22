"""Library store with thread-safe CRUD operations."""

from pathlib import Path

from app.models.library import Library
from app.storage.base_store import BaseStore
from app.storage.file_io_utils import atomic_write_json, load_json


class LibraryStore(BaseStore[Library]):
    """Thread-safe store for Library entities."""

    def __init__(self, data_dir: Path, persist: bool = True) -> None:
        super().__init__(persist=persist)
        self._data_path = data_dir / "libraries.json"

    def _get_id(self, item: Library) -> str:
        return item.id

    def _save(self) -> None:
        data = {id_: lib.model_dump(mode="json") for id_, lib in self._data.items()}
        atomic_write_json(self._data_path, data)

    async def load(self) -> None:
        async with self._lock.write():
            raw_data = load_json(self._data_path)
            self._data = {id_: Library(**lib_data) for id_, lib_data in raw_data.items()}
