"""Abstract base class for entity stores.

Provides common CRUD pattern with RWLock protection and persistence.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

from pydantic import BaseModel

from app.storage.rwlock import AsyncRWLock

ModelT = TypeVar("ModelT", bound=BaseModel)


class BaseStore(ABC, Generic[ModelT]):
    """Base class for thread-safe entity stores.

    Provides:
    - RWLock protection (multiple readers OR single writer)
    - Common CRUD operations

    Subclasses must implement:
    - _get_id: Extract ID from model instance
    - _save: Persist data to storage
    - load: Load data from storage
    """

    def __init__(self, persist: bool = True) -> None:
        self._data: dict[str, ModelT] = {}
        self._lock = AsyncRWLock()
        self._persist = persist

    @abstractmethod
    def _get_id(self, item: ModelT) -> str:
        """Get the ID from an item."""
        pass

    @abstractmethod
    def _save(self) -> None:
        """Persist data to storage. Called after write operations."""
        pass

    @abstractmethod
    async def load(self) -> None:
        """Load data from storage into self._data."""
        pass

    async def create(self, item: ModelT) -> ModelT:
        """Create a new item. Raises ValueError if ID already exists."""
        item_id = self._get_id(item)
        async with self._lock.write():
            if item_id in self._data:
                raise ValueError(f"Item with id '{item_id}' already exists")
            self._data[item_id] = item
            if self._persist:
                self._save()
            return item

    async def get(self, item_id: str) -> Optional[ModelT]:
        """Get an item by ID. Returns None if not found."""
        async with self._lock.read():
            return self._data.get(item_id)

    async def list_all(self) -> list[ModelT]:
        """List all items."""
        async with self._lock.read():
            return list(self._data.values())

    async def delete(self, item_id: str) -> bool:
        """Delete an item. Returns True if deleted, False if not found."""
        async with self._lock.write():
            if item_id not in self._data:
                return False
            del self._data[item_id]
            if self._persist:
                self._save()
            return True

    async def exists(self, item_id: str) -> bool:
        """Check if an item exists."""
        async with self._lock.read():
            return item_id in self._data

    async def clear(self) -> None:
        """Clear all data."""
        async with self._lock.write():
            self._data.clear()
