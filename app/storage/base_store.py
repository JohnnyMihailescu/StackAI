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
    - Auto-incrementing integer IDs

    Subclasses must implement:
    - _get_id: Extract ID from model instance
    - _save: Persist data to storage
    - load: Load data from storage
    """

    def __init__(self) -> None:
        self._data: dict[int, ModelT] = {}
        self._next_id: int = 1
        self._lock = AsyncRWLock()

    @abstractmethod
    def _get_id(self, item: ModelT) -> int:
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

    def _allocate_id(self) -> int:
        """Allocate and return the next available ID."""
        new_id = self._next_id
        self._next_id += 1
        return new_id

    async def get(self, item_id: int) -> Optional[ModelT]:
        """Get an item by ID. Returns None if not found."""
        async with self._lock.read():
            return self._data.get(item_id)

    async def list_all(self) -> list[ModelT]:
        """List all items."""
        async with self._lock.read():
            return list(self._data.values())

    async def delete(self, item_id: int) -> bool:
        """Delete an item. Returns True if deleted, False if not found."""
        async with self._lock.write():
            if item_id not in self._data:
                return False
            del self._data[item_id]
            self._save()
            return True

    async def exists(self, item_id: int) -> bool:
        """Check if an item exists."""
        async with self._lock.read():
            return item_id in self._data

    async def clear(self) -> None:
        """Clear all data."""
        async with self._lock.write():
            self._data.clear()
            self._next_id = 1
