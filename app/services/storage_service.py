"""Storage service - singleton for thread-safe data access."""

from pathlib import Path

from app.config import settings
from app.storage.chunk_store import ChunkStore
from app.storage.document_store import DocumentStore
from app.storage.index_store import IndexStore
from app.storage.library_store import LibraryStore


class StorageService:
    """Singleton service for thread-safe storage access.

    Provides access to all data stores (libraries, documents, chunks, indexes)
    with RWLock protection and disk persistence.

    Initialize at app startup via `await StorageService.initialize()`.
    """

    _libraries: LibraryStore | None = None
    _documents: DocumentStore | None = None
    _chunks: ChunkStore | None = None
    _indexes: IndexStore | None = None

    @classmethod
    async def initialize(cls, persist: bool = True, data_dir: str | None = None) -> None:
        """Initialize all stores and load data from disk."""
        path = Path(data_dir or settings.data_dir)
        path.mkdir(parents=True, exist_ok=True)

        cls._libraries = LibraryStore(path, persist=persist)
        cls._documents = DocumentStore(path, persist=persist)
        cls._chunks = ChunkStore(path, persist=persist)
        cls._indexes = IndexStore(data_dir=path / "indexes", persist=persist)

        await cls._libraries.load()
        await cls._documents.load()
        await cls._chunks.load()
        await cls._indexes.load()

    @classmethod
    def libraries(cls) -> LibraryStore:
        """Get the library store."""
        if cls._libraries is None:
            raise RuntimeError("StorageService not initialized")
        return cls._libraries

    @classmethod
    def documents(cls) -> DocumentStore:
        """Get the document store."""
        if cls._documents is None:
            raise RuntimeError("StorageService not initialized")
        return cls._documents

    @classmethod
    def chunks(cls) -> ChunkStore:
        """Get the chunk store."""
        if cls._chunks is None:
            raise RuntimeError("StorageService not initialized")
        return cls._chunks

    @classmethod
    def indexes(cls) -> IndexStore:
        """Get the index store."""
        if cls._indexes is None:
            raise RuntimeError("StorageService not initialized")
        return cls._indexes

    @classmethod
    def get_stats(cls) -> dict:
        """Get storage statistics."""
        return {
            "libraries": len(cls._libraries._data) if cls._libraries else 0,
            "documents": len(cls._documents._data) if cls._documents else 0,
            "chunks": len(cls._chunks._data) if cls._chunks else 0,
            "indexes": len(cls._indexes._indexes) if cls._indexes else 0,
        }

    @classmethod
    async def clear_all(cls) -> None:
        """Clear all stores. Useful for testing."""
        if cls._libraries:
            await cls._libraries.clear()
        if cls._documents:
            await cls._documents.clear()
        if cls._chunks:
            await cls._chunks.clear()
        if cls._indexes:
            await cls._indexes.clear()
