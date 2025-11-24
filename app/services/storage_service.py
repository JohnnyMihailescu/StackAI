"""Storage service - singleton for thread-safe data access."""

from pathlib import Path

from app.config import settings
from app.storage.chunk_store import ChunkStore
from app.storage.document_store import DocumentStore
from app.storage.flat_index_store import FlatIndexStore
from app.storage.ivf_index_store import IVFIndexStore
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
    _flat_index_store: FlatIndexStore | None = None
    _ivf_index_store: IVFIndexStore | None = None

    @classmethod
    async def initialize(cls, data_dir: str | None = None) -> None:
        """Initialize all stores and load data from disk."""
        path = Path(data_dir or settings.data_dir)
        path.mkdir(parents=True, exist_ok=True)

        cls._libraries = LibraryStore(path)
        cls._documents = DocumentStore(path)
        cls._chunks = ChunkStore(path)

        index_path = path / "indexes"
        cls._flat_index_store = FlatIndexStore(data_dir=index_path)
        cls._ivf_index_store = IVFIndexStore(data_dir=index_path)

        await cls._libraries.load()
        await cls._documents.load()
        await cls._chunks.load()

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
    def flat_index_store(cls) -> FlatIndexStore:
        """Get the flat index store."""
        if cls._flat_index_store is None:
            raise RuntimeError("StorageService not initialized")
        return cls._flat_index_store

    @classmethod
    def ivf_index_store(cls) -> IVFIndexStore:
        """Get the IVF index store."""
        if cls._ivf_index_store is None:
            raise RuntimeError("StorageService not initialized")
        return cls._ivf_index_store

    @classmethod
    def get_stats(cls) -> dict:
        """Get storage statistics."""
        flat_count = len(cls._flat_index_store.list_libraries()) if cls._flat_index_store else 0
        ivf_count = len(cls._ivf_index_store.list_libraries()) if cls._ivf_index_store else 0
        return {
            "libraries": len(cls._libraries._data) if cls._libraries else 0,
            "documents": len(cls._documents._data) if cls._documents else 0,
            "chunks": len(cls._chunks._data) if cls._chunks else 0,
            "flat_indexes": flat_count,
            "ivf_indexes": ivf_count,
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
        # Index stores don't have async clear - they're stateless
        # Clearing is done by deleting files, which happens when indexes are deleted
