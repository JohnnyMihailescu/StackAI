"""Tests for data persistence to disk."""

import asyncio
import json
from pathlib import Path

import pytest
from unittest.mock import patch

from app.services.storage_service import StorageService
from app.models.library import LibraryCreate
from app.models.document import DocumentCreate
from app.models.chunk import ChunkCreate


# Mock embedding for chunk creation
MOCK_EMBEDDING_DIM = 1024


def mock_embed_texts(texts: list[str]) -> list[list[float]]:
    """Create fake embeddings."""
    return [[0.1] * MOCK_EMBEDDING_DIM for _ in texts]


class TestPersistence:
    """Test suite for verifying data persists to disk correctly."""

    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create a temporary data directory that auto-cleans."""
        return tmp_path / "test_data"

    def test_library_persists_to_disk(self, data_dir):
        """Test that libraries survive restart."""
        # Initialize with persistence
        asyncio.run(StorageService.initialize(data_dir=data_dir))

        # Create a library
        lib_create = LibraryCreate(name="Persistent Library", description="Test")
        lib = asyncio.run(StorageService.libraries().create(lib_create))
        lib_id = lib.id

        # Verify file was created
        lib_file = data_dir / "libraries.json"
        assert lib_file.exists()

        # Clear in-memory data (simulate restart)
        asyncio.run(StorageService.clear_all())

        # Re-initialize and load from disk
        asyncio.run(StorageService.initialize(data_dir=data_dir))

        # Verify data was loaded
        loaded_lib = asyncio.run(StorageService.libraries().get(lib_id))
        assert loaded_lib is not None
        assert loaded_lib.name == "Persistent Library"
        assert loaded_lib.description == "Test"

    def test_document_persists_to_disk(self, data_dir):
        """Test that documents survive restart."""
        asyncio.run(StorageService.initialize(data_dir=data_dir))

        # Create library and document
        lib = asyncio.run(StorageService.libraries().create(LibraryCreate(name="Lib")))
        doc_create = DocumentCreate(name="Persistent Doc", source="test.pdf")
        doc = asyncio.run(StorageService.documents().create(lib.id, doc_create))
        doc_id = doc.id

        # Verify file was created
        doc_file = data_dir / "documents.json"
        assert doc_file.exists()

        # Restart
        asyncio.run(StorageService.clear_all())
        asyncio.run(StorageService.initialize(data_dir=data_dir))

        # Verify data was loaded
        loaded_doc = asyncio.run(StorageService.documents().get(doc_id))
        assert loaded_doc is not None
        assert loaded_doc.name == "Persistent Doc"
        assert loaded_doc.source == "test.pdf"
        assert loaded_doc.library_id == lib.id

    def test_chunk_metadata_persists(self, data_dir):
        """Test that chunk metadata persists (embeddings are in index files)."""
        asyncio.run(StorageService.initialize(data_dir=data_dir))

        # Create library, document, and chunk
        lib = asyncio.run(StorageService.libraries().create(LibraryCreate(name="Lib")))
        doc = asyncio.run(StorageService.documents().create(
            lib.id, DocumentCreate(name="Doc")
        ))

        chunk_create = ChunkCreate(
            text="Test chunk text",
            metadata={"page": 1, "position": 0}
        )
        chunk = asyncio.run(StorageService.chunks().create(doc.id, chunk_create))
        chunk_id = chunk.id

        # Verify file was created
        chunk_file = data_dir / "chunks.json"
        assert chunk_file.exists()

        # Verify embeddings are NOT in chunks.json
        with open(chunk_file) as f:
            data = json.load(f)
            # Check that embedding is excluded from storage
            chunk_data = data["items"][str(chunk_id)]
            assert "embedding" not in chunk_data or chunk_data["embedding"] is None

        # Restart
        asyncio.run(StorageService.clear_all())
        asyncio.run(StorageService.initialize(data_dir=data_dir))

        # Verify chunk metadata was loaded (without embedding)
        loaded_chunk = asyncio.run(StorageService.chunks().get(chunk_id))
        assert loaded_chunk is not None
        assert loaded_chunk.text == "Test chunk text"
        assert loaded_chunk.metadata["page"] == 1
        assert loaded_chunk.embedding is None  # Not stored in chunks.json

    def test_index_persists_to_disk(self, data_dir):
        """Test that vector indexes persist to disk."""
        with patch(
            "app.services.embeddings.EmbeddingService.embed_texts",
            side_effect=mock_embed_texts
        ):
            asyncio.run(StorageService.initialize(data_dir=data_dir))

            # Create library, document, and chunks with embeddings
            lib = asyncio.run(StorageService.libraries().create(
                LibraryCreate(name="Lib")
            ))
            doc = asyncio.run(StorageService.documents().create(
                lib.id, DocumentCreate(name="Doc")
            ))

            # Create chunks (this also adds to index)
            from app.services.embeddings import EmbeddingService
            from app.services.search_service import SearchService
            import numpy as np

            chunk1 = asyncio.run(StorageService.chunks().create(
                doc.id, ChunkCreate(text="First chunk")
            ))
            chunk2 = asyncio.run(StorageService.chunks().create(
                doc.id, ChunkCreate(text="Second chunk")
            ))

            # Add vectors to index
            embeddings = EmbeddingService.embed_texts(["First chunk", "Second chunk"])
            vectors = np.array(embeddings)
            asyncio.run(SearchService.add_vectors(
                lib.id, vectors, [chunk1.id, chunk2.id],
                index_type=lib.index_type, metric=lib.metric
            ))

            # Verify index file was created
            index_dir = data_dir / "indexes"
            assert index_dir.exists()

            # For flat index, check for metadata and vectors files
            flat_index_dir = index_dir / str(lib.id)
            assert flat_index_dir.exists()
            assert (flat_index_dir / "index_meta.json").exists()
            assert (flat_index_dir / "vectors.npy").exists()

    def test_id_counters_persist(self, data_dir):
        """Test that ID counters persist across restarts."""
        asyncio.run(StorageService.initialize(data_dir=data_dir))

        # Create a library with ID 1
        lib1 = asyncio.run(StorageService.libraries().create(
            LibraryCreate(name="Lib 1")
        ))
        assert lib1.id == 1

        # Restart
        asyncio.run(StorageService.clear_all())
        asyncio.run(StorageService.initialize(data_dir=data_dir))

        # Create another library - should get ID 2, not 1
        lib2 = asyncio.run(StorageService.libraries().create(
            LibraryCreate(name="Lib 2")
        ))
        assert lib2.id == 2

    def test_json_format_is_correct(self, data_dir):
        """Test that persisted JSON has correct format."""
        asyncio.run(StorageService.initialize(data_dir=data_dir))

        # Create a library
        lib = asyncio.run(StorageService.libraries().create(
            LibraryCreate(
                name="Format Test",
                description="Testing JSON format",
                index_type="ivf",
                metric="euclidean"
            )
        ))

        # Read and verify JSON structure
        lib_file = data_dir / "libraries.json"
        with open(lib_file) as f:
            data = json.load(f)

        # Check structure
        assert "_meta" in data
        assert "next_id" in data["_meta"]
        assert "items" in data
        assert str(lib.id) in data["items"]

        # Check library data
        lib_data = data["items"][str(lib.id)]
        assert lib_data["name"] == "Format Test"
        assert lib_data["description"] == "Testing JSON format"
        assert lib_data["index_type"] == "ivf"
        assert lib_data["metric"] == "euclidean"
        assert "created_at" in lib_data
        assert "updated_at" in lib_data

    def test_multiple_restarts(self, data_dir):
        """Test data survives multiple restart cycles."""
        # First session - create data
        asyncio.run(StorageService.initialize(data_dir=data_dir))
        lib1 = asyncio.run(StorageService.libraries().create(
            LibraryCreate(name="Session 1")
        ))
        lib1_id = lib1.id
        asyncio.run(StorageService.clear_all())

        # Second session - add more data
        asyncio.run(StorageService.initialize(data_dir=data_dir))
        lib2 = asyncio.run(StorageService.libraries().create(
            LibraryCreate(name="Session 2")
        ))
        lib2_id = lib2.id
        asyncio.run(StorageService.clear_all())

        # Third session - verify both exist
        asyncio.run(StorageService.initialize(data_dir=data_dir))
        loaded_lib1 = asyncio.run(StorageService.libraries().get(lib1_id))
        loaded_lib2 = asyncio.run(StorageService.libraries().get(lib2_id))

        assert loaded_lib1.name == "Session 1"
        assert loaded_lib2.name == "Session 2"
        asyncio.run(StorageService.clear_all())
