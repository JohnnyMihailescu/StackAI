"""Tests for search endpoint."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.services.storage_service import StorageService


# Mock embedding dimension
MOCK_EMBEDDING_DIM = 1024


def mock_embed_texts(texts: list[str]) -> list[list[float]]:
    """Create fake embeddings for each text.

    We create slightly different embeddings so similarity calculations vary.
    """
    embeddings = []
    for i, text in enumerate(texts):
        # Create slightly different embeddings for each text
        embedding = [0.1 + (i * 0.01)] * MOCK_EMBEDDING_DIM
        embeddings.append(embedding)
    return embeddings


def mock_embed_query(text: str) -> list[float]:
    """Create fake embedding for query (similar to first chunk)."""
    return [0.1] * MOCK_EMBEDDING_DIM


class TestSearchEndpoint:
    """Test suite for POST /libraries/{library_id}/search endpoint."""

    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create temporary data directory for index persistence."""
        return tmp_path / "test_data"

    @pytest.fixture(autouse=True)
    def setup(self, data_dir):
        """Initialize storage with persistence (indexes require disk I/O)."""
        asyncio.run(StorageService.initialize(data_dir=data_dir))
        yield
        asyncio.run(StorageService.clear_all())

    @pytest.fixture
    def client(self):
        """Create test client with mocked embeddings."""
        with patch(
            "app.routers.chunks.EmbeddingService.embed_texts",
            side_effect=mock_embed_texts
        ), patch(
            "app.routers.search.EmbeddingService.embed_query",
            side_effect=mock_embed_query
        ):
            yield TestClient(app)

    @pytest.fixture
    def library_with_chunks(self, client):
        """Create a library with documents and chunks for testing.

        Returns (library, document, chunks)
        """
        # Create library
        lib = client.post("/api/v1/libraries", json={"name": "Search Test Lib"}).json()

        # Create document
        doc = client.post(
            f"/api/v1/libraries/{lib['id']}/documents",
            json={"name": "Test Doc"}
        ).json()

        # Create chunks
        chunks_data = [
            {"text": "Machine learning is a subset of AI"},
            {"text": "Python is a programming language"},
            {"text": "Neural networks mimic brain structure"},
            {"text": "Deep learning uses multiple layers"},
            {"text": "Natural language processing handles text"},
        ]

        chunk_resp = client.post(
            f"/api/v1/documents/{doc['id']}/chunks/batch",
            json={"chunks": chunks_data}
        ).json()

        return lib, doc, chunk_resp["chunks"]

    # -------------------------------------------------------------------------
    # Basic Search Tests
    # -------------------------------------------------------------------------

    def test_search_basic(self, client, library_with_chunks):
        """Test basic search returns results."""
        lib, doc, chunks = library_with_chunks

        payload = {"query": "What is machine learning?", "k": 3}
        response = client.post(f"/api/v1/libraries/{lib['id']}/search", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is machine learning?"
        assert data["result_count"] == 3
        assert len(data["results"]) == 3

        # Verify result structure
        for result in data["results"]:
            assert "chunk" in result
            assert "score" in result
            assert isinstance(result["score"], float)
            assert "id" in result["chunk"]
            assert "text" in result["chunk"]

    def test_search_library_not_found(self, client):
        """Test search on non-existent library returns 404."""
        payload = {"query": "test", "k": 3}
        response = client.post("/api/v1/libraries/999/search", json=payload)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_search_empty_library(self, client):
        """Test search on library with no chunks returns empty results."""
        # Create empty library
        lib = client.post("/api/v1/libraries", json={"name": "Empty Lib"}).json()

        payload = {"query": "test", "k": 3}
        response = client.post(f"/api/v1/libraries/{lib['id']}/search", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["result_count"] == 0
        assert data["results"] == []

    # -------------------------------------------------------------------------
    # K Parameter Tests
    # -------------------------------------------------------------------------

    def test_search_k_larger_than_corpus(self, client, library_with_chunks):
        """Test requesting more results than available chunks."""
        lib, _, chunks = library_with_chunks

        # We have 5 chunks, request 100
        payload = {"query": "test", "k": 100}
        response = client.post(f"/api/v1/libraries/{lib['id']}/search", json=payload)

        assert response.status_code == 200
        data = response.json()
        # Should return all 5 chunks
        assert data["result_count"] == 5
        assert len(data["results"]) == 5

    def test_search_k_one(self, client, library_with_chunks):
        """Test searching for just top-1 result."""
        lib, _, _ = library_with_chunks

        payload = {"query": "test", "k": 1}
        response = client.post(f"/api/v1/libraries/{lib['id']}/search", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["result_count"] == 1
        assert len(data["results"]) == 1

    # -------------------------------------------------------------------------
    # Embedding Inclusion Tests
    # -------------------------------------------------------------------------

    def test_search_without_embeddings(self, client, library_with_chunks):
        """Test that embeddings are excluded by default."""
        lib, _, _ = library_with_chunks

        payload = {"query": "test", "k": 2}
        response = client.post(f"/api/v1/libraries/{lib['id']}/search", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Embeddings should be None or not present
        for result in data["results"]:
            assert result["chunk"]["embedding"] is None

    def test_search_with_embeddings(self, client, library_with_chunks):
        """Test including embeddings in results."""
        lib, _, _ = library_with_chunks

        payload = {"query": "test", "k": 2}
        # Use query parameter to request embeddings
        response = client.post(
            f"/api/v1/libraries/{lib['id']}/search",
            json=payload,
            params={"include_embedding": "true"}
        )

        assert response.status_code == 200
        data = response.json()

        # Embeddings should be present
        for result in data["results"]:
            assert result["chunk"]["embedding"] is not None
            assert isinstance(result["chunk"]["embedding"], list)
            assert len(result["chunk"]["embedding"]) == MOCK_EMBEDDING_DIM

    # -------------------------------------------------------------------------
    # Result Ordering Tests
    # -------------------------------------------------------------------------

    def test_search_results_ordered_by_score(self, client, library_with_chunks):
        """Test that results are ordered by similarity score (descending)."""
        lib, _, _ = library_with_chunks

        payload = {"query": "test", "k": 5}
        response = client.post(f"/api/v1/libraries/{lib['id']}/search", json=payload)

        assert response.status_code == 200
        data = response.json()

        scores = [result["score"] for result in data["results"]]
        # Scores should be in descending order
        assert scores == sorted(scores, reverse=True)

    def test_search_scores_are_valid(self, client, library_with_chunks):
        """Test that similarity scores are in valid range."""
        lib, _, _ = library_with_chunks

        payload = {"query": "test", "k": 5}
        response = client.post(f"/api/v1/libraries/{lib['id']}/search", json=payload)

        assert response.status_code == 200
        data = response.json()

        # For cosine similarity, scores should be between -1 and 1
        for result in data["results"]:
            assert -1.0 <= result["score"] <= 1.0

    # -------------------------------------------------------------------------
    # Multiple Libraries Tests
    # -------------------------------------------------------------------------

    def test_search_only_searches_specified_library(self, client):
        """Test that search only returns results from the specified library."""
        # Create two libraries with different content
        lib1 = client.post("/api/v1/libraries", json={"name": "Lib 1"}).json()
        lib2 = client.post("/api/v1/libraries", json={"name": "Lib 2"}).json()

        doc1 = client.post(
            f"/api/v1/libraries/{lib1['id']}/documents",
            json={"name": "Doc 1"}
        ).json()
        doc2 = client.post(
            f"/api/v1/libraries/{lib2['id']}/documents",
            json={"name": "Doc 2"}
        ).json()

        # Add chunks to lib1
        client.post(
            f"/api/v1/documents/{doc1['id']}/chunks",
            json={"text": "Content in library one"}
        )

        # Add chunks to lib2
        chunk2 = client.post(
            f"/api/v1/documents/{doc2['id']}/chunks",
            json={"text": "Content in library two"}
        ).json()

        # Search lib1
        resp1 = client.post(
            f"/api/v1/libraries/{lib1['id']}/search",
            json={"query": "content", "k": 10}
        )

        # Should only get results from lib1
        results1 = resp1.json()["results"]
        assert len(results1) == 1
        assert chunk2["id"] not in [r["chunk"]["id"] for r in results1]

    # -------------------------------------------------------------------------
    # Validation Tests
    # -------------------------------------------------------------------------

    def test_search_missing_query(self, client, library_with_chunks):
        """Test that missing query returns validation error."""
        lib, _, _ = library_with_chunks

        payload = {"k": 3}  # Missing query
        response = client.post(f"/api/v1/libraries/{lib['id']}/search", json=payload)

        assert response.status_code == 422  # Pydantic validation error
