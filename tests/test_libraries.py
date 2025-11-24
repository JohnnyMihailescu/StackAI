"""Tests for library router endpoints."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.storage_service import StorageService


class TestLibraryEndpoints:
    """Test suite for library CRUD endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Initialize and clear storage before each test."""
        data_dir = str(tmp_path / "test_data")
        asyncio.run(StorageService.initialize(data_dir=data_dir))
        yield
        asyncio.run(StorageService.clear_all())

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    # -------------------------------------------------------------------------
    # POST /libraries - Create
    # -------------------------------------------------------------------------

    def test_create_library_minimal(self, client):
        """Test creating a library with just a name."""
        payload = {"name": "Test Library"}

        response = client.post("/api/v1/libraries", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Library"
        assert isinstance(data["id"], int)
        assert data["description"] is None
        assert data["index_type"] == "flat"  # Default
        assert data["metric"] == "cosine"  # Default

    def test_create_library_full(self, client):
        """Test creating a library with all fields."""
        payload = {
            "name": "Full Library",
            "description": "A test library",
            "index_type": "ivf",
            "metric": "euclidean",
        }

        response = client.post("/api/v1/libraries", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Full Library"
        assert data["description"] == "A test library"
        assert data["index_type"] == "ivf"
        assert data["metric"] == "euclidean"

    def test_create_library_duplicate_name(self, client):
        """Test that duplicate library names return 409 Conflict."""
        payload = {"name": "Duplicate Test"}

        # First creation should succeed
        response1 = client.post("/api/v1/libraries", json=payload)
        assert response1.status_code == 201

        # Second creation with same name should fail
        response2 = client.post("/api/v1/libraries", json=payload)
        assert response2.status_code == 409
        assert "already exists" in response2.json()["detail"]

    def test_create_library_missing_name(self, client):
        """Test that missing name returns validation error."""
        payload = {"description": "No name provided"}

        response = client.post("/api/v1/libraries", json=payload)

        assert response.status_code == 422  # Pydantic validation error

    def test_create_library_sequential_ids(self, client):
        """Test that library IDs are assigned sequentially."""
        lib1 = client.post("/api/v1/libraries", json={"name": "Lib 1"}).json()
        lib2 = client.post("/api/v1/libraries", json={"name": "Lib 2"}).json()
        lib3 = client.post("/api/v1/libraries", json={"name": "Lib 3"}).json()

        assert lib2["id"] == lib1["id"] + 1
        assert lib3["id"] == lib2["id"] + 1

    # -------------------------------------------------------------------------
    # GET /libraries/{id} - Read
    # -------------------------------------------------------------------------

    def test_get_library(self, client):
        """Test getting a library by ID."""
        # Create a library first
        create_resp = client.post(
            "/api/v1/libraries",
            json={"name": "Get Test", "description": "Testing get"}
        )
        lib_id = create_resp.json()["id"]

        # Get the library
        response = client.get(f"/api/v1/libraries/{lib_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == lib_id
        assert data["name"] == "Get Test"
        assert data["description"] == "Testing get"

    def test_get_library_not_found(self, client):
        """Test getting a non-existent library returns 404."""
        response = client.get("/api/v1/libraries/999")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    # -------------------------------------------------------------------------
    # GET /libraries - List
    # -------------------------------------------------------------------------

    def test_list_libraries_empty(self, client):
        """Test listing libraries when none exist."""
        response = client.get("/api/v1/libraries")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_libraries(self, client):
        """Test listing multiple libraries."""
        # Create several libraries
        client.post("/api/v1/libraries", json={"name": "Lib A"})
        client.post("/api/v1/libraries", json={"name": "Lib B"})
        client.post("/api/v1/libraries", json={"name": "Lib C"})

        response = client.get("/api/v1/libraries")

        assert response.status_code == 200
        libraries = response.json()
        assert len(libraries) == 3
        names = {lib["name"] for lib in libraries}
        assert names == {"Lib A", "Lib B", "Lib C"}

    # -------------------------------------------------------------------------
    # DELETE /libraries/{id} - Delete
    # -------------------------------------------------------------------------

    def test_delete_library(self, client):
        """Test deleting a library."""
        # Create a library
        create_resp = client.post("/api/v1/libraries", json={"name": "To Delete"})
        lib_id = create_resp.json()["id"]

        # Delete it
        response = client.delete(f"/api/v1/libraries/{lib_id}")
        assert response.status_code == 204

        # Verify it's gone
        get_resp = client.get(f"/api/v1/libraries/{lib_id}")
        assert get_resp.status_code == 404

    def test_delete_library_not_found(self, client):
        """Test deleting a non-existent library returns 404."""
        response = client.delete("/api/v1/libraries/999")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_library_cascades(self, client):
        """Test that deleting a library also deletes its documents."""
        # Create library
        lib_resp = client.post("/api/v1/libraries", json={"name": "Cascade Test"})
        lib_id = lib_resp.json()["id"]

        # Create document in the library
        doc_resp = client.post(
            f"/api/v1/libraries/{lib_id}/documents",
            json={"name": "Test Doc"}
        )
        doc_id = doc_resp.json()["id"]

        # Delete library
        client.delete(f"/api/v1/libraries/{lib_id}")

        # Verify document is also gone
        doc_get = client.get(f"/api/v1/libraries/{lib_id}/documents/{doc_id}")
        assert doc_get.status_code == 404
