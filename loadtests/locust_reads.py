"""
Read-only load test for StackAI API.

Tests read performance and concurrent read scaling.
Good for establishing baseline performance.

Usage:
    locust -f loadtests/locust_reads.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between


class ReadUser(HttpUser):
    """User that only performs read operations."""

    wait_time = between(0.5, 2)

    # Cache library/document IDs discovered during test
    library_ids: list[str] = []
    document_ids: list[tuple[str, str]] = []  # (doc_id, library_id)

    def on_start(self):
        """Discover existing data when user starts."""
        self._discover_data()

    def _discover_data(self):
        """Fetch existing libraries and documents to read from."""
        if not ReadUser.library_ids:
            response = self.client.get("/api/v1/libraries")
            if response.status_code == 200:
                libraries = response.json()
                ReadUser.library_ids = [lib["id"] for lib in libraries]

                # Get documents for each library
                for lib_id in ReadUser.library_ids[:5]:  # Limit to first 5
                    resp = self.client.get(f"/api/v1/libraries/{lib_id}/documents")
                    if resp.status_code == 200:
                        for doc in resp.json():
                            ReadUser.document_ids.append((doc["id"], lib_id))

    @task(1)
    def health_check(self):
        """Check API health."""
        self.client.get("/health")

    @task(5)
    def list_libraries(self):
        """List all libraries."""
        self.client.get("/api/v1/libraries")

    @task(3)
    def get_library(self):
        """Get a specific library by ID."""
        if ReadUser.library_ids:
            import random
            lib_id = random.choice(ReadUser.library_ids)
            self.client.get(f"/api/v1/libraries/{lib_id}")

    @task(4)
    def list_documents(self):
        """List documents in a library."""
        if ReadUser.library_ids:
            import random
            lib_id = random.choice(ReadUser.library_ids)
            self.client.get(f"/api/v1/libraries/{lib_id}/documents")

    @task(2)
    def list_chunks(self):
        """List chunks in a document."""
        if ReadUser.document_ids:
            import random
            doc_id, _ = random.choice(ReadUser.document_ids)
            self.client.get(f"/api/v1/documents/{doc_id}/chunks")
