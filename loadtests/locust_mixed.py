"""
Mixed workload load test for StackAI API.

Simulates realistic production traffic with multiple user types:
- ReadHeavyUser: Mostly browsing and searching (weight=5)
- WriteUser: Adding content (weight=1)

This tests how reads and writes interact under the RWLock.

Usage:
    locust -f loadtests/locust_mixed.py --host=http://localhost:8000

Run only readers:
    locust -f loadtests/locust_mixed.py --host=http://localhost:8000 ReadHeavyUser

Cleanup:
    python scripts/cleanup_loadtest.py
"""

import random
from locust import HttpUser, task, between

import sys
sys.path.insert(0, ".")
from loadtests.common import random_id, random_text, random_query


class SharedState:
    """Shared state across all user types."""

    library_ids: list[int] = []
    document_ids: list[tuple[int, int]] = []  # (doc_id, library_id)
    initialized: bool = False


def discover_data(client) -> None:
    """Discover existing data. Called once by first user."""
    if SharedState.initialized:
        return

    response = client.get("/api/v1/libraries")
    if response.status_code == 200:
        libraries = response.json()
        SharedState.library_ids = [lib["id"] for lib in libraries]

        for lib_id in SharedState.library_ids[:5]:
            resp = client.get(f"/api/v1/libraries/{lib_id}/documents")
            if resp.status_code == 200:
                for doc in resp.json():
                    SharedState.document_ids.append((doc["id"], lib_id))

    SharedState.initialized = True


class ReadHeavyUser(HttpUser):
    """
    User who mostly reads and searches.
    Represents typical API consumers querying for information.
    """

    weight = 5  # 5x more likely to spawn than WriteUser
    wait_time = between(0.5, 2)

    def on_start(self):
        discover_data(self.client)

    @task(3)
    def list_libraries(self):
        """Browse available libraries."""
        self.client.get("/api/v1/libraries")

    @task(2)
    def get_library(self):
        """View a specific library."""
        if SharedState.library_ids:
            lib_id = random.choice(SharedState.library_ids)
            self.client.get(f"/api/v1/libraries/{lib_id}")

    @task(3)
    def list_documents(self):
        """Browse documents in a library."""
        if SharedState.library_ids:
            lib_id = random.choice(SharedState.library_ids)
            self.client.get(f"/api/v1/libraries/{lib_id}/documents")

    @task(10)
    def search(self):
        """Search for information (main use case)."""
        if SharedState.library_ids:
            lib_id = random.choice(SharedState.library_ids)
            payload = {
                "query": random_query(),
                "k": random.choice([5, 10, 20]),
            }
            self.client.post(
                f"/api/v1/libraries/{lib_id}/search",
                json=payload,
                name="/api/v1/libraries/{id}/search",
            )


class WriteUser(HttpUser):
    """
    User who adds new content.
    Represents data ingestion pipelines or content creators.
    """

    weight = 1  # Less common than readers
    wait_time = between(1, 3)

    def on_start(self):
        discover_data(self.client)

    @task(2)
    def create_library(self):
        """Create a new library with random index type (flat or ivf)."""
        unique_name = f"Load Test Library {random_id()}"
        index_type = random.choice(["flat", "ivf"])
        payload = {
            "name": unique_name,
            "index_type": index_type,
        }

        with self.client.post(
            "/api/v1/libraries",
            json=payload,
            catch_response=True,
            name=f"/api/v1/libraries (index={index_type})",
        ) as response:
            if response.status_code == 201:
                lib_id = response.json()["id"]
                SharedState.library_ids.append(lib_id)
                response.success()
            elif response.status_code == 409:
                response.success()  # Name conflict, not a failure
            else:
                response.failure(f"Status {response.status_code}")

    @task(3)
    def create_document(self):
        """Create a document."""
        if not SharedState.library_ids:
            return

        lib_id = random.choice(SharedState.library_ids)
        unique_name = f"Document {random_id()}"
        payload = {
            "name": unique_name,
        }

        with self.client.post(
            f"/api/v1/libraries/{lib_id}/documents",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                doc_id = response.json()["id"]
                SharedState.document_ids.append((doc_id, lib_id))
                response.success()
            elif response.status_code in (404, 409):
                response.success()  # Library deleted or name conflict
            else:
                response.failure(f"Status {response.status_code}")

    @task(5)
    def create_chunk(self):
        """Create a chunk with embedding."""
        if not SharedState.document_ids:
            return

        doc_id, _ = random.choice(SharedState.document_ids)
        payload = {
            "text": random_text(),
        }

        with self.client.post(
            f"/api/v1/documents/{doc_id}/chunks",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                response.success()
            elif response.status_code == 404:
                response.success()  # Document was deleted, not a failure
            else:
                response.failure(f"Status {response.status_code}")
