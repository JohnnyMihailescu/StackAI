"""
Write-heavy load test for StackAI API.

Tests write contention and RWLock exclusive writer behavior.
Will create test data - run cleanup script afterward.

Usage:
    locust -f loadtests/locust_writes.py --host=http://localhost:8000

Cleanup:
    python scripts/cleanup_loadtest.py
"""

import random
from locust import HttpUser, task, between

import sys
sys.path.insert(0, ".")
from loadtests.common import random_id, random_text


class WriteUser(HttpUser):
    """User that performs write operations to test write contention."""

    wait_time = between(0.5, 2)

    # Shared state - IDs created during the test (server-generated)
    library_ids: list[int] = []
    document_ids: list[tuple[int, int]] = []  # (doc_id, library_id)

    @task(3)
    def create_library(self):
        """Create a new library with random index type (flat or ivf)."""
        unique_name = f"Load Test Library {random_id()}"
        index_type = random.choice(["flat", "ivf"])
        payload = {
            "name": unique_name,
            "description": "Created by Locust write test",
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
                WriteUser.library_ids.append(lib_id)
                response.success()
            elif response.status_code == 409:
                response.success()  # Name conflict, not a failure
            else:
                response.failure(f"Status {response.status_code}")

    @task(5)
    def create_document(self):
        """Create a document in an existing library."""
        if not WriteUser.library_ids:
            return

        lib_id = random.choice(WriteUser.library_ids)
        unique_name = f"Load Test Document {random_id()}"
        payload = {
            "name": unique_name,
            "metadata": {"source": "locust_writes"},
        }

        with self.client.post(
            f"/api/v1/libraries/{lib_id}/documents",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                doc_id = response.json()["id"]
                WriteUser.document_ids.append((doc_id, lib_id))
                response.success()
            elif response.status_code == 409:
                response.success()  # Name conflict within library
            elif response.status_code == 404:
                response.success()  # Library was deleted, not a failure
            else:
                response.failure(f"Status {response.status_code}")

    @task(8)
    def create_chunk(self):
        """Create a single chunk (triggers embedding generation)."""
        if not WriteUser.document_ids:
            return

        doc_id, lib_id = random.choice(WriteUser.document_ids)
        payload = {
            "text": random_text(),
            "metadata": {"position": random.randint(0, 100)},
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

    @task(2)
    def create_chunks_batch(self):
        """Create multiple chunks in a batch."""
        if not WriteUser.document_ids:
            return

        doc_id, lib_id = random.choice(WriteUser.document_ids)
        batch_size = random.choice([5, 10, 20])

        chunks = [
            {
                "text": random_text(),
                "metadata": {"position": i, "batch": True},
            }
            for i in range(batch_size)
        ]

        with self.client.post(
            f"/api/v1/documents/{doc_id}/chunks/batch",
            json={"chunks": chunks},
            name=f"/api/v1/documents/{{id}}/chunks/batch (n={batch_size})",
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                response.success()
            elif response.status_code == 404:
                response.success()  # Document was deleted, not a failure
            else:
                response.failure(f"Status {response.status_code}")
