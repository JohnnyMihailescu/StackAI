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

    # Shared state - IDs created during the test
    library_ids: list[str] = []
    document_ids: list[tuple[str, str]] = []  # (doc_id, library_id)

    @task(3)
    def create_library(self):
        """Create a new library."""
        lib_id = random_id("loadtest_lib_")
        payload = {
            "id": lib_id,
            "name": f"Load Test Library {lib_id}",
            "description": "Created by Locust write test",
        }

        with self.client.post(
            "/api/v1/libraries",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                WriteUser.library_ids.append(lib_id)
                response.success()
            elif response.status_code == 409:
                response.success()  # Already exists, not a failure
            else:
                response.failure(f"Status {response.status_code}")

    @task(5)
    def create_document(self):
        """Create a document in an existing library."""
        if not WriteUser.library_ids:
            return

        lib_id = random.choice(WriteUser.library_ids)
        doc_id = random_id("loadtest_doc_")
        payload = {
            "id": doc_id,
            "library_id": lib_id,
            "name": f"Load Test Document {doc_id}",
            "metadata": {"source": "locust_writes"},
        }

        with self.client.post(
            f"/api/v1/libraries/{lib_id}/documents",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                WriteUser.document_ids.append((doc_id, lib_id))
                response.success()
            elif response.status_code == 409:
                response.success()
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
        chunk_id = random_id("loadtest_chunk_")
        payload = {
            "id": chunk_id,
            "document_id": doc_id,
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
            elif response.status_code in (404, 409):
                response.success()  # Document gone or chunk exists
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
                "id": random_id("loadtest_chunk_"),
                "document_id": doc_id,
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
            elif response.status_code in (404, 409):
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
