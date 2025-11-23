"""
Search-focused load test for StackAI API.

Stress tests the search endpoint which involves:
- Cohere API call for query embedding
- Vector similarity search (FlatIndex)
- Chunk retrieval from storage

Requires existing data with embeddings. Seed data first:
    python scripts/seed_data.py --library all

Usage:
    locust -f loadtests/locust_search.py --host=http://localhost:8000
"""

import random
from locust import HttpUser, task, between, events

import sys
sys.path.insert(0, ".")
from loadtests.common import random_query


class SearchUser(HttpUser):
    """User that only performs search operations."""

    wait_time = between(0.5, 1.5)  # Faster to stress search

    # Libraries discovered at startup
    library_ids: list[str] = []

    def on_start(self):
        """Discover libraries with data to search."""
        self._discover_libraries()

    def _discover_libraries(self):
        """Find libraries that have documents (and likely chunks)."""
        if not SearchUser.library_ids:
            response = self.client.get("/api/v1/libraries")
            if response.status_code == 200:
                libraries = response.json()
                SearchUser.library_ids = [lib["id"] for lib in libraries]

        if not SearchUser.library_ids:
            print("WARNING: No libraries found. Search tests will skip.")

    @task(10)
    def search_small(self):
        """Search with k=5 (small result set)."""
        if not SearchUser.library_ids:
            return

        lib_id = random.choice(SearchUser.library_ids)
        payload = {
            "query": random_query(),
            "k": 5,
        }
        self.client.post(
            f"/api/v1/libraries/{lib_id}/search",
            json=payload,
            name="/api/v1/libraries/{id}/search (k=5)",
        )

    @task(5)
    def search_medium(self):
        """Search with k=20 (medium result set)."""
        if not SearchUser.library_ids:
            return

        lib_id = random.choice(SearchUser.library_ids)
        payload = {
            "query": random_query(),
            "k": 20,
        }
        self.client.post(
            f"/api/v1/libraries/{lib_id}/search",
            json=payload,
            name="/api/v1/libraries/{id}/search (k=20)",
        )

    @task(2)
    def search_large(self):
        """Search with k=50 (large result set)."""
        if not SearchUser.library_ids:
            return

        lib_id = random.choice(SearchUser.library_ids)
        payload = {
            "query": random_query(),
            "k": 50,
        }
        self.client.post(
            f"/api/v1/libraries/{lib_id}/search",
            json=payload,
            name="/api/v1/libraries/{id}/search (k=50)",
        )


# Print stats summary at end
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary focused on search performance."""
    print("\n" + "=" * 60)
    print("Search Load Test Summary")
    print("=" * 60)

    stats = environment.stats
    for entry in stats.entries.values():
        if "search" in entry.name:
            print(f"\n{entry.name}:")
            print(f"  Requests:    {entry.num_requests}")
            print(f"  Failures:    {entry.num_failures}")
            print(f"  Avg (ms):    {entry.avg_response_time:.0f}")
            print(f"  Min (ms):    {entry.min_response_time:.0f}")
            print(f"  Max (ms):    {entry.max_response_time:.0f}")
            print(f"  p50 (ms):    {entry.get_response_time_percentile(0.5):.0f}")
            print(f"  p95 (ms):    {entry.get_response_time_percentile(0.95):.0f}")
            print(f"  p99 (ms):    {entry.get_response_time_percentile(0.99):.0f}")

    print("=" * 60 + "\n")
