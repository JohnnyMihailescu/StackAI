"""StackAI API Client for use in notebooks and scripts."""

import httpx
from typing import Any


class StackAIClient:
    """Client for interacting with the StackAI vector database API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/api/v1",
        timeout: float = 30.0,
    ):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check if the server is running. Returns True if healthy."""
        try:
            response = httpx.get(self.base_url.replace("/api/v1", "/health"))
            return response.status_code == 200
        except httpx.ConnectError:
            return False

    def print_health(self):
        """Print server health status."""
        if self.health_check():
            print("Server is running")
        else:
            print("Cannot connect to server. Start it with: make start")

    # -------------------------------------------------------------------------
    # Libraries
    # -------------------------------------------------------------------------

    def create_library(
        self,
        name: str,
        description: str | None = None,
        index_type: str = "flat",
        metric: str = "cosine",
    ) -> dict[str, Any]:
        """Create a new library. Returns the created library with server-generated ID.

        Args:
            name: Library name (must be unique)
            description: Optional description
            index_type: Index type - "flat" (brute force, 100% recall) or "ivf" (faster, approximate)
            metric: Distance metric - "cosine" or "euclidean"
        """
        data = {"name": name, "index_type": index_type, "metric": metric}
        if description:
            data["description"] = description
        response = self.client.post("/libraries", json=data)
        response.raise_for_status()
        return response.json()

    def get_library(self, library_id: int) -> dict[str, Any]:
        """Get a library by ID."""
        response = self.client.get(f"/libraries/{library_id}")
        response.raise_for_status()
        return response.json()

    def get_libraries(self) -> list[dict[str, Any]]:
        """List all libraries."""
        response = self.client.get("/libraries")
        response.raise_for_status()
        return response.json()

    def delete_library(self, library_id: int) -> None:
        """Delete a library and all its documents/chunks."""
        response = self.client.delete(f"/libraries/{library_id}")
        response.raise_for_status()

    def print_libraries(self):
        """Print all libraries."""
        libraries = self.get_libraries()
        print(f"Libraries ({len(libraries)}):")
        for lib in libraries:
            print(f"  - [{lib['id']}] {lib['name']}")

    # -------------------------------------------------------------------------
    # Documents
    # -------------------------------------------------------------------------

    def create_document(
        self,
        library_id: int,
        name: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new document in a library. Returns the created document with server-generated ID."""
        data = {"name": name}
        if source:
            data["source"] = source
        if metadata:
            data["metadata"] = metadata
        response = self.client.post(f"/libraries/{library_id}/documents", json=data)
        response.raise_for_status()
        return response.json()

    def get_document(self, library_id: int, document_id: int) -> dict[str, Any]:
        """Get a document by ID."""
        response = self.client.get(f"/libraries/{library_id}/documents/{document_id}")
        response.raise_for_status()
        return response.json()

    def get_documents(self, library_id: int) -> list[dict[str, Any]]:
        """List all documents in a library."""
        response = self.client.get(f"/libraries/{library_id}/documents")
        response.raise_for_status()
        return response.json()

    def delete_document(self, library_id: int, document_id: int) -> None:
        """Delete a document and all its chunks."""
        response = self.client.delete(
            f"/libraries/{library_id}/documents/{document_id}"
        )
        response.raise_for_status()

    def print_documents(self, library_id: int):
        """Print all documents in a library."""
        docs = self.get_documents(library_id)
        print(f"Documents in library {library_id} ({len(docs)}):")
        for doc in docs:
            print(f"  - [{doc['id']}] {doc['name']}")

    # -------------------------------------------------------------------------
    # Chunks
    # -------------------------------------------------------------------------

    def create_chunk(
        self,
        document_id: int,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a single chunk (will auto-generate embedding). Returns chunk with server-generated ID."""
        data = {"text": text}
        if metadata:
            data["metadata"] = metadata
        response = self.client.post(f"/documents/{document_id}/chunks", json=data)
        response.raise_for_status()
        return response.json()

    def create_chunks_batch(
        self, document_id: int, chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Create multiple chunks in a batch (up to 500).

        Each chunk dict should have 'text' and optionally 'metadata'.
        IDs are assigned by the server.
        """
        response = self.client.post(
            f"/documents/{document_id}/chunks/batch", json={"chunks": chunks}
        )
        response.raise_for_status()
        return response.json()

    def get_chunk(
        self, chunk_id: int, include_embedding: bool = False
    ) -> dict[str, Any]:
        """Get a chunk by ID."""
        params = {"include_embedding": "true"} if include_embedding else {}
        response = self.client.get(f"/chunks/{chunk_id}", params=params)
        response.raise_for_status()
        return response.json()

    def get_chunks(
        self, document_id: int, include_embedding: bool = False
    ) -> list[dict[str, Any]]:
        """List all chunks in a document."""
        params = {"include_embedding": "true"} if include_embedding else {}
        response = self.client.get(f"/documents/{document_id}/chunks", params=params)
        response.raise_for_status()
        return response.json()

    def delete_chunk(self, chunk_id: int) -> None:
        """Delete a chunk."""
        response = self.client.delete(f"/chunks/{chunk_id}")
        response.raise_for_status()

    def print_chunks(
        self, document_id: int, show_text: bool = True, max_length: int | None = None
    ):
        """Print all chunks in a document.

        Args:
            document_id: The document ID to list chunks for.
            show_text: Whether to show chunk text (default: True).
            max_length: Max characters to show per chunk. None = full text (default).
        """
        chunks = self.get_chunks(document_id)
        print(f"Chunks in document {document_id} ({len(chunks)}):")
        for chunk in chunks:
            if show_text:
                text = chunk["text"]
                if max_length and len(text) > max_length:
                    text = f"{text[:max_length]}..."
                print(f"  [{chunk['id']}] {text}")
            else:
                print(f"  - {chunk['id']}")

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search(
        self,
        library_id: int,
        query: str,
        k: int = 3,
        include_embedding: bool = False,
    ) -> dict[str, Any]:
        """Search a library for similar chunks."""
        params = {"include_embedding": "true"} if include_embedding else {}
        response = self.client.post(
            f"/libraries/{library_id}/search",
            json={"query": query, "k": k},
            params=params,
        )
        response.raise_for_status()
        return response.json()

    def print_search(self, query: str, library_id: int = 1, k: int = 3):
        """Search a library and display formatted results."""
        try:
            data = self.search(library_id, query, k)
        except httpx.HTTPStatusError as e:
            print(f"Error: {e.response.status_code} - {e.response.text}")
            return

        print(f'Query: "{data["query"]}"')
        print(f"Library: {library_id}")
        print(f"Results: {data['result_count']}")
        print("-" * 60)

        for i, result in enumerate(data["results"], 1):
            chunk = result["chunk"]
            print(f"\n[{i}] Score: {result['score']:.4f}")
            print(f"    Doc: {chunk['document_id']}")
            print(f"    Text: {chunk['text']}")


# Default client instance for simple usage
_default_client: StackAIClient | None = None


def get_client() -> StackAIClient:
    """Get or create the default client instance."""
    global _default_client
    if _default_client is None:
        _default_client = StackAIClient()
    return _default_client


__all__ = ["StackAIClient", "get_client"]
