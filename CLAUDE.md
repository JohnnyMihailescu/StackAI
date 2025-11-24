# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working with Claude Code

**IMPORTANT:** Always propose a plan and get user approval BEFORE making code changes. The user expects to review high-level plans for each step before implementation.

## Project Overview

StackAI is a vector database API built with FastAPI. This is a learning project that implements vector indexing and similarity search from scratch (without external vector database libraries like FAISS). Embeddings are generated using Cohere's API.

**Technology Stack:**
- Python 3.12+
- FastAPI for API framework
- NumPy for vector operations
- Cohere for embeddings (embed-english-v3.0, 1024 dimensions)
- uv for dependency management
- pytest for testing

## Development Setup

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate   # Linux/Mac/WSL
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
uv sync

# Run development server
make start  # or: uvicorn app.main:app --reload

# Run tests
make test   # or: pytest tests/ -v
```

**API Documentation:** http://localhost:8000/docs (Swagger) or http://localhost:8000/redoc

## Docker Setup

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d --build

# Seed test data (after container is running)
docker compose exec api python scripts/seed_data.py --library all

# View logs
docker compose logs -f

# Stop
docker compose down

# Stop and delete data volume
docker compose down -v
```

**Docker files:**
- `Dockerfile` - Multi-stage build with Python 3.12-slim and uv
- `docker-compose.yml` - Local development setup with volume persistence
- `.dockerignore` - Excludes `.venv`, tests, notebooks, `data/`, etc.

**Data persistence:** Uses a named Docker volume (`stackai-data`). Data survives container restarts but is deleted with `docker compose down -v`.

**Environment variables:** Pass `COHERE_API_KEY` via environment or `.env` file (not copied into image).

## Project Structure

```
app/
├── main.py                    # FastAPI app with lifespan management
├── config.py                  # Settings via pydantic-settings
├── logging_config.py          # Logging setup
├── models/
│   ├── library.py             # Library model
│   ├── document.py            # Document model
│   ├── chunk.py               # Chunk model + batch request/response
│   ├── search.py              # Search request/response models
│   └── enums.py               # IndexType, DistanceMetric enums
├── routers/
│   ├── libraries.py           # Library CRUD endpoints
│   ├── documents.py           # Document CRUD endpoints
│   ├── chunks.py              # Chunk CRUD + batch endpoints
│   └── search.py              # Similarity search endpoint
├── services/
│   ├── embeddings.py          # Cohere embedding service
│   ├── storage_service.py     # Singleton for storage access
│   ├── search_service.py      # Vector search orchestration
│   └── indexes/
│       ├── base.py            # Abstract BaseIndex interface
│       ├── flat_index.py      # Brute-force cosine similarity search
│       ├── ivf_index.py       # IVF index with k-means clustering
│       ├── sequential_kmeans.py # Sequential k-means for clustering
│       └── utils.py           # Shared index utilities
└── storage/
    ├── rwlock.py              # Async Reader-Writer lock
    ├── base_store.py          # Abstract base for entity stores
    ├── library_store.py       # Library CRUD with locking
    ├── document_store.py      # Document CRUD with locking
    ├── chunk_store.py         # Chunk CRUD + embeddings with locking
    ├── flat_index_store.py    # Flat index persistence
    ├── ivf_index_store.py     # IVF index persistence
    └── file_io_utils.py       # Atomic file I/O utilities

data/                          # Persisted data (gitignored)
├── libraries.json             # Library metadata (JSON dict keyed by ID)
├── documents.json             # Document metadata (JSON dict keyed by ID)
├── chunks.json                # Chunk metadata only (no embeddings)
└── indexes/                   # Vector indexes
    ├── {library_id}.npz       # Flat index: NumPy arrays (vectors + chunk IDs)
    └── {library_id}/          # IVF index: directory per library
        ├── index_meta.json    # Index configuration
        ├── centroids.npy      # Cluster centroids (k × d)
        └── cluster_*.npy      # Vectors for each cluster

tests/
├── test_flat_index.py         # 319 tests for FlatIndex
├── test_ivf_index.py          # 445 tests for IVFIndex
├── test_chunks_batch.py       # 176 tests for batch endpoint
├── test_libraries.py          # 192 tests for library endpoints
├── test_documents.py          # 321 tests for document endpoints
├── test_search.py             # 293 tests for search functionality
└── test_persistence.py        # 248 tests for data persistence

notebooks/
├── vector_database_exploration.ipynb  # Initial exploration
└── index_comparison.ipynb             # Flat vs IVF performance comparison

Dockerfile                     # Container build definition
docker-compose.yml             # Local development orchestration
.dockerignore                  # Files excluded from Docker build
```

## Domain Model

The core data hierarchy is: **Library → Document → Chunk**

**Library** - Collection of documents
- `id` (int, server-generated), `name` (unique), `description` (optional)
- `created_at`, `updated_at` timestamps

**Document** - Belongs to a library
- `id` (int, server-generated), `library_id`, `name` (unique within library), `source` (optional)
- `metadata` dict for custom fields (year, authors, etc.)

**Chunk** - Piece of text with embedding
- `id` (int, server-generated), `document_id`, `text`
- `embedding` (1024-dim float array, auto-generated via Cohere)
- `metadata` dict for position, page, etc.

**ID System:**
- All IDs are **server-generated integers** (auto-increment)
- Create requests only need data (name, text), not IDs
- Lookups use integer IDs in URL paths: `/libraries/1`, `/documents/2`
- Names are for display/uniqueness constraints, not lookups

**Note:** Documents come pre-chunked. No automatic chunking logic.

## Vector Indexing

Vector indexes are implemented from scratch using NumPy:

**Flat Index** (`app/services/indexes/flat_index.py`)
- Brute force search using cosine similarity
- O(n) search complexity
- 100% recall accuracy
- Suitable for <100k vectors
- Use when accuracy is critical

**IVF Index** (`app/services/indexes/ivf_index.py`)
- Inverted File index using k-means clustering
- Partitions vectors into clusters, only searches relevant clusters at query time
- Sequential k-means implementation that adapts as vectors are added
- Lazy loading: cluster vectors loaded on-demand during search (saves memory)
- Configurable speed/accuracy tradeoff via `n_clusters` and `n_probe` settings
- O(n_probe × n/n_clusters) search complexity
- Suitable for larger datasets (>100k vectors)
- Use when speed is more important than perfect recall

All indexes implement the `BaseIndex` interface:
- `add(vectors, ids)` - Add vectors to index
- `search(query_vector, k)` - Find k nearest neighbors
- `delete(ids)` - Remove vectors by ID
- `get_vector(id)` - Retrieve a vector by ID
- `get_stats()` - Get index statistics
- `save_to_store(store)` - Persist index to disk
- `load_from_store(store, library_id)` - Load index from disk

## API Endpoints

**Base URL:** `/api/v1`

| Resource | Method | Endpoint | Description |
|----------|--------|----------|-------------|
| Libraries | POST | `/libraries` | Create library |
| | GET | `/libraries` | List all libraries |
| | GET | `/libraries/{library_id}` | Get library |
| | DELETE | `/libraries/{library_id}` | Delete library |
| Documents | POST | `/libraries/{library_id}/documents` | Create document |
| | GET | `/libraries/{library_id}/documents` | List documents in library |
| | GET | `/libraries/{library_id}/documents/{document_id}` | Get document |
| | DELETE | `/libraries/{library_id}/documents/{document_id}` | Delete document |
| Chunks | POST | `/documents/{document_id}/chunks` | Create single chunk (auto-embeds) |
| | POST | `/documents/{document_id}/chunks/batch` | Create multiple chunks (up to 500) |
| | GET | `/documents/{document_id}/chunks` | List chunks in document |
| | GET | `/chunks/{chunk_id}` | Get chunk by ID |
| | DELETE | `/chunks/{chunk_id}` | Delete chunk |
| Search | POST | `/libraries/{library_id}/search` | Similarity search |

**Query Parameters:**
- `include_embedding=true` - Include embedding vectors in chunk responses (GET chunks, search results). Embeddings are excluded by default to reduce response size.

**Health:** `GET /` and `GET /health`

## Key Design Decisions

1. **Thread-safe storage with RWLock:** All data access goes through `StorageService` which uses async Reader-Writer locks for concurrency control.

2. **Pre-chunked documents:** No automatic chunking. Users provide documents already split into chunks.

3. **Cosine similarity:** Using cosine similarity for vector comparisons (normalized dot product).

4. **From-scratch indexing:** Implementing vector indexes without external libraries for learning purposes.

5. **Cohere embeddings:** Using Cohere's API for generating embeddings rather than local models.

6. **Persistence without database:** JSON for metadata, NumPy binary for embeddings. No external database dependencies.

## Storage & Concurrency

**IMPORTANT:** All code MUST follow async patterns. This is critical for thread-safety.

### Architecture

```
Routers → StorageService → Individual Stores (with RWLock) → Persistence
```

- **StorageService** (`app/services/storage_service.py`): Singleton providing access to all stores
- **Stores** (`app/storage/`): Each entity type has its own store with RWLock protection
- **Persistence**: JSON for metadata, NumPy `.npz`/`.npy` for embeddings and index data

### Memory Model

**Metadata (libraries, documents, chunks):**
- Loaded entirely into memory at startup from JSON files
- All CRUD operations work on in-memory dicts (`self._data`)
- Changes are persisted to disk on each write operation
- **Implication:** Memory usage scales with number of entities and chunk text size

**Embeddings (vectors):**
- **Flat indexes:** Lazy loaded from `.npz` files on each operation (search, add, delete)
- **IVF indexes:** Metadata loaded at startup, cluster vectors loaded on-demand during search
- NOT cached in memory between operations
- Each search loads vectors fresh from disk
- **Implication:** Low memory usage for embeddings, but disk I/O on every search

This design prioritizes memory efficiency for embeddings (which are large: 1024 floats × 4 bytes = 4KB per chunk) over query latency. IVF indexes further optimize by only loading the cluster vectors being searched. For high-throughput production use, consider adding an LRU cache for frequently-accessed indexes.

### Async Requirements

All endpoint handlers and storage operations MUST be `async`:

```python
# CORRECT - async endpoint using await
@router.get("/libraries/{library_id}")
async def get_library(library_id: int):
    library = await StorageService.libraries().get(library_id)
    ...

# WRONG - sync endpoint (breaks concurrency)
@router.get("/libraries/{library_id}")
def get_library(library_id: int):
    ...
```

### Reader-Writer Lock Pattern

The `AsyncRWLock` allows multiple concurrent readers OR a single exclusive writer:

```python
# Multiple readers can run simultaneously
async with self._lock.read():
    return self._data.get(item_id)

# Writers get exclusive access
async with self._lock.write():
    self._data[item_id] = item
    self._save()
```

**Why this matters:**
- 20 users searching simultaneously → all proceed concurrently (read locks)
- 1 user writing while 19 read → writer waits for readers, then gets exclusive access
- Prevents race conditions and data corruption

### Adding New Entities

To add a new entity type:

1. Create model in `app/models/`
2. Create store extending `BaseStore[YourModel]` in `app/storage/`
3. Implement `_get_id()`, `_save()`, and `load()` methods
4. Add to `StorageService` as a new store
5. Create router with `async def` endpoints that `await` storage calls

### Testing

In tests, initialize storage with `persist=False`:

```python
@pytest.fixture(autouse=True)
def setup(self):
    asyncio.run(StorageService.initialize(persist=False))
    yield
    asyncio.run(StorageService.clear_all())
```

## Embedding Service

The `EmbeddingService` (`app/services/embeddings.py`) wraps Cohere's API:

- **Model:** `embed-english-v3.0` (1024 dimensions)
- **Batch size:** 96 texts per API call (Cohere's limit)
- `embed_texts(texts)` - For document storage (`input_type="search_document"`)
- `embed_query(text)` - For search queries (`input_type="search_query"`)

Initialized at app startup via FastAPI's lifespan context manager.

## Configuration

Settings loaded from `.env` via pydantic-settings (`app/config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `COHERE_API_KEY` | (required) | Cohere API key |
| `COHERE_EMBED_MODEL` | `embed-english-v3.0` | Embedding model |
| `COHERE_BATCH_SIZE` | `96` | Max texts per API call |
| `MAX_BATCH_SIZE` | `500` | Max chunks per batch request |
| `IVF_DEFAULT_CLUSTERS` | `100` | Number of clusters for IVF index |
| `IVF_DEFAULT_N_PROBE` | `10` | Number of clusters to search at query time |
| `DEBUG` | `false` | Enable debug mode (sets log level to DEBUG) |
| `DATA_DIR` | `data` | Directory for persisted data |

## Logging

Logging is configured in `app/logging_config.py` and uses Python's standard `logging` module.

**Setup:** Each module uses `logger = logging.getLogger(__name__)` to get a logger that inherits from the `app` namespace.

**Log level:** INFO by default, DEBUG when `DEBUG=true` in `.env`.

**What's logged:**
- Startup/shutdown and storage stats
- Library, document, chunk create/delete operations (with names, not IDs)
- Search queries with result counts
- Embedding batch operations
- Index loading and vector additions (DEBUG level)

**Adding logging to new modules:**
```python
import logging

logger = logging.getLogger(__name__)

# Then use:
logger.info("Operation description")
logger.debug("Detailed info for debugging")
```

## Current Limitations / TODOs

- **In-memory metadata:** All chunk text and metadata is loaded into memory at startup. Not suitable for millions of chunks without a proper database.
- **No index caching:** Embeddings are loaded from disk on every search. For high-throughput use cases, consider adding an LRU cache for indexes.
