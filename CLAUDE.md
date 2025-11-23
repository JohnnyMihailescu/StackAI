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
│   └── chunk.py               # Chunk model + batch request/response
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
│       └── flat_index.py      # Brute-force cosine similarity search
└── storage/
    ├── rwlock.py              # Async Reader-Writer lock
    ├── base_store.py          # Abstract base for entity stores
    ├── library_store.py       # Library CRUD with locking
    ├── document_store.py      # Document CRUD with locking
    ├── chunk_store.py         # Chunk CRUD + embeddings with locking
    └── file_io_utils.py       # Atomic file I/O utilities

data/                          # Persisted data (gitignored)
├── libraries.json             # Library metadata (JSON dict keyed by ID)
├── documents.json             # Document metadata (JSON dict keyed by ID)
├── chunks.json                # Chunk metadata only (no embeddings)
└── indexes/                   # Vector indexes, one .npz file per library
    └── {library_id}.npz       # NumPy arrays: vectors + chunk IDs

tests/
├── test_flat_index.py         # 33 tests for FlatIndex
└── test_chunks_batch.py       # 13 tests for batch endpoint

Dockerfile                     # Container build definition
docker-compose.yml             # Local development orchestration
.dockerignore                  # Files excluded from Docker build
```

## Domain Model

The core data hierarchy is: **Library → Document → Chunk**

**Library** - Collection of documents
- `id`, `name`, `description` (optional)
- `created_at`, `updated_at` timestamps

**Document** - Belongs to a library
- `id`, `library_id`, `name`, `source` (optional)
- `metadata` dict for custom fields (year, authors, etc.)

**Chunk** - Piece of text with embedding
- `id`, `document_id`, `text`
- `embedding` (1024-dim float array, auto-generated via Cohere)
- `metadata` dict for position, page, etc.

**Note:** Documents come pre-chunked. No automatic chunking logic.

## Vector Indexing

Vector indexes are implemented from scratch using NumPy:

**Flat Index** (`app/services/indexes/flat_index.py`)
- Brute force search using cosine similarity
- O(n) search complexity
- 100% recall accuracy
- Suitable for <100k vectors
- Use when accuracy is critical

**IVF Index** (planned)
- Inverted File index using k-means clustering
- Faster search by only checking relevant clusters
- Configurable speed/accuracy tradeoff
- Suitable for larger datasets

All indexes implement the `BaseIndex` interface:
- `add(vectors, ids)` - Add vectors to index
- `search(query_vector, k)` - Find k nearest neighbors
- `delete(ids)` - Remove vectors by ID
- `get_stats()` - Get index statistics

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
- **Persistence**: JSON for metadata, NumPy `.npz` for embeddings

### Async Requirements

All endpoint handlers and storage operations MUST be `async`:

```python
# CORRECT - async endpoint using await
@router.get("/libraries/{library_id}")
async def get_library(library_id: str):
    library = await StorageService.libraries().get(library_id)
    ...

# WRONG - sync endpoint (breaks concurrency)
@router.get("/libraries/{library_id}")
def get_library(library_id: str):
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

- **Delete from index:** Deleting a chunk doesn't remove it from the vector index (stale vectors may remain until restart)
