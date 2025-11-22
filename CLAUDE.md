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

## Project Structure

```
app/
├── main.py                    # FastAPI app with lifespan management
├── config.py                  # Settings via pydantic-settings
├── models/
│   ├── library.py             # Library model
│   ├── document.py            # Document model
│   └── chunk.py               # Chunk model + batch request/response
├── routers/
│   ├── libraries.py           # Library CRUD endpoints
│   ├── documents.py           # Document CRUD endpoints
│   └── chunks.py              # Chunk CRUD + batch endpoints
├── services/
│   ├── embeddings.py          # Cohere embedding service
│   └── indexes/
│       ├── base.py            # Abstract BaseIndex interface
│       └── flat_index.py      # Brute-force cosine similarity search
└── storage/                   # Placeholder for persistence layer

tests/
├── test_flat_index.py         # 33 tests for FlatIndex
└── test_chunks_batch.py       # 13 tests for batch endpoint
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

**Health:** `GET /` and `GET /health`

## Key Design Decisions

1. **In-memory storage first:** Using Python dictionaries for now. Persistence will be added later.

2. **Pre-chunked documents:** No automatic chunking. Users provide documents already split into chunks.

3. **Cosine similarity:** Using cosine similarity for vector comparisons (normalized dot product).

4. **From-scratch indexing:** Implementing vector indexes without external libraries for learning purposes.

5. **Cohere embeddings:** Using Cohere's API for generating embeddings rather than local models.

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
| `DEBUG` | `false` | Enable debug mode |
| `DATA_DIR` | `data` | Directory for persisted data |

## Current Limitations / TODOs

- **In-memory only:** All data stored in Python dicts, lost on restart
- **Index not wired up:** FlatIndex exists but chunks aren't added to it yet
- **No search endpoint:** Can't query vectors yet
- **Delete cascading:** Deleting a document doesn't delete its chunks
- **Delete from index:** Deleting a chunk doesn't remove it from the index
