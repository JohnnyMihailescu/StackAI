# StackAI - Vector Database API

A vector database implemented from scratch using NumPy, featuring multiple indexing strategies and similarity search capabilities.

## Overview

StackAI is a FastAPI-based vector database that demonstrates core concepts of similarity search and vector indexing. Unlike production vector databases that rely on external libraries like FAISS, StackAI implements everything from scratch to provide a deep understanding of how vector databases work under the hood.

**Key Features:**
- **RESTful API** - FastAPI with automatic OpenAPI documentation
- **Multiple Index Types** - Flat (brute-force) and IVF (clustered) indexes
- **Embedding Generation** - Automatic vector embeddings via Cohere API
- **Similarity Search** - Similarity search using cosine similarity or euclidean distance, configurable per library.
- **Concurrent Access** - Thread-safe storage with async Reader-Writer locks
- **No External Database** - File-based persistence using JSON and NumPy

**Data Model:**
```
Library → Document → Chunk (with embedding vector)
```

Each library contains documents, each document contains text chunks, and each chunk automatically gets a 1024-dimensional embedding vector for similarity search.

## Quick Start (Local Development)

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Cohere API key (get one free at [cohere.com](https://cohere.com))

### Setup

1. **Clone and setup environment:**
```bash
# Clone the repository
git clone <your-repo-url>
cd Stackai

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate   # Linux/Mac/WSL
# OR
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
uv sync
```

2. **Configure environment:**
```bash
# Create .env file with your Cohere API key
echo "COHERE_API_KEY=your_key_here" > .env
```

3. **Start the server:**
```bash
# Using make
make start

# OR using uvicorn directly
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Exploring the API

**Option 1: Interactive API Documentation (Browser)**

Open your browser and visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

You can test all endpoints directly from the browser interface!

**Option 2: Load Sample Data**

Seed the database with pre-built test collections:

```bash
# Seed all sample libraries (recipes, support, products)
python scripts/seed_data.py --library all

# Or seed individual libraries
python scripts/seed_data.py --library recipes
python scripts/seed_data.py --library support
python scripts/seed_data.py --library products
```

After seeding, you can:
- Search for similar recipes: POST to `/api/v1/libraries/1/search` with query "chicken pasta"
- Browse libraries: GET `/api/v1/libraries`
- Explore documents: GET `/api/v1/libraries/1/documents`

**Option 3: Jupyter Notebooks**

Explore the system interactively with notebooks:

```bash
# Install jupyter if needed
uv pip install jupyter

# Start jupyter
jupyter notebook

# Open notebooks/
# - vector_database_exploration.ipynb: Basic concepts and usage
# - index_comparison.ipynb: Compare Flat vs IVF index performance
```

The notebooks demonstrate:
- Creating libraries, documents, and chunks
- Generating embeddings
- Performing similarity searches
- Comparing index performance (Flat vs IVF)

## Quick Start (Docker)

### Prerequisites
- Docker and Docker Compose
- Cohere API key

### Setup and Run

1. **Configure environment:**
```bash
# Create .env file with your Cohere API key
echo "COHERE_API_KEY=your_key_here" > .env
```

2. **Build and start the container:**
```bash
# Build and run in foreground
docker compose up --build

# OR run in background
docker compose up -d --build
```

The API will be available at `http://localhost:8000`

3. **Seed test data (after container is running):**
```bash
# Seed all sample libraries
docker compose exec api python scripts/seed_data.py --library all
```

4. **View logs:**
```bash
docker compose logs -f
```

5. **Stop the container:**
```bash
# Stop and keep data
docker compose down

# Stop and delete all data
docker compose down -v
```

**Data Persistence:**
Docker uses a named volume (`stackai-data`) to persist data between restarts. Your data survives container restarts unless you use `docker compose down -v`.

### Accessing the API

Once running (local or Docker), visit:
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## Running Tests

```bash
# Run all tests
make test

# OR using pytest directly
pytest tests/ -v

# Run specific test file
pytest tests/test_search.py -v

# Run with coverage
pytest tests/ -v --cov=app
```

**Test Coverage:**
- `test_flat_index.py` - Flat index implementation (319 tests)
- `test_ivf_index.py` - IVF index implementation (445 tests)
- `test_search.py` - Search functionality (293 tests)
- `test_persistence.py` - Data persistence (248 tests)
- `test_libraries.py` - Library endpoints (192 tests)
- `test_documents.py` - Document endpoints (321 tests)
- `test_chunks_batch.py` - Batch chunk operations (176 tests)

## Load Testing with Locust

StackAI includes comprehensive load tests using [Locust](https://locust.io/) to stress test different aspects of the system.

### Installing Locust

```bash
# Locust should be installed with uv sync, but if needed:
uv pip install locust
```

### Available Load Tests

**1. Search-Only Load Test** (`locust_search.py`)
- Stress tests the search endpoint
- Tests Cohere API integration, vector search, and data retrieval
- Requires seeded data

```bash
# Seed data first
python scripts/seed_data.py --library all

# Run search load test
locust -f loadtests/locust_search.py --host=http://localhost:8000
```

**2. Read-Heavy Load Test** (`locust_reads.py`)
- Tests read operations (list libraries, documents, chunks)
- Validates concurrent read performance under Reader-Writer locks

```bash
locust -f loadtests/locust_reads.py --host=http://localhost:8000
```

**3. Write-Heavy Load Test** (`locust_writes.py`)
- Tests write operations (create libraries, documents, chunks)
- Stresses embedding generation and index updates

```bash
locust -f loadtests/locust_writes.py --host=http://localhost:8000
```

**4. Mixed Workload** (`locust_mixed.py`)
- Simulates realistic production traffic
- Multiple user types with different behaviors:
  - **ReadHeavyUser** (weight=5): Mostly browsing and searching
  - **WriteUser** (weight=1): Adding content
- Tests Reader-Writer lock interactions

```bash
locust -f loadtests/locust_mixed.py --host=http://localhost:8000

# Run only readers
locust -f loadtests/locust_mixed.py --host=http://localhost:8000 ReadHeavyUser
```

### Running Load Tests

1. **Start the API server** (if not already running):
```bash
# Local
make start

# OR Docker
docker compose up -d
```

2. **Seed test data** (if needed):
```bash
# Local
python scripts/seed_data.py --library all

# Docker
docker compose exec api python scripts/seed_data.py --library all
```

3. **Launch Locust:**
```bash
locust -f loadtests/locust_<test>.py --host=http://localhost:8000
```

4. **Open Locust UI:**
- Navigate to http://localhost:8089
- Configure number of users and spawn rate
- Start swarming and monitor real-time statistics

### Interpreting Results

Locust provides:
- **Requests/sec** - Throughput of the API
- **Response times** - P50, P95, P99 percentiles
- **Failure rate** - Percentage of failed requests
- **Charts** - Real-time visualization of performance

**What to watch for:**
- Search operations should handle 10-50 req/s (limited by Cohere API)
- Read operations should handle 100+ req/s
- Write operations should handle 20-50 req/s (limited by embedding generation)
- Mixed workloads test Reader-Writer lock fairness

### Cleanup After Load Tests

```bash
# Stop the API and delete test data
docker compose down -v

# OR manually cleanup
python scripts/cleanup_loadtest.py  # If this script exists
```

## Project Structure

```
app/                    # FastAPI application
   main.py            # Application entry point
   routers/           # API endpoints
   services/          # Business logic (embeddings, search, indexes)
   storage/           # Data persistence with RWLock
   models/            # Pydantic models

loadtests/             # Locust load tests
notebooks/             # Jupyter notebooks for exploration
scripts/               # Utility scripts (seed data, etc.)
tests/                 # Pytest test suite
data/                  # Persisted data (gitignored)
```

## Index Types

**Flat Index:**
- Brute force cosine similarity search
- 100% recall accuracy
- O(n) search complexity
- Best for: <100k vectors, when accuracy is critical

**IVF Index:**
- Inverted File index with k-means clustering
- Approximate nearest neighbor search
- O(n_probe � n/n_clusters) complexity
- Configurable speed/accuracy tradeoff
- Best for: >100k vectors, when speed matters

Configure via environment variables:
- `IVF_DEFAULT_CLUSTERS=100` - Number of clusters
- `IVF_DEFAULT_N_PROBE=10` - Number of clusters to search

## Configuration

Key environment variables (create `.env` file):

```bash
# Required
COHERE_API_KEY=your_key_here

# Optional
COHERE_EMBED_MODEL=embed-english-v3.0
COHERE_BATCH_SIZE=96
MAX_BATCH_SIZE=500
IVF_DEFAULT_CLUSTERS=100
IVF_DEFAULT_N_PROBE=10
DEBUG=false
DATA_DIR=data
```

## API Endpoints

**Base URL:** `/api/v1`

- Libraries: `POST/GET/DELETE /libraries`
- Documents: `POST/GET/DELETE /libraries/{id}/documents`
- Chunks: `POST/GET/DELETE /documents/{id}/chunks` (with batch endpoint)
- Search: `POST /libraries/{id}/search`

See full API documentation at http://localhost:8000/docs

## Contributing

This is a learning project! Feel free to:
- Experiment with different indexing strategies
- Add new distance metrics
- Implement additional features
- Optimize performance

## Developer Documentation

For detailed technical documentation, architecture decisions, and developer guidelines, see [CLAUDE.md](CLAUDE.md).

## License

[Your License Here]
