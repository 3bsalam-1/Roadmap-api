# Learning Roadmap API

A production-ready FastAPI service that generates structured learning roadmaps using a RAG (Retrieval-Augmented Generation) pipeline. The system uses pre-collected PDF roadmaps as a knowledge base. If the requested track exists in the knowledge base, the LLM answers grounded in the PDF. If not, the LLM generates a roadmap from its own knowledge in the same structured format.

## Features

- **Natural Language Prompts** - Send prompts like "generate a python roadmap for beginner" or just "python"
- **RAG Pipeline** - Retrieves relevant content from PDF knowledge base using Pinecone vector store with HuggingFace embeddings
- **Smart Fallback** - Generates roadmaps from LLM knowledge when PDF is not available
- **JSON Caching** - Generated roadmaps are cached to avoid redundant LLM calls
- **Track Detection** - Automatically detects available tracks in the knowledge base
- **Admin API** - Upload PDFs directly to the knowledge base via API
- **Rate Limiting** - Protected with 10 requests per minute per IP

## Tech Stack

- **Framework**: FastAPI
- **LLM**: GitHub Models — `openai/gpt-4.1` via Azure AI Inference SDK (direct, not langchain)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace (local, free)
- **Vector Store**: Pinecone (cloud-hosted vector database)
- **PDF Storage**: Cloudinary
- **PDF Parsing**: `pypdf` + `langchain_community.document_loaders.PyPDFLoader`
- **Rate Limiting**: slowapi

## Project Structure

```
roadmap-api/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── roadmap.py       # Roadmap endpoints (generate, tracks, health)
│   │       └── admin.py          # Admin endpoints (upload PDFs, list PDFs)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Settings via pydantic-settings
│   │   ├── logging.py           # Logging setup
│   │   └── security.py          # API key authentication for admin routes
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingestor.py          # PDF ingestion & chunking to Pinecone
│   │   ├── retriever.py         # Pinecone vector store & retrieval
│   │   ├── chain.py             # RAG chain (direct Azure AI Inference)
│   │   ├── detector.py          # Track detection logic
│   │   ├── prompt_parser.py     # Natural language prompt parsing
│   │   └── cache.py             # JSON roadmap caching
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── roadmap.py           # Pydantic request/response models
│   └── utils/
│       ├── __init__.py
│       ├── cloudinary_utils.py  # Cloudinary PDF upload & listing
│       └── pdf_utils.py         # PDF helpers
├── data/
│   ├── pdfs/                    # Place your roadmap PDFs here
│   └── generated/               # Cached generated roadmaps (JSON)
├── chroma_db/                   # (Legacy, Pinecone is now used)
├── scripts/
│   └── ingest_pdfs.py           # One-time script to ingest all PDFs
├── tests/
│   ├── __init__.py
│   ├── test_roadmap.py
│   └── test_rag.py
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── render.yaml                  # Render.com deployment configuration
└── README.md
```

## Quick Start

### 1. Clone and Setup

```bash
cd roadmap-api
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` with all required credentials:

```bash
# GitHub Models (LLM)
GITHUB_TOKEN=your_github_token_here

# Pinecone (Vector Store)
PINECONE_API_KEY=your_pinecone_api_key_here

# Cloudinary (PDF Storage)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Security (Admin API Key)
ADMIN_API_KEY=your_strong_random_secret_key_here
```

Get your GitHub token from: https://github.com/settings/tokens
Get your Pinecone API key from: https://app.pinecone.io/
Get your Cloudinary credentials from: https://cloudinary.com/

### 3. Add Roadmap PDFs

Place your roadmap PDFs in `data/pdfs/`. Use snake_case naming:
- `machine_learning.pdf`
- `web_development.pdf`
- `devops.pdf`

### 4. Run with Docker

```bash
docker-compose up --build
```

The API will be available at http://localhost:8000

### 5. Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest PDFs
python scripts/ingest_pdfs.py

# Run the server
uvicorn app.main:app --reload
```

## API Endpoints

### Generate Roadmap (Natural Language)

```bash
# Natural language prompts
curl -X POST http://localhost:8000/api/v1/roadmap/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "i want to learn python for beginner"}'

curl -X POST http://localhost:8000/api/v1/roadmap/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "generate a roadmap to learn react for advanced developers"}'
```

### Generate Roadmap (Structured)

```bash
# Traditional JSON format (still supported)
curl -X POST http://localhost:8000/api/v1/roadmap/generate \
  -H "Content-Type: application/json" \
  -d '{"track": "machine learning", "level": "beginner"}'

# Just track (level defaults to beginner)
curl -X POST http://localhost:8000/api/v1/roadmap/generate \
  -H "Content-Type: application/json" \
  -d '{"track": "python"}'
```

### List Available Tracks

```bash
curl http://localhost:8000/api/v1/roadmap/tracks
```

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Admin: Upload PDF

```bash
# Upload a PDF to the knowledge base (requires API key)
curl -X POST http://localhost:8000/api/v1/admin/pdfs/upload \
  -H "X-API-Key: your_admin_api_key" \
  -F "file=@data/pdfs/python.pdf" \
  -F "track_name=python"
```

### Admin: List PDFs

```bash
# List all PDFs stored in Cloudinary (requires API key)
curl -X GET http://localhost:8000/api/v1/admin/pdfs \
  -H "X-API-Key: your_admin_api_key"
```

### API Documentation

Visit http://localhost:8000/docs for interactive Swagger UI.

## How It Works

1. **Natural Language Parsing**: When a user sends a prompt (e.g., "i want to learn python for beginner"), the system uses an LLM to extract the track and level from the natural language.

2. **PDF Ingestion**: PDFs are loaded, split into chunks, embedded using HuggingFace embeddings, and stored in Pinecone.

3. **Track Detection**: When a user requests a roadmap, the system checks if the track exists in the knowledge base using similarity search in Pinecone.

4. **RAG Generation**: If the track is found in the knowledge base, the LLM generates a roadmap grounded in the PDF content.

5. **LLM Generation**: If the track is not found, the LLM generates a roadmap from its own knowledge in the same structured format.

6. **Caching**: Generated roadmaps are saved as JSON files in `data/generated/` (or `/data/generated` in production) to avoid redundant LLM calls for the same requests.

## Supported Levels

- `beginner`
- `intermediate`
- `advanced`

## Deployment

### Docker (Local)

```bash
docker-compose up --build
```

### Render.com

The project includes `render.yaml` for easy deployment to Render.com:

```bash
# Using Render CLI
render blueprint render.yaml
```

Or connect your GitHub repository to Render and it will automatically detect the `render.yaml` configuration.

Required environment variables on Render:
- `GITHUB_TOKEN`
- `PINECONE_API_KEY`
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`
- `ADMIN_API_KEY`
- `GENERATED_CACHE_DIR=/data/generated`

The Render deployment includes a 1GB disk mount at `/data` for storing generated roadmaps.

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_rag.py

# Run with coverage
pytest --cov=app tests/
```

## License

MIT
