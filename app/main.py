from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import time

from app.core.config import settings
from app.core.logging import logger
from app.core.limiter import limiter
from app.api.routes import roadmap, admin
from app.rag.retriever import get_vector_store


# Rate limiter is now imported from app.core.limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Roadmap API...")
    
    # Initialize rate limiter state
    app.state.limiter = limiter
    
    # Don't initialize Pinecone at startup - defer to first request
    # This avoids hanging during deployment when services might not be ready
    logger.info("Roadmap API ready (Pinecone will initialize on first request)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Roadmap API...")


app = FastAPI(
    title="Learning Roadmap API",
    description="AI-powered service that generates structured learning roadmaps using RAG pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s"
    )
    
    return response


# Include routers
app.include_router(roadmap.router, prefix=settings.API_PREFIX)
app.include_router(admin.router, prefix=settings.API_PREFIX)


# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "Learning Roadmap API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
