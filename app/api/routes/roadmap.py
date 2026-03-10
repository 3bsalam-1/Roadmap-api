import asyncio
from fastapi import APIRouter, HTTPException, Request
from app.core.limiter import limiter

from app.schemas.roadmap import (
    RoadmapRequest, 
    RoadmapResponse, 
    TracksResponse, 
    HealthResponse
)
from app.rag.detector import track_detector
from app.rag.chain import roadmap_chain
from app.rag.cache import generation_cache
from app.rag.retriever import get_document_count
from app.rag.prompt_parser import prompt_parser
from app.core.logging import logger


router = APIRouter()


@router.post("/roadmap/generate", response_model=RoadmapResponse)
@limiter.limit("10/minute")
async def generate_roadmap(request: Request, body: RoadmapRequest) -> RoadmapResponse:
    """
    Generate a learning roadmap for the specified track.
    Rate limited to 10 requests per minute per IP.
    
    Supports both simple track names and natural language prompts:
    - Simple: "python", "machine learning"
    - Natural language: "i want to learn python", "generate a roadmap for react for beginners"
    
    Decision flow:
    1. Parse the prompt (simple or via LLM) to extract track and level
    2. Run TrackDetector against Pinecone (PDF knowledge base)
       → Found: run RAG chain → return with source="knowledge_base"
    3. Not found: check GenerationCache.get(track, level)
       → Cache hit: return cached JSON with source="llm_generated" (no LLM call)
    4. Cache miss: run Generation chain (LLM)
       → Save result via GenerationCache.save(track, level, roadmap)
       → Return with source="llm_generated"
    """
    # Step 0: Parse natural language prompt to extract track and level
    parsed_track, parsed_level = await asyncio.to_thread(
        prompt_parser.parse,
        body.track
    )
    
    # Use parsed values, but request.level takes precedence if explicitly provided
    track = parsed_track.strip()
    level = body.level or parsed_level or "beginner"
    
    logger.info(f"Generating roadmap for: {track}, level: {level} (from prompt: {body.track})")
    
    try:
        # Step 1 & 2: Check Pinecone knowledge base
        track_found, relevant_docs = await asyncio.to_thread(
            track_detector.detect, 
            track
        )
        
        if track_found:
            # Knowledge base hit - use RAG chain
            logger.info(f"Track found in knowledge base: {track}")
            result = await asyncio.to_thread(
                roadmap_chain.generate_from_rag,
                track,
                relevant_docs
            )
            
            # Override level if specified in request
            if body.level:
                result["level"] = body.level
            
            return RoadmapResponse(**result)
        
        # Step 3: Check cache for generated roadmaps
        logger.info(f"Track not in knowledge base, checking cache: {track}")
        cached_roadmap = await asyncio.to_thread(
            generation_cache.get,
            track,
            level
        )
        
        if cached_roadmap:
            # Cache hit - return cached version
            logger.info(f"Cache hit for: {track} ({level})")
            # Override level if specified in request
            if body.level:
                cached_roadmap["level"] = body.level
            return RoadmapResponse(**cached_roadmap)
        
        # Step 4: Cache miss - generate with LLM
        logger.info(f"Cache miss, generating with LLM: {track}")
        result = await asyncio.to_thread(
            roadmap_chain.generate_from_llm,
            track,
            level
        )
        
        # Override level if specified in request
        if body.level:
            result["level"] = body.level
        
        # Save to cache for future requests
        await asyncio.to_thread(
            generation_cache.save,
            track,
            level,
            result
        )
        
        return RoadmapResponse(**result)
        
    except Exception as e:
        logger.error(f"Error generating roadmap: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to generate roadmap: {str(e)}")


@router.get("/roadmap/tracks", response_model=TracksResponse)
async def get_available_tracks() -> TracksResponse:
    """
    Get list of all available tracks split by source.
    
    - knowledge_base: tracks from Pinecone (PDF-based)
    - cached: tracks from JSON cache (LLM-generated)
    """
    try:
        # Get knowledge base tracks from Pinecone
        kb_tracks = await asyncio.to_thread(track_detector.get_available_tracks)
        
        # Get cached tracks from generation cache
        cached_tracks = await asyncio.to_thread(generation_cache.list_cached)
        
        return TracksResponse(
            knowledge_base=kb_tracks,
            cached=cached_tracks
        )
    except Exception as e:
        logger.error(f"Error getting tracks: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to get tracks: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint with comprehensive status:
    - Pinecone vector store status and document count
    - Cloudinary connectivity
    - Cache directory and roadmap count
    """
    from pathlib import Path
    import cloudinary
    import cloudinary.api
    from app.core.config import settings
    
    pinecone_status = "unknown"
    doc_count = 0
    
    try:
        doc_count = get_document_count()
        pinecone_status = "connected"
    except Exception as e:
        logger.error(f"Pinecone health check failed: {e}")
        pinecone_status = "error"
    
    # Check Cloudinary connectivity
    cloudinary_status = "unknown"
    try:
        cloudinary.api.ping()
        cloudinary_status = "connected"
    except Exception as e:
        logger.warning(f"Cloudinary health check failed: {e}")
        cloudinary_status = "error"
    
    # Check cache volume and count
    cache_count = 0
    cache_status = "unknown"
    try:
        cache_path = Path(settings.GENERATED_CACHE_DIR)
        if cache_path.exists():
            cache_count = len(list(cache_path.glob("*.json")))
            cache_status = "ok"
        else:
            cache_status = "not_found"
    except Exception as e:
        logger.warning(f"Cache health check failed: {e}")
        cache_status = "error"
    
    logger.info(f"Health check: Pinecone={pinecone_status}, Cloudinary={cloudinary_status}, Cache={cache_status} ({cache_count} cached)")
    
    all_ok = pinecone_status == "connected"
    
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        chromadb_status=pinecone_status,
        documents_indexed=doc_count
    )
