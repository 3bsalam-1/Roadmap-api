from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logging import logger


# Global instances
_pc = None
_embeddings = None
_vector_store = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create the HuggingFace embeddings instance."""
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_pinecone_client() -> Pinecone:
    """Initialize Pinecone client and ensure index exists."""
    global _pc
    if _pc is None:
        logger.info(f"Initializing Pinecone client")
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Check if index exists, create if not
        existing = [i.name for i in _pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in existing:
            logger.info(f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}")
            _pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=384,  # all-MiniLM-L6-v2 output dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION,
                ),
            )
    return _pc


def get_vector_store() -> PineconeVectorStore:
    """Get or create the Pinecone vector store."""
    global _vector_store
    if _vector_store is None:
        embeddings = get_embeddings()
        pc = get_pinecone_client()
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        
        _vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
        )
        logger.info("PineconeVectorStore initialized")
    return _vector_store


def get_index() -> Pinecone:
    """Get the Pinecone index directly for low-level operations."""
    pc = get_pinecone_client()
    return pc.Index(settings.PINECONE_INDEX_NAME)


def get_document_count() -> int:
    """Get the total number of vectors in the index."""
    try:
        index = get_index()
        return index.describe_index_stats()["total_vector_count"]
    except Exception as e:
        logger.warning(f"Could not get document count: {e}")
        return 0


def reset():
    """Reset the vector store (for testing)."""
    global _vector_store, _pc, _embeddings
    _vector_store = None
    _pc = None
    _embeddings = None
