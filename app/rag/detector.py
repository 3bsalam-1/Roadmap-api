from typing import Tuple, List
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import logger
from app.rag.retriever import get_vector_store, get_index
from app.rag.ingestor import normalize_track_name


class TrackDetector:
    """Detects whether a requested track exists in the knowledge base."""
    
    def __init__(self):
        self._vector_store = None
    
    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store
    
    def detect(self, track_query: str) -> Tuple[bool, List[Document]]:
        """
        Detect if the requested track exists in the knowledge base.
        
        Returns:
            Tuple of (track_found: bool, relevant_docs: list[Document])
            - Performs similarity search against Pinecone
            - If max similarity score > SIMILARITY_THRESHOLD: track found
            - Returns docs for use in RAG chain
        """
        # Normalize the track name
        normalized_query = normalize_track_name(track_query)
        logger.info(f"Detecting track: {normalized_query}")
        
        # Perform similarity search with scores
        try:
            results = self.vector_store.similarity_search_with_score(
                normalized_query, 
                k=settings.TOP_K_RESULTS
            )
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return False, []
        
        if not results:
            logger.info("No documents found for query")
            return False, []
        
        # Get the best score
        # Pinecone returns (document, score) where score is already similarity (0-1)
        best_doc, best_score = results[0]
        
        logger.info(f"Best similarity score: {best_score:.3f} (threshold: {settings.SIMILARITY_THRESHOLD})")
        
        if best_score >= settings.SIMILARITY_THRESHOLD:
            logger.info(f"Track detected in knowledge base: {normalized_query}")
            # Extract documents from results
            documents = [doc for doc, _ in results]
            return True, documents
        else:
            logger.info(f"Track NOT found in knowledge base: {normalized_query}")
            return False, []
    
    def get_available_tracks(self) -> List[str]:
        """Get list of all available tracks from the knowledge base."""
        try:
            index = get_index()
            
            # Query with a dummy vector to get all unique track values from metadata
            # Pinecone doesn't support direct metadata queries well, so we need a workaround
            # We'll use the describe_index_stats to get an idea, then query by known patterns
            stats = index.describe_index_stats()
            
            # Get unique track values by querying with empty filter
            # This is a limitation of Pinecone - we need to iterate or use a different approach
            # For now, we'll return empty and rely on the cache + explicit uploads
            # A better approach would be to maintain a separate metadata index
            
            # Return empty list - tracks will come from cache + explicit uploads
            return []
            
        except Exception as e:
            logger.error(f"Error getting available tracks: {e}")
            return []


# Global instance
track_detector = TrackDetector()
