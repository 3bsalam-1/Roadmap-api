"""Tests for RAG components - detector, retriever, and chain."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from app.rag.detector import TrackDetector, track_detector
from app.rag.ingestor import normalize_track_name, extract_track_from_filename


class TestTrackDetector:
    """Tests for the TrackDetector class."""
    
    @patch('app.rag.detector.vector_store_manager')
    def test_detect_known_track(self, mock_manager):
        """Test detection of a known track in the knowledge base."""
        # Setup mock
        mock_vs = MagicMock()
        mock_manager.vector_store = mock_vs
        
        mock_doc = Document(
            page_content="Test content about machine learning",
            metadata={"source": "machine_learning.pdf", "track": "machine learning", "page": 1}
        )
        
        # Returns (document, distance) tuples - lower distance = more similar
        mock_vs.similarity_search_with_score.return_value = [
            (mock_doc, 0.2)  # Low distance = high similarity
        ]
        
        detector = TrackDetector()
        detector.vector_store = mock_vs
        
        # Execute
        track_found, docs = detector.detect("machine learning")
        
        # Assert
        assert track_found is True
        assert len(docs) > 0
    
    @patch('app.rag.detector.vector_store_manager')
    def test_detect_unknown_track(self, mock_manager):
        """Test detection of an unknown track (not in knowledge base)."""
        # Setup mock
        mock_vs = MagicMock()
        mock_manager.vector_store = mock_vs
        
        # Returns high distance = low similarity (below threshold)
        mock_vs.similarity_search_with_score.return_value = [
            (Document(page_content="Random content", metadata={}), 1.8)
        ]
        
        detector = TrackDetector()
        detector.vector_store = mock_vs
        
        # Execute
        track_found, docs = detector.detect("quantum computing")  # Unknown track
        
        # Assert
        assert track_found is False
        assert len(docs) == 0
    
    @patch('app.rag.detector.vector_store_manager')
    def test_get_available_tracks(self, mock_manager):
        """Test retrieving available tracks from knowledge base."""
        # Setup mock
        mock_collection = MagicMock()
        mock_manager.get_collection.return_value = mock_collection
        
        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "machine_learning.pdf", "track": "machine learning"},
                {"source": "web_development.pdf", "track": "web development"},
                {"source": "machine_learning.pdf", "track": "machine learning"},  # Duplicate
            ]
        }
        
        detector = TrackDetector()
        
        # Execute
        tracks = detector.get_available_tracks()
        
        # Assert
        assert "machine learning" in tracks
        assert "web development" in tracks
        assert len(tracks) == 2  # No duplicates


class TestTrackNormalization:
    """Tests for track name normalization."""
    
    def test_normalize_track_name(self):
        """Test track name normalization."""
        assert normalize_track_name("Machine Learning") == "machine learning"
        assert normalize_track_name("  WEB DEVELOPMENT  ") == "web development"
    
    def test_track_aliases(self):
        """Test track name aliases."""
        assert normalize_track_name("ML") == "machine learning"
        assert normalize_track_name("AI") == "artificial intelligence"
        assert normalize_track_name("web dev") == "web development"
    
    def test_extract_track_from_filename(self):
        """Test extracting track name from PDF filename."""
        assert extract_track_from_filename("machine_learning.pdf") == "machine learning"
        assert extract_track_from_filename("web_development.pdf") == "web development"
        assert extract_track_from_filename("data_science.pdf") == "data science"
