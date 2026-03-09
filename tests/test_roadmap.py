"""Tests for the Roadmap API endpoints."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.roadmap import RoadmapRequest, RoadmapResponse


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    @patch('app.api.routes.roadmap.vector_store_manager')
    def test_health_check_healthy(self, mock_manager):
        """Test health check when ChromaDB is connected."""
        mock_manager.get_document_count.return_value = 100
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "documents_indexed" in data
    
    @patch('app.api.routes.roadmap.vector_store_manager')
    def test_health_check_error(self, mock_manager):
        """Test health check when ChromaDB has an error."""
        mock_manager.get_document_count.side_effect = Exception("Connection error")
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200  # Still returns 200, just with degraded status
        data = response.json()
        assert data["status"] == "degraded"


class TestTracksEndpoint:
    """Tests for the /roadmap/tracks endpoint."""
    
    @patch('app.api.routes.roadmap.track_detector')
    def test_get_available_tracks(self, mock_detector):
        """Test getting available tracks."""
        mock_detector.get_available_tracks.return_value = [
            "machine learning",
            "web development",
            "devops"
        ]
        
        response = client.get("/api/v1/roadmap/tracks")
        
        assert response.status_code == 200
        data = response.json()
        assert "tracks" in data
        assert len(data["tracks"]) == 3
    
    @patch('app.api.routes.roadmap.track_detector')
    def test_get_tracks_error(self, mock_detector):
        """Test error handling when getting tracks fails."""
        mock_detector.get_available_tracks.side_effect = Exception("Database error")
        
        response = client.get("/api/v1/roadmap/tracks")
        
        assert response.status_code == 503


class TestGenerateEndpoint:
    """Tests for the /roadmap/generate endpoint."""
    
    @patch('app.api.routes.roadmap.roadmap_chain')
    @patch('app.api.routes.roadmap.track_detector')
    def test_generate_known_track(self, mock_detector, mock_chain):
        """Test generating a roadmap for a known track (RAG)."""
        # Setup mocks
        mock_detector.detect.return_value = (True, [])
        mock_chain.generate_from_rag.return_value = {
            "track": "machine learning",
            "total_duration_weeks": 12,
            "level": "beginner",
            "source": "knowledge_base",
            "phases": [],
            "prerequisites": [],
            "career_outcomes": []
        }
        
        response = client.post(
            "/api/v1/roadmap/generate",
            json={"track": "machine learning", "level": "beginner"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["track"] == "machine learning"
        assert data["source"] == "knowledge_base"
    
    @patch('app.api.routes.roadmap.roadmap_chain')
    @patch('app.api.routes.roadmap.track_detector')
    def test_generate_unknown_track(self, mock_detector, mock_chain):
        """Test generating a roadmap for an unknown track (LLM generation)."""
        # Setup mocks
        mock_detector.detect.return_value = (False, [])
        mock_chain.generate_from_llm.return_value = {
            "track": "quantum computing",
            "total_duration_weeks": 16,
            "level": "advanced",
            "source": "llm_generated",
            "phases": [],
            "prerequisites": [],
            "career_outcomes": []
        }
        
        response = client.post(
            "/api/v1/roadmap/generate",
            json={"track": "quantum computing", "level": "advanced"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["track"] == "quantum computing"
        assert data["source"] == "llm_generated"
    
    def test_generate_invalid_request(self):
        """Test generating with invalid request data."""
        response = client.post(
            "/api/v1/roadmap/generate",
            json={}  # Missing required 'track' field
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.api.routes.roadmap.roadmap_chain')
    @patch('app.api.routes.roadmap.track_detector')
    def test_generate_error_handling(self, mock_detector, mock_chain):
        """Test error handling when generation fails."""
        mock_detector.detect.side_effect = Exception("Detection error")
        
        response = client.post(
            "/api/v1/roadmap/generate",
            json={"track": "machine learning"}
        )
        
        assert response.status_code == 503


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
