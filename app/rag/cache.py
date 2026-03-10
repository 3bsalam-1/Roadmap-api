import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from app.core.config import settings
from app.core.logging import logger


class GenerationCache:
    """
    Manages the JSON cache for LLM-generated roadmaps.
    
    This cache is completely separate from ChromaDB and is never used 
    as a retrieval source for the RAG chain. It only stores generated
    roadmaps to avoid redundant LLM calls.
    """
    
    def __init__(self):
        self.cache_dir = Path(settings.GENERATED_CACHE_DIR)
        # Handle cloud platforms like Render where /data may be a mounted volume
        # with different permissions, as well as Windows local development
        try:
            if not self.cache_dir.exists():
                # Check if parent exists - if not, we're in a restricted environment
                if not self.cache_dir.parent.exists():
                    # Use local fallback directory in project root (data/generated)
                    logger.warning(f"Parent directory {self.cache_dir.parent} does not exist, using local fallback")
                    self.cache_dir = Path(__file__).parent.parent.parent / "data" / "generated"
                else:
                    # Parent exists, try to create the directory
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            logger.warning(f"Cannot create cache directory {self.cache_dir}: {e}")
            # Fall back to using a local directory if cloud storage is not available
            self.cache_dir = Path(__file__).parent.parent.parent / "data" / "generated"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError as e:
            logger.warning(f"Cannot create cache directory {self.cache_dir}: {e}")
            # Fall back to using a local directory
            self.cache_dir = Path(__file__).parent.parent.parent / "data" / "generated"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"GenerationCache initialized at: {self.cache_dir}")
    
    def _key(self, track: str, level: str) -> str:
        """
        Normalize track + level into a safe filename key.
        
        Args:
            track: The track name (e.g., "flutter development")
            level: The difficulty level (e.g., "beginner")
            
        Returns:
            A safe filename key (e.g., "flutter_development__beginner")
        """
        normalized = track.lower().strip().replace(" ", "_")
        return f"{normalized}__{level}"
    
    def get(self, track: str, level: str) -> Optional[Dict[str, Any]]:
        """
        Return cached roadmap if it exists, else None.
        
        Args:
            track: The track name
            level: The difficulty level
            
        Returns:
            Cached roadmap dict or None if not found
        """
        key = self._key(track, level)
        path = self.cache_dir / f"{key}.json"
        
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                logger.info(f"Cache hit for: {key}")
                return cached
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading cache file {path}: {e}")
                return None
        
        logger.info(f"Cache miss for: {key}")
        return None
    
    def save(self, track: str, level: str, roadmap: Dict[str, Any]) -> None:
        """
        Save a generated roadmap to the JSON cache.
        
        Args:
            track: The track name
            level: The difficulty level
            roadmap: The roadmap dictionary to cache
        """
        key = self._key(track, level)
        path = self.cache_dir / f"{key}.json"
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(roadmap, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached roadmap for: {key}")
        except IOError as e:
            logger.error(f"Error saving cache file {path}: {e}")
            raise
    
    def list_cached(self) -> List[str]:
        """
        Return list of all cached track keys.
        
        Returns:
            List of cached keys (e.g., ["flutter_development__beginner", "game_development__advanced"])
        """
        try:
            return [f.stem for f in self.cache_dir.glob("*.json")]
        except Exception as e:
            logger.error(f"Error listing cached files: {e}")
            return []
    
    def get_cached_tracks_by_level(self) -> Dict[str, List[str]]:
        """
        Get cached tracks organized by level.
        
        Returns:
            Dict with keys as levels and values as list of track names
        """
        cached = self.list_cached()
        result: Dict[str, List[str]] = {"beginner": [], "intermediate": [], "advanced": []}
        
        for key in cached:
            # Parse key format: "track_name__level"
            if "__" in key:
                track_part, level = key.rsplit("__", 1)
                # Convert back to display format
                track_display = track_part.replace("_", " ").title()
                if level in result:
                    result[level].append(track_display)
        
        return result


# Global instance
generation_cache = GenerationCache()
