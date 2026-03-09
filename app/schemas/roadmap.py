from pydantic import BaseModel
from typing import Optional, Literal


class RoadmapRequest(BaseModel):
    track: str  # e.g., "machine learning", "flutter development"
    level: Optional[Literal["beginner", "intermediate", "advanced"]] = None


class ResourceSchema(BaseModel):
    type: Literal["course", "book", "documentation", "video"]
    title: str
    url: Optional[str] = None


class TopicSchema(BaseModel):
    name: str
    subtopics: list[str]
    resources: list[ResourceSchema]
    estimated_hours: int


class PhaseSchema(BaseModel):
    phase_number: int
    title: str
    duration_weeks: int
    description: str
    topics: list[TopicSchema]


class RoadmapResponse(BaseModel):
    track: str
    total_duration_weeks: int
    level: str
    source: Literal["knowledge_base", "llm_generated"]
    phases: list[PhaseSchema]
    prerequisites: list[str]
    career_outcomes: list[str]


class TracksResponse(BaseModel):
    knowledge_base: list[str]
    cached: list[str]


class HealthResponse(BaseModel):
    status: str
    chromadb_status: str
    documents_indexed: int
