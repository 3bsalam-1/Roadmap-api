import json
from typing import Dict, Any, List
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import logger


# Initialize LLM client
def get_llm_client() -> ChatCompletionsClient:
    """Initialize the LLM client using GitHub Models via Azure AI Inference."""
    return ChatCompletionsClient(
        endpoint=settings.GITHUB_MODELS_ENDPOINT,
        credential=AzureKeyCredential(settings.GITHUB_TOKEN),
    )


# Format documents for the prompt
def format_docs(docs: List[Document]) -> str:
    """Format documents into a context string."""
    return "\n\n".join(
        f"--- Page {doc.metadata.get('page', 'N/A')} ---\n{doc.page_content}"
        for doc in docs
    )


# RAG Prompt (when track PDF exists)
def build_rag_prompt(track: str, context: str) -> str:
    """Build prompt for RAG-based generation."""
    return f"""You are an expert learning path designer. Using ONLY the following context from verified roadmap documents, create a structured learning roadmap for: {track}

Context:
{context}

Generate the roadmap in the exact JSON format specified. Base it strictly on the provided context.

JSON format:
{{
  "track": "string",
  "total_duration_weeks": number,
  "level": "beginner|intermediate|advanced",
  "phases": [
    {{
      "phase_number": number,
      "title": "string",
      "duration_weeks": number,
      "description": "string",
      "topics": [
        {{
          "name": "string",
          "subtopics": ["string"],
          "resources": [
            {{"type": "course|book|documentation|video", "title": "string", "url": "string or null"}}
          ],
          "estimated_hours": number
        }}
      ]
    }}
  ],
  "prerequisites": ["string"],
  "career_outcomes": ["string"]
}}
"""


# Generation Prompt (when track PDF does NOT exist)
def build_generation_prompt(track: str, level: str = "beginner") -> str:
    """Build prompt for LLM-based generation."""
    return f"""You are an expert learning path designer. No pre-built roadmap exists for "{track}", so generate one from your knowledge.

Generate a comprehensive, structured learning roadmap for: {track}

IMPORTANT: The difficulty level must be: {level}

Use the same JSON format and quality level as professional roadmaps. Structure it with phases, topics, resources, and time estimates.

JSON format:
{{
  "track": "string",
  "total_duration_weeks": number,
  "level": "beginner|intermediate|advanced",
  "phases": [
    {{
      "phase_number": number,
      "title": "string",
      "duration_weeks": number,
      "description": "string",
      "topics": [
        {{
          "name": "string",
          "subtopics": ["string"],
          "resources": [
            {{"type": "course|book|documentation|video", "title": "string", "url": "string or null"}}
          ],
          "estimated_hours": number
        }}
      ]
    }}
  ],
  "prerequisites": ["string"],
  "career_outcomes": ["string"]
}}
"""


class RoadmapChain:
    """Manages the RAG and generation chains using Azure AI Inference SDK."""
    
    def __init__(self):
        self.client = get_llm_client()
        self.model = settings.LLM_MODEL
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from the LLM response."""
        # Try to find JSON in the response
        try:
            # Try direct parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            # Try to find any JSON object
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from response: {response}")
    
    def generate_from_rag(self, track: str, context: List[Document]) -> Dict[str, Any]:
        """Generate roadmap using RAG (with knowledge base context)."""
        logger.info(f"Generating roadmap via RAG for: {track}")
        
        try:
            context_str = format_docs(context)
            prompt = build_rag_prompt(track, context_str)
            
            response = self.client.complete(
                messages=[
                    SystemMessage(prompt),
                ],
                temperature=0.7,
                top_p=1.0,
                model=self.model
            )
            
            result = self._parse_json_response(response.choices[0].message.content)
            result["source"] = "knowledge_base"
            return result
        except Exception as e:
            logger.error(f"Error in RAG chain: {e}")
            raise
    
    def generate_from_llm(self, track: str, level: str = "beginner") -> Dict[str, Any]:
        """Generate roadmap using LLM's own knowledge."""
        logger.info(f"Generating roadmap via LLM for: {track}, level: {level}")
        
        try:
            prompt = build_generation_prompt(track, level)
            
            response = self.client.complete(
                messages=[
                    SystemMessage(prompt),
                ],
                temperature=0.7,
                top_p=1.0,
                model=self.model
            )
            
            result = self._parse_json_response(response.choices[0].message.content)
            result["source"] = "llm_generated"
            return result
        except Exception as e:
            logger.error(f"Error in generation chain: {e}")
            raise


# Global instance
roadmap_chain = RoadmapChain()
