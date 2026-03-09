import json
import re
from typing import Optional, Tuple

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage
from azure.core.credentials import AzureKeyCredential

from app.core.config import settings
from app.core.logging import logger


# Prompt template for parsing user prompts
PARSE_PROMPT_TEMPLATE = """You are a prompt parser. Extract the learning track and difficulty level from the user's prompt.

Rules:
1. Extract the MAIN learning topic/track (e.g., "python", "machine learning", "react", "web development")
2. Extract the difficulty level if mentioned (beginner, intermediate, advanced)
3. If no level is specified, use "beginner" as default
4. Return ONLY valid JSON, no markdown, no explanations

Return this exact JSON format:
{{
    "track": "extracted track name in lowercase",
    "level": "beginner | intermediate | advanced"
}}

Example conversions:
- "i want to learn python" → {{"track": "python", "level": "beginner"}}
- "generate a roadmap for machine learning for advanced" → {{"track": "machine learning", "level": "advanced"}}
- "python roadmap for beginners" → {{"track": "python", "level": "beginner"}}
- "react js for intermediate developers" → {{"track": "react", "level": "intermediate"}}
- "i want you to generate python roadmap for beginner" → {{"track": "python", "level": "beginner"}}

User's prompt: {user_prompt}
"""


class PromptParser:
    """Parses natural language prompts to extract track and level using LLM."""
    
    def __init__(self):
        self.client = ChatCompletionsClient(
            endpoint=settings.GITHUB_MODELS_ENDPOINT,
            credential=AzureKeyCredential(settings.GITHUB_TOKEN),
        )
        self.model = settings.LLM_MODEL
    
    def parse(self, user_prompt: str) -> Tuple[str, str]:
        """
        Parse a natural language prompt to extract track and level.
        
        Args:
            user_prompt: The user's natural language prompt
            
        Returns:
            Tuple of (track, level)
        """
        logger.info(f"Parsing prompt: {user_prompt}")
        
        # Check if prompt is already simple (structured)
        if self._is_simple_prompt(user_prompt):
            return self._extract_simple(user_prompt)
        
        # Use LLM to parse complex prompts
        try:
            return self._parse_with_llm(user_prompt)
        except Exception as e:
            logger.error(f"Error parsing prompt with LLM: {e}")
            # Fallback: try simple extraction
            return self._extract_simple(user_prompt)
    
    def _is_simple_prompt(self, prompt: str) -> bool:
        """Check if prompt is already in simple format."""
        # Simple prompts are short and contain just the track name
        # e.g., "python", "machine learning", "web development"
        cleaned = prompt.lower().strip()
        
        # If it's very short and doesn't contain common NLP words
        simple_indicators = [
            len(cleaned.split()) <= 2,  # Short phrase
            not any(word in cleaned for word in [
                'want', 'generate', 'create', 'roadmap', 'learn',
                'teach', 'show', 'make', 'help', 'please'
            ])
        ]
        
        return all(simple_indicators)
    
    def _extract_simple(self, prompt: str) -> Tuple[str, str]:
        """Extract track and level from simple prompts."""
        cleaned = prompt.lower().strip()
        
        # Extract level if present
        level = "beginner"
        if "advanced" in cleaned:
            level = "advanced"
        elif "intermediate" in cleaned:
            level = "intermediate"
        
        # Clean up the track name
        # Remove common phrases
        track = cleaned
        for phrase in ["roadmap", "for", "to", "learn", "generate", "create"]:
            track = track.replace(phrase, "")
        track = " ".join(track.split())  # Normalize whitespace
        
        # Remove level words from track
        for level_word in ["beginner", "intermediate", "advanced", "developer", "developers"]:
            track = track.replace(level_word, "")
        track = " ".join(track.split())
        
        if not track:
            track = cleaned  # Fallback to original
            
        logger.info(f"Simple parse result: track='{track}', level='{level}'")
        return track, level
    
    def _parse_with_llm(self, user_prompt: str) -> Tuple[str, str]:
        """Use LLM to parse complex natural language prompts."""
        prompt = PARSE_PROMPT_TEMPLATE.format(user_prompt=user_prompt)
        
        response = self.client.complete(
            messages=[
                SystemMessage(prompt),
            ],
            temperature=0.1,  # Low temperature for consistent parsing
            top_p=1.0,
            model=self.model
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON from response
        result = self._parse_json_response(content)
        
        track = result.get("track", user_prompt).strip()
        level = result.get("level", "beginner").strip().lower()
        
        # Validate level
        if level not in ["beginner", "intermediate", "advanced"]:
            level = "beginner"
        
        logger.info(f"LLM parse result: track='{track}', level='{level}'")
        return track, level
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        # Try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON from response: {response}")


# Global instance
prompt_parser = PromptParser()
