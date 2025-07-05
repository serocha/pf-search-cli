#!/usr/bin/env python3
"""
LLM client for Gemini integration with RAG functionality
"""
import os
import json
import logging
import time
from typing import List, Dict, Any
import google.genai as genai
from google.genai import types

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Suppress httpx logging specifically (this causes the "HTTP Request" messages)
logging.getLogger("httpx").setLevel(logging.WARNING)

class TTRPGLLMClient:
    """LLM client for answering TTRPG questions using RAG"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        
        # Initialize client
        from util import get_api_key
        api_key = get_api_key()
        
        self.client = genai.Client(api_key=api_key)
        
        # Cache config values
        llm_config = self.config["llm"]
        rate_config = self.config.get("rate_limiting", {})
        
        self.model_name = llm_config["model_name"]
        self.system_instruction = llm_config["system_instruction"]
        self.max_output_tokens = llm_config["max_output_tokens"]
        self.temperature = llm_config["temperature"]
        self.top_results = llm_config["top_results_for_context"]
        
        self.rate_limit_delay = rate_config.get("delay_seconds", 1.0)
        
        logger.info(f"Initialized LLM client with model: {self.model_name}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise
    
    def _create_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context from search results"""
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results[:self.top_results], 1):
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            section = metadata.get('section', 'Unknown')
            dataset = metadata.get('dataset', 'Unknown Rulebook')
            similarity = result.get('similarity', 0.0)
            
            # Format: "Context X (from Rulebook: Dataset, Section, similarity: X.XXX):"
            context_parts.append(
                f"Context {i} (from Rulebook: {dataset}, {section}, similarity: {similarity:.3f}):\n{text}"
            )
        
        return "\n\n".join(context_parts)
    
    def _rate_limit(self):
        """Apply rate limiting if needed"""
        if hasattr(self, '_last_request_time'):
            time_since_last = time.time() - self._last_request_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
    
    def answer_question(self, question: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate an answer to a TTRPG question using RAG"""
        
        context = self._create_context(search_results)
        
        if not context:
            return f'Not found.'
        
        # Create prompt using f-string template
        prompt = f"""Based on the context only, answer the user's question. Be accurate. cite the Rulebook Name & page number(s).

CONTEXT:
{context}

QUESTION: "{question}"

Format: Use **bold text** to highlight. Use _italics_ to subtext.
Rulebook Name: Smartly edit the dataset name to make it human readable, e.g. **Player Core 1, p.100**.
Accuracy: Repeat retrieved info VERBATIM. Include all relevant info for the rule, stats, class, etc.

ANSWER:"""
        
        try:
            self._rate_limit()
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    max_output_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                ),
            )
            
            self._last_request_time = time.time()
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return f"Sorry, I couldn't generate a response: {str(e)}" 