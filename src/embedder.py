#!/usr/bin/env python3
"""
Core embedding functionality for TTRPG content processing
"""
import os
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any
from google import genai
from google.genai.types import EmbedContentConfig

# Set up logging with reduced verbosity
logging.basicConfig(level=logging.WARNING)  # Changed from INFO to WARNING
logger = logging.getLogger(__name__)

class TTRPGEmbedder:
    """Core embedder for TTRPG content with smart chunking and rate limiting"""
    
    # Class constants
    RATE_LIMIT_ERRORS = ['rate limit', 'quota', 'too many requests', '429']
    TOKEN_LIMIT_ERRORS = ['token limit', 'too long', 'exceeds']
    SENTENCE_ENDINGS = ['. \n', '.\n', '. ', '! ', '? ']
    FORMATTING_DELIMITERS = ['```', '---', '***', '===']
    PUNCTUATION = '`~!@#$%^&*()_+-=[]{}|;:,.<>?/\\'
    METADATA_WORDS = ['page', 'printed', 'isbn', 'copyright', 'www', '.com']
    SECTION_STARTERS = ['chapter ', 'section ', 'part ', 'appendix ']
    
    # Content classification keywords
    CONTENT_KEYWORDS = {
        'rules': ['roll', 'dice', 'd20', 'combat', 'attack', 'damage'],
        'character': ['character', 'class', 'attribute', 'skill'],
        'magic': ['spell', 'magic', 'mana', 'cast'],
        'equipment': ['equipment', 'weapon', 'armor', 'item']
    }
    
    NON_CONTENT_PATTERNS = [
        r'^\d+$', r'^page \d+$', r'^www\.', r'^\w+\.com', r'^copyright', r'^printed in'
    ]
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        
        # Initialize Gemini client
        from util import get_api_key
        api_key = get_api_key()
        
        self.client = genai.Client(api_key=api_key)
        
        # Cache frequently accessed config values
        embedding_config = self.config["embedding"]
        rate_config = self.config.get("rate_limiting", {})
        chunking_config = self.config["chunking"]
        
        self.model_name = embedding_config["model_name"]
        self.dimension = embedding_config["dimension"]
        self.task_type_document = embedding_config["task_type_document"]
        self.task_type_query = embedding_config["task_type_query"]
        
        self.rate_limit_delay = rate_config.get("delay_seconds", 1.0)
        self.max_retries = rate_config.get("max_retries", 3)
        self.backoff_factor = rate_config.get("backoff_factor", 2.0)
        
        self.max_chunk_tokens = chunking_config["max_chunk_tokens"]
        self.chars_per_token = chunking_config["estimate_chars_per_token"]
        self.min_tokens = max(50, self.max_chunk_tokens // 8)
        
        logger.info(f"Initialized embedder with model: {self.model_name}")
        logger.info(f"Rate limiting: {self.rate_limit_delay}s delay, {self.max_retries} max retries")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Config error: {e}")
            raise
    
    def _handle_api_request(self, text: str, task_type: str) -> List[float]:
        """Handle single API request with retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                if attempt > 0 or hasattr(self, '_last_request_time'):
                    time_since_last = time.time() - getattr(self, '_last_request_time', 0)
                    if time_since_last < self.rate_limit_delay:
                        time.sleep(self.rate_limit_delay - time_since_last)
                
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config=EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimension),
                )
                
                self._last_request_time = time.time()
                return response.embeddings[0].values
                
            except Exception as e:
                if not self._should_retry(e, attempt, text):
                    raise
                
                backoff_time = self.rate_limit_delay * (self.backoff_factor ** attempt)
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries + 1} failed, backing off {backoff_time:.2f}s")
                time.sleep(backoff_time)
        
        raise RuntimeError(f"Failed to generate embedding after {self.max_retries + 1} attempts")
    
    def _should_retry(self, error: Exception, attempt: int, text: str) -> bool:
        """Determine if request should be retried"""
        if attempt >= self.max_retries:
            return False
            
        error_str = str(error).lower()
        
        if any(phrase in error_str for phrase in self.RATE_LIMIT_ERRORS):
            return True
        elif any(phrase in error_str for phrase in self.TOKEN_LIMIT_ERRORS):
            logger.error(f"Text too long for embedding: {len(text)} chars")
            raise ValueError(f"Text exceeds token limit: {len(text)} chars")
        else:
            logger.error(f"Embedding error: {error}")
            return True
    
    def embed_text(self, text: str, task_type: str | None = None) -> List[float]:
        """Generate embedding for a single text"""
        return self._handle_api_request(text, task_type or self.task_type_document)
    
    def embed_texts_batch(self, texts: List[str], task_type: str | None = None) -> List[List[float]]:
        """Embed multiple texts with rate limiting"""
        task_type = task_type or self.task_type_document
        embeddings = []
        total_texts = len(texts)
        
        logger.info(f"Embedding {total_texts} texts...")
        
        for i, text in enumerate(texts):
            if i > 0 and i % 10 == 0:
                logger.info(f"Progress: {i}/{total_texts} ({(i/total_texts)*100:.1f}%)")
            
            try:
                embeddings.append(self._handle_api_request(text, task_type))
            except Exception as e:
                logger.error(f"Failed to embed text {i+1}: {text[:200]}...")
                raise
        
        logger.info(f"Successfully embedded {len(embeddings)} texts")
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding optimized for query"""
        return self._handle_api_request(query, self.task_type_query)
    
    def chunk_text(self, text: str, filename: str = "", dataset_name: str = "") -> List[Dict[str, Any]]:
        """Split text into semantic chunks"""
        max_chars = self.max_chunk_tokens * self.chars_per_token
        min_chars = self.min_tokens * self.chars_per_token
        
        text = self._preprocess_text(text)
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_section = "Content"
        chunk_start_line = 0
        line_counter = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            lines_in_para = paragraph.count('\n') + 1
            
            # Handle section headers
            if self._is_section_header(paragraph):
                if len(current_chunk.strip()) >= min_chars:
                    chunks.append(self._create_chunk(
                        current_chunk.strip(), current_section, filename, chunk_start_line, line_counter-1, dataset_name
                    ))
                
                current_section = paragraph[:50]
                current_chunk = paragraph + '\n\n'
                chunk_start_line = line_counter
            else:
                current_chunk += paragraph + '\n\n'
            
            # Handle oversized chunks
            if len(current_chunk) > max_chars:
                break_point = self._find_break_point(current_chunk, max_chars)
                
                if break_point > min_chars:
                    chunk_text = current_chunk[:break_point].strip()
                    chunks.append(self._create_chunk(chunk_text, current_section, filename, chunk_start_line, line_counter, dataset_name))
                    
                    remainder = current_chunk[break_point:].strip()
                    current_chunk = remainder + '\n\n' if remainder else ""
                    chunk_start_line = line_counter
            
            line_counter += lines_in_para
        
        # Add final chunk
        if len(current_chunk.strip()) >= min_chars:
            chunks.append(self._create_chunk(current_chunk.strip(), current_section, filename, chunk_start_line, line_counter-1, dataset_name))
        
        return [chunk for chunk in chunks if self._is_valid_content(chunk['text'])]
    
    def _preprocess_text(self, text: str) -> str:
        """Clean up text before chunking"""
        return '\n'.join(
            line for line in text.split('\n')
            if not self._is_non_content(line.strip()) and (len(line.strip()) >= 3 or line.strip().isdigit())
        )
    
    def _is_non_content(self, line: str) -> bool:
        """Check if line should be filtered out"""
        return (not line or 
                line in self.FORMATTING_DELIMITERS or
                all(c in self.PUNCTUATION for c in line) or
                (len(line) <= 2 and not line.isalnum()))
    
    def _is_section_header(self, paragraph: str) -> bool:
        """Detect section headers"""
        lines = paragraph.split('\n')
        if len(lines) > 3:
            return False
        
        first_line = lines[0].strip()
        if len(first_line) < 5 or any(word in first_line.lower() for word in self.METADATA_WORDS):
            return False
        
        return any([
            first_line.isupper() and 5 <= len(first_line) <= 40 and first_line.replace(' ', '').isalpha(),
            first_line.istitle() and 8 <= len(first_line) <= 50 and ' ' in first_line,
            first_line.startswith('#'),
            bool(re.match(r'^\d+\.?\s+[A-Z][a-z]', first_line)),
            any(first_line.lower().startswith(word) for word in self.SECTION_STARTERS),
        ])
    
    def _find_break_point(self, text: str, max_chars: int) -> int:
        """Find semantic break point in text"""
        if len(text) <= max_chars:
            return len(text)
        
        search_area = text[:max_chars]
        min_break = max_chars // 2
        
        # Try paragraph breaks first
        last_para = search_area.rfind('\n\n')
        if last_para > min_break:
            return last_para + 2
        
        # Try sentence endings
        for pattern in self.SENTENCE_ENDINGS:
            last_sentence = search_area.rfind(pattern)
            if last_sentence > min_break:
                return last_sentence + len(pattern)
        
        # Fall back to word boundary
        last_space = search_area.rfind(' ')
        return last_space + 1 if last_space > min_break else max_chars
    
    def _is_valid_content(self, text: str) -> bool:
        """Check if content is worth keeping"""
        text = text.strip()
        
        if (len(text) < 20 or 
            sum(1 for c in text if c.isalpha()) < 10 or
            text.isdigit() or 
            len(text.split()) == 1):
            return False
        
        return not any(re.match(pattern, text.lower()) for pattern in self.NON_CONTENT_PATTERNS)
    
    def _create_chunk(self, text: str, section: str, filename: str, start_line: int, end_line: int, dataset_name: str = "") -> Dict[str, Any]:
        """Create chunk with metadata"""
        return {
            'text': text,
            'metadata': {
                'filename': filename,
                'dataset': dataset_name,
                'section': section,
                'start_line': start_line,
                'end_line': end_line,
                'tokens': len(text) // self.chars_per_token,
                'text_preview': text[:100] + "..." if len(text) > 100 else text,
                'content_type': self._classify_content(text)
            }
        }
    
    def _classify_content(self, text: str) -> str:
        """Classify content type using keyword matching"""
        text_lower = text.lower()
        
        for content_type, keywords in self.CONTENT_KEYWORDS.items():
            if any(word in text_lower for word in keywords):
                return content_type
        
        return 'general' 