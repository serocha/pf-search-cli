#!/usr/bin/env python3
"""
Search methods for the TTRPG vector database with LLM integration
"""
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from embedder import TTRPGEmbedder
from vector_store import FAISSVectorStore
from llm_client import TTRPGLLMClient

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class TTRPGSearcher:
    """Search interface for TTRPG vector database with LLM integration"""
    
    def __init__(self, config_path: str = "config.json"):
        self.embedder = TTRPGEmbedder(config_path)
        self.config = self.embedder.config
        self.vector_store = None
        self.llm_client = self._init_llm_client(config_path)
        self.output_dir = Path(self.config["paths"]["output_dir"])
        self.load_vector_store()
    
    def _init_llm_client(self, config_path: str) -> Optional[TTRPGLLMClient]:
        """Initialize LLM client with error handling"""
        try:
            return TTRPGLLMClient(config_path)
        except Exception as e:
            logger.warning(f"Could not initialize LLM client: {e}")
            return None
    
    def _require_vector_store(self):
        """Ensure vector store is loaded, raise if not"""
        if not self.vector_store:
            raise ValueError("Vector store not loaded")
    
    def _create_filter_func(self, content_type: Optional[str] = None, 
                          filter_field: Optional[str] = None, 
                          filter_value: Optional[str] = None) -> Optional[Callable]:
        """Create filter function based on parameters"""
        if content_type:
            return lambda r: r.get('metadata', {}).get('content_type') == content_type
        elif filter_field and filter_value:
            return lambda r: filter_value.lower() in r.get('metadata', {}).get(filter_field, '').lower()
        return None
    
    def _build_response(self, query: str, results: List[Dict], **kwargs) -> Dict[str, Any]:
        """Build standard response dictionary"""
        base_response = {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
        base_response.update(kwargs)
        return base_response
    
    def load_vector_store(self):
        """Load vector store from disk"""
        store_path = self.output_dir / self.config["paths"]["vector_store_name"]
        
        if store_path.with_suffix('.faiss').exists():
            self.vector_store = FAISSVectorStore()
            self.vector_store.load(str(store_path))
            stats = self.vector_store.get_stats()
            logger.info(f"Loaded {stats['total_vectors']} vectors")
        else:
            logger.warning(f"Vector store not found at {store_path}")
    
    def _search_with_filter(self, query: str, k: int, filter_func: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Unified search with optional filtering"""
        self._require_vector_store()
        assert self.vector_store is not None  # Help type checker after validation
        
        # Search with extra results to allow for filtering
        search_k = k * 3 if filter_func else k
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, k=search_k)
        
        # Apply filter and limit results
        if filter_func and results:
            results = [r for r in results if filter_func(r)][:k]
        
        # Update ranks
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        return results
    
    def search(self, query: str, k: int = 5, content_type: Optional[str] = None, 
               filter_field: Optional[str] = None, filter_value: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for similar content with optional filtering
        
        Args:
            query: Search query
            k: Number of results to return
            content_type: Filter by content type ('rules', 'character', 'magic', 'equipment')
            filter_field: Generic filter field ('section', 'filename', etc.)
            filter_value: Value to filter by
        """
        try:
            filter_func = self._create_filter_func(content_type, filter_field, filter_value)
            results = self._search_with_filter(query, k, filter_func)
            
            # Build filter info for response
            filter_info = {}
            if content_type:
                filter_info['content_type'] = content_type
            elif filter_field and filter_value:
                filter_info[filter_field] = filter_value
            
            return self._build_response(query, results, **filter_info)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"error": str(e)}
    
    def search_with_llm(self, query: str, k: Optional[int] = None, content_type: Optional[str] = None) -> Dict[str, Any]:
        """Search and generate an LLM response using RAG"""
        if not self.llm_client:
            return {"error": "LLM client not available"}
        
        # Use config default if k not specified
        k = k or self.config.get("llm", {}).get("top_results_for_context", 5)
        
        # Perform vector search
        search_result = self.search(query, k, content_type)
        if "error" in search_result or search_result["total_results"] == 0:
            llm_response = "Not found." if search_result.get("total_results", 0) == 0 else search_result["error"]
            return {
                "query": query,
                "search_results": search_result.get('results', []),
                "llm_response": llm_response,
                "sources_used": 0
            }
        
        # Generate LLM response
        try:
            llm_response = self.llm_client.answer_question(query, search_result['results'])
            actual_sources_used = min(len(search_result['results']), self.llm_client.top_results)
            
            return {
                "query": query,
                "search_results": search_result['results'],
                "llm_response": llm_response,
                "llm_success": True,
                "sources_used": actual_sources_used,
                "model": self.llm_client.model_name
            }
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {
                "query": query,
                "search_results": search_result['results'],
                "llm_response": f"Error generating response: {str(e)}",
                "llm_success": False,
                "sources_used": 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        self._require_vector_store()
        assert self.vector_store is not None  # Help type checker after validation
        
        stats = self.vector_store.get_stats()
        
        # Aggregate metadata using defaultdict for cleaner code
        metadata_fields = ['content_type', 'filename', 'section']
        for field in metadata_fields:
            aggregated = defaultdict(int)
            for metadata in self.vector_store.metadata:
                value = metadata.get(field, 'unknown')
                aggregated[value] += 1
            stats[f"{field}s"] = dict(aggregated)
        
        stats.update({
            "unique_files": len(stats['filenames']),
            "model": self.embedder.model_name,
            "llm_available": self.llm_client is not None
        })
        
        return stats
    
    def list_available(self, field: str) -> List[str]:
        """Get list of available values for a metadata field"""
        self._require_vector_store()
        assert self.vector_store is not None  # Help type checker after validation
        
        values = {
            metadata.get(field) 
            for metadata in self.vector_store.metadata 
            if metadata.get(field) and metadata.get(field) != 'unknown'
        }
        
        # Filter out None values and ensure all are strings
        string_values = {v for v in values if v is not None}
        return sorted(string_values)

def main():
    """Basic validation that the search system can initialize"""
    try:
        searcher = TTRPGSearcher()
        
        if not searcher.vector_store:
            print("❌ Vector store not loaded. Run page_processor.py first.")
            return
        
        stats = searcher.get_stats()
        print(f"✅ Search system ready: {stats['total_vectors']} vectors from {stats['unique_files']} files")
        print(f"   LLM available: {stats['llm_available']}")
        
    except Exception as e:
        print(f"❌ Error initializing search system: {e}")

if __name__ == "__main__":
    main() 