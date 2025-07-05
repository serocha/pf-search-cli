"""
FAISS-based vector store for similarity search
"""
import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    # Constants
    SUPPORTED_INDEX_TYPES = {"L2", "IP"}
    DEFAULT_DIMENSION = 3072
    
    def __init__(self, dimension: int = DEFAULT_DIMENSION, index_type: str = "L2"):
        """Initialize FAISS vector store"""
        if index_type not in self.SUPPORTED_INDEX_TYPES:
            raise ValueError(f"index_type must be one of {self.SUPPORTED_INDEX_TYPES}")
            
        self.dimension = dimension
        self.index_type = index_type
        self.index = faiss.IndexFlatL2(dimension) if index_type == "L2" else faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        
        logger.info(f"Initialized FAISS index with {dimension}D vectors, type: {index_type}")
    
    def _process_embeddings(self, embeddings: List[List[float]]) -> np.ndarray:
        """Convert embeddings to numpy array and normalize if needed for IP index"""
        embeddings_np = np.array(embeddings, dtype=np.float32)
        return normalize(embeddings_np, norm='l2') if self.index_type == "IP" else embeddings_np
    
    def _create_metadata(self, count: int, start_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Create default metadata for embeddings"""
        start_id = start_id or len(self.metadata)
        return [{"id": start_id + i} for i in range(count)]
    
    def _get_file_paths(self, filepath: str) -> Tuple[Path, Path]:
        """Get FAISS and JSON file paths"""
        base_path = Path(filepath)
        return base_path.with_suffix('.faiss'), base_path.with_suffix('.json')
    
    def _calculate_similarity(self, distance: float) -> float:
        """Calculate similarity score from distance"""
        return 1 - distance if self.index_type == "L2" else float(distance)
    
    def add_embeddings(self, embeddings: List[List[float]], texts: List[str], 
                      metadata: Optional[List[Dict[str, Any]]] = None):
        """Add embeddings to the vector store"""
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")
        
        # Process and add embeddings
        embeddings_np = self._process_embeddings(embeddings)
        self.index.add(embeddings_np)
        
        # Store texts and metadata
        self.texts.extend(texts)
        self.metadata.extend(metadata or self._create_metadata(len(texts)))
        
        logger.info(f"Added {len(embeddings)} embeddings to vector store. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        if self.index.ntotal == 0:
            return []
        
        # Process query and search
        query_np = self._process_embeddings([query_embedding])
        distances, indices = self.index.search(query_np, min(k, self.index.ntotal))
        
        # Build results
        return [
            {
                "rank": i + 1,
                "distance": float(distance),
                "similarity": self._calculate_similarity(distance),
                "text": self.texts[idx],
                "metadata": self.metadata[idx].copy()
            }
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0]))
            if idx >= 0
        ]
    
    def save(self, filepath: str):
        """Save the vector store to disk"""
        faiss_path, json_path = self._get_file_paths(filepath)
        
        # Save FAISS index and metadata
        faiss.write_index(self.index, str(faiss_path))
        
        data = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "texts": self.texts,
            "metadata": self.metadata
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved vector store to {filepath}")
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        faiss_path, json_path = self._get_file_paths(filepath)
        
        # Load FAISS index and metadata
        self.index = faiss.read_index(str(faiss_path))
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Restore state
        for attr in ["dimension", "index_type", "texts", "metadata"]:
            setattr(self, attr, data[attr])
        
        logger.info(f"Loaded vector store from {filepath}. Contains {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_texts": len(self.texts),
            "total_metadata": len(self.metadata)
        }

def load_embeddings_from_json(filepath: str) -> Tuple[List[List[float]], List[str], List[Dict[str, Any]]]:
    """Load embeddings from a JSON file generated by the embedding service"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    embeddings = data["embeddings"]
    texts = data["chunks"]
    
    # Use existing metadata or create legacy format
    if "metadata" in data and isinstance(data["metadata"], list):
        metadata = data["metadata"]
    else:
        # Create legacy metadata with minimal required fields
        metadata = [
            {
                "chunk_id": i,
                "filename": data["filename"],
                "model": data["model"],
                "text_preview": text[:100] + ("..." if len(text) > 100 else ""),
                "section": "Legacy",
                "content_type": "general",
                "tokens": len(text) // 4,
                "start_line": 0,
                "end_line": 0
            }
            for i, text in enumerate(texts)
        ]
    
    return embeddings, texts, metadata 