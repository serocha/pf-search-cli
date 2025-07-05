#!/usr/bin/env python3
"""
Process pages from TTRPG datasets and create embeddings
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from embedder import TTRPGEmbedder
from vector_store import FAISSVectorStore

# Set up logging with reduced verbosity
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class PageProcessor:
    """Process pages and create vector embeddings"""
    
    def __init__(self, config_path: str = "config.json"):
        self.embedder = TTRPGEmbedder(config_path)
        self.config = self.embedder.config
        self.data_dir = Path(self.config["paths"]["data_dir"])
        self.output_dir = Path(self.config["paths"]["output_dir"])
        self.output_dir.mkdir(exist_ok=True)
        self.batch_size = self.config["processing"]["batch_size"]
        
        logger.info(f"Initialized processor. Data: {self.data_dir}, Output: {self.output_dir}")
    
    def _extract_page_number(self, file_path: Path) -> Optional[int]:
        """Extract page number from filename like 'page_001.txt'"""
        try:
            return int(file_path.stem.split('_')[1])
        except (IndexError, ValueError):
            logger.warning(f"Could not extract page number from {file_path.name}")
            return None
    
    def _get_page_files(self, dataset_path: Path) -> List[Path]:
        """Get all page files from dataset directory"""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        page_files = sorted(list(dataset_path.glob("page_*.txt")))
        if not page_files:
            raise FileNotFoundError(f"No page files found in {dataset_path}")
        
        return page_files
    
    def _process_single_page(self, page_file: Path, dataset_name: str = "") -> List[Dict[str, Any]]:
        """Process a single page file and return chunk data"""
        try:
            content = page_file.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading {page_file.name}: {e}")
            raise
        
        if not content.strip():
            logger.warning(f"Empty page: {page_file.name}")
            return []
        
        chunks_data = self.embedder.chunk_text(content, page_file.name, dataset_name)
        if chunks_data:
            logger.info(f"Processed {page_file.name}: {len(chunks_data)} chunks")
        return chunks_data
    
    def _process_batch(self, page_files: List[Path], dataset_name: str = "") -> Tuple[List[str], List[Any], List[Dict]]:
        """Process a batch of page files with rate-limited embedding"""
        # Collect all chunks from pages
        all_chunks_data = []
        for page_file in page_files:
            chunks_data = self._process_single_page(page_file, dataset_name)
            all_chunks_data.extend(chunks_data)
        
        if not all_chunks_data:
            logger.warning("No chunks to embed in this batch")
            return [], [], []
        
        # Batch embed all texts
        texts = [chunk['text'] for chunk in all_chunks_data]
        logger.info(f"Embedding {len(texts)} chunks...")
        
        try:
            embeddings = self.embedder.embed_texts_batch(texts)
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            raise
        
        # Extract results
        chunks = [chunk['text'] for chunk in all_chunks_data]
        metadata = [chunk['metadata'] for chunk in all_chunks_data]
        
        logger.info(f"Successfully embedded {len(embeddings)} chunks")
        return chunks, embeddings, metadata
    
    def _save_embeddings(self, dataset_name: str, chunks: List[str], embeddings: List[Any], 
                        metadata: List[Dict], total_pages: int, start_page: Optional[int] = None, 
                        end_page: Optional[int] = None) -> Path:
        """Save embedding results to JSON file"""
        # Create filename with optional range suffix
        suffix = ""
        if start_page is not None or end_page is not None:
            start = start_page or 1
            end = end_page or total_pages
            suffix = f"_pages_{start:03d}-{end:03d}"
        
        output_file = self.output_dir / f"{dataset_name}{suffix}_embeddings.json"
        
        result_data = {
            "dataset_name": dataset_name,
            "total_chunks": len(chunks),
            "chunks": chunks,
            "embeddings": embeddings,
            "metadata": metadata,
            "model": self.embedder.model_name,
            "config": self.config
        }
        
        try:
            output_file.write_text(json.dumps(result_data, indent=2, ensure_ascii=False), encoding='utf-8')
        except Exception as e:
            logger.error(f"Error saving embeddings to {output_file}: {e}")
            raise
        
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")
        return output_file
    
    def process_dataset(self, dataset_name: str, start_page: Optional[int] = None, 
                       end_page: Optional[int] = None, rebuild_vector_store: bool = True) -> Dict[str, Any]:
        """Process a specific dataset directory with optional page range"""
        dataset_path = self.data_dir / dataset_name
        
        # Get all page files first, then filter if needed
        all_page_files = self._get_page_files(dataset_path)
        
        if start_page is None and end_page is None:
            page_files = all_page_files
        else:
            # Filter by page range
            page_files = []
            for file_path in all_page_files:
                page_num = self._extract_page_number(file_path)
                if page_num is None:
                    continue
                if start_page is not None and page_num < start_page:
                    continue
                if end_page is not None and page_num > end_page:
                    continue
                page_files.append(file_path)
            
            if not page_files:
                raise ValueError(f"No pages found in range {start_page}-{end_page}")
        
        # Log processing info
        page_range_info = f" (pages {start_page or 1}-{end_page or len(all_page_files)})" if (start_page or end_page) else ""
        logger.info(f"Processing {len(page_files)} pages in {dataset_name}{page_range_info}")
        
        # Process in batches
        all_chunks, all_embeddings, all_metadata = [], [], []
        
        for i in range(0, len(page_files), self.batch_size):
            batch_files = page_files[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info(f"Processing batch {batch_num}: pages {i+1}-{min(i+self.batch_size, len(page_files))}")
            
            chunks, embeddings, metadata = self._process_batch(batch_files, dataset_name)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
            all_metadata.extend(metadata)
        
        # Save results
        self._save_embeddings(dataset_name, all_chunks, all_embeddings, all_metadata, 
                            len(all_page_files), start_page, end_page)
        
        # Optionally rebuild vector store
        if rebuild_vector_store:
            logger.info("Rebuilding vector store...")
            self._create_vector_store()
        
        return {
            "total_chunks": len(all_chunks),
            "total_pages": len(page_files)
        }
    
    def process_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Process all datasets in the data directory"""
        dataset_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if not dataset_dirs:
            logger.warning(f"No dataset directories found in {self.data_dir}")
            return {}
        
        logger.info(f"Found {len(dataset_dirs)} datasets: {[d.name for d in dataset_dirs]}")
        
        results = {}
        for dataset_dir in dataset_dirs:
            try:
                result = self.process_dataset(dataset_dir.name)
                results[dataset_dir.name] = result
            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_dir.name}: {e}")
                results[dataset_dir.name] = {"error": str(e)}
        
        self._create_vector_store()
        return results
    
    def _create_vector_store(self):
        """Create and save vector store from all embedding files"""
        embedding_files = list(self.output_dir.glob("*_embeddings.json"))
        
        if not embedding_files:
            logger.warning("No embedding files found for vector store creation")
            return
        
        logger.info(f"Creating vector store from {len(embedding_files)} embedding files")
        
        # Initialize vector store
        vector_store = FAISSVectorStore(
            dimension=self.config["embedding"]["dimension"],
            index_type=self.config["vector_store"]["index_type"]
        )
        
        # Load and add all embeddings
        total_added = 0
        for file_path in embedding_files:
            try:
                data = json.loads(file_path.read_text(encoding='utf-8'))
                embeddings, chunks, metadata = data["embeddings"], data["chunks"], data["metadata"]
                vector_store.add_embeddings(embeddings, chunks, metadata)
                total_added += len(embeddings)
                logger.info(f"Added {len(embeddings)} vectors from {file_path.name}")
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
        
        # Save vector store
        if total_added > 0:
            store_path = self.output_dir / self.config["paths"]["vector_store_name"]
            vector_store.save(str(store_path))
            logger.info(f"Saved vector store with {total_added} vectors to {store_path}")
        else:
            logger.error("No vectors added to store")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process TTRPG pages for vector search",
        epilog="""Examples:
  %(prog)s --dataset 'Pathfinder 2e Player Core - Clean Pages'
  %(prog)s --dataset 'Dataset Name' --start-page 1 --end-page 10
  %(prog)s --dataset 'Dataset Name' --no-rebuild
  %(prog)s --rebuild-only
  %(prog)s --all""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=str, help="Process specific dataset directory")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--start-page", type=int, help="Starting page number (1-based, inclusive)")
    parser.add_argument("--end-page", type=int, help="Ending page number (1-based, inclusive)")
    parser.add_argument("--no-rebuild", action="store_true", help="Don't rebuild vector store after processing")
    parser.add_argument("--rebuild-only", action="store_true", help="Only rebuild vector store from existing embedding files")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_page is not None and args.start_page < 1:
        parser.error("--start-page must be 1 or greater")
    if args.end_page is not None and args.end_page < 1:
        parser.error("--end-page must be 1 or greater")
    if args.start_page is not None and args.end_page is not None and args.start_page > args.end_page:
        parser.error("--start-page cannot be greater than --end-page")
    if (args.start_page or args.end_page) and args.all:
        parser.error("Cannot use page range with --all")
    if args.rebuild_only and (args.dataset or args.all):
        parser.error("--rebuild-only cannot be used with processing options")
    if not (args.dataset or args.all or args.rebuild_only):
        parser.print_help()
        return
    
    try:
        processor = PageProcessor(args.config)
        
        if args.rebuild_only:
            print("🔄 Rebuilding vector store from existing embedding files...")
            processor._create_vector_store()
            print("✅ Vector store rebuilt successfully")
        
        elif args.dataset:
            rebuild_store = not args.no_rebuild
            result = processor.process_dataset(args.dataset, args.start_page, args.end_page, rebuild_store)
            page_info = f"pages {args.start_page or 1}-{args.end_page or result['total_pages']}" if (args.start_page or args.end_page) else "all pages"
            print(f"✅ Processed {args.dataset} ({page_info}): {result['total_chunks']} chunks from {result['total_pages']} pages")
            
            if not rebuild_store:
                print("ℹ️  Vector store was not rebuilt. Run with --rebuild-only to update it.")
        
        elif args.all:
            results = processor.process_all_datasets()
            print("✅ Processing complete:")
            for dataset, info in results.items():
                status = "❌" if "error" in info else "✅"
                message = info.get("error", f"{info['total_chunks']} chunks from {info['total_pages']} pages")
                print(f"  {status} {dataset}: {message}")
        
        print("\nUse terminal_search.py to search the vector database")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 