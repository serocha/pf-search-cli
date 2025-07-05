#!/usr/bin/env python3
"""
TTRPG Vector Search System - Simple CLI Entry Point
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import main functionality
from page_processor import PageProcessor
from terminal_search import main as terminal_search_main

def show_help():
    """Show simple help"""
    print("""🎲 Rule Search - Quick Commands
=====================================

Commands:
  python run.py list                    - List available datasets
  python run.py process 'Dataset Name' [start] [end]  - Process pages
  python run.py rebuild                 - Rebuild vector store only
  python run.py search                  - Start AI search terminal
  python run.py check                   - Check system setup
  python run.py help                    - Show this help

Examples:
  python run.py check
  python run.py process 'Pathfinder 2e Player Core - Clean Pages' 1 10
  python run.py search
""")

def list_datasets():
    """List available datasets"""
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    datasets = [d for d in data_dir.iterdir() if d.is_dir()]
    if not datasets:
        print("❌ No datasets found in data/")
        return False
    
    print("Available Datasets:")
    for dataset in datasets:
        page_files = list(dataset.glob("page_*.txt"))
        print(f"  - '{dataset.name}' ({len(page_files)} pages)")
    return True

def check_setup():
    """Check if the system is properly set up"""
    print("🔧 Checking system setup...")
    
    issues = []
    
    # Check config file
    config_file = Path("config.json")
    if config_file.exists():
        print("✅ config.json found")
    else:
        issues.append("❌ config.json not found")
    
    # Check data directory and datasets
    data_dir = Path("data")
    if not data_dir.exists():
        issues.append("❌ data/ directory not found")
    else:
        datasets = [d for d in data_dir.iterdir() if d.is_dir()]
        if not datasets:
            issues.append("❌ No dataset directories found in data/")
        else:
            print(f"✅ Found {len(datasets)} dataset(s):")
            for dataset in datasets:
                page_files = list(dataset.glob("page_*.txt"))
                print(f"    - {dataset.name} ({len(page_files)} pages)")
    
    # Check API key
    try:
        from util import get_api_key
        get_api_key()
        print("✅ GEMINI_API_KEY is set")
    except ValueError as e:
        issues.append(f"❌ {e}")
    except ImportError:
        issues.append("❌ Could not import util module")
    
    # Check output directory
    output_dir = Path("output")
    if output_dir.exists():
        embeddings = list(output_dir.glob("*_embeddings.json"))
        vector_store = list(output_dir.glob("vector_store.*"))
        
        if embeddings:
            print(f"✅ Found {len(embeddings)} embedding file(s)")
        
        print("✅ Vector store found" if vector_store else "ℹ️  No vector store found - run process command to create")
    else:
        print("ℹ️  Output directory will be created automatically")
    
    # Check LLM availability
    try:
        from llm_client import TTRPGLLMClient
        TTRPGLLMClient()
        print("✅ LLM client initialized - AI search available")
    except Exception as e:
        issues.append(f"⚠️  LLM client unavailable: {str(e)[:50]}...")
        print("ℹ️  Vector search will work, but AI features won't be available")
    
    # Display results
    if issues:
        print("\n🔧 Setup Issues:")
        for issue in issues:
            print(f"  {issue}")
        return False
    
    print("\n🎉 System appears to be set up correctly!")
    return True

def process_dataset(dataset_name, start_page=None, end_page=None):
    """Process a dataset using PageProcessor directly"""
    try:
        processor = PageProcessor()
        result = processor.process_dataset(
            dataset_name=dataset_name,
            start_page=start_page,
            end_page=end_page
        )
        print(f"✅ Processed {result['total_pages']} pages, {result['total_chunks']} chunks")
        return True
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return False

def rebuild_vector_store():
    """Rebuild vector store only"""
    try:
        processor = PageProcessor()
        processor._create_vector_store()
        print("✅ Vector store rebuilt successfully")
        return True
    except Exception as e:
        print(f"❌ Error rebuilding vector store: {e}")
        return False

def main():
    """Main command dispatcher"""
    args = sys.argv[1:]
    
    if not args or args[0] in ['help', '-h', '--help']:
        show_help()
        return
    
    command = args[0].lower()
    
    if command == 'list':
        list_datasets()
    elif command == 'check':
        check_setup()
    elif command == 'process':
        if len(args) < 2:
            print("❌ Dataset name required")
            print("Usage: python run.py process 'Dataset Name' [start_page] [end_page]")
            return
        
        dataset_name = args[1]
        start_page = int(args[2]) if len(args) > 2 else None
        end_page = int(args[3]) if len(args) > 3 else None
        
        print(f"Processing dataset: {dataset_name}")
        if start_page:
            print(f"Starting from page: {start_page}")
        if end_page:
            print(f"Ending at page: {end_page}")
        
        process_dataset(dataset_name, start_page, end_page)
    elif command == 'rebuild':
        rebuild_vector_store()
    elif command == 'search':
        print("Starting search terminal...")
        terminal_search_main()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python run.py help' for available commands")

if __name__ == "__main__":
    main() 