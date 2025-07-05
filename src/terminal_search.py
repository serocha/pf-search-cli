#!/usr/bin/env python3
"""
Interactive terminal search for TTRPG vector database with LLM integration
"""
import re
import sys
import time
import threading
from search_methods import TTRPGSearcher

# Constants
def get_help_text(searcher=None):
    """Generate help text with dynamic content types"""
    content_types_str = "rules, character, magic, equipment, general"  # Default
    if searcher:
        try:
            content_types = searcher.list_available('content_type')
            if content_types:
                content_types_str = ", ".join(content_types)
        except:
            pass  # Fall back to default
    
    return f"""
Search Commands:
  <query>                    - AI-powered search (uses LLM)
  raw <query>                - Vector search only (no LLM)
  <query> --k=<number>       - Limit results (default: 5 for LLM, 5 for raw)
  <query> --type=<type>      - Filter by content type
  <query> --section=<name>   - Filter by section
  <query> --file=<name>      - Filter by filename
  <query> --full             - Show full text in results (raw search only)
  help                       - Show this help
  stats                      - Show database statistics
  types                      - List available content types
  sections                   - List available sections
  files                      - List available files
  quit                       - Exit

Content Types: {content_types_str}

Examples:
  How do combat rules work?
  raw combat rules --type=rules --k=3
  What are the different character classes?
  raw wizard spells --file=Player_Core --full"""

PARAM_PATTERNS = {
    'k': (r'--k=(\d+)', int),
    'content_type': (r'--type=(\w+)', str),
    'section': (r'--section=([^\s]+)', str),
    'filename': (r'--file=([^\s]+)', str),
}

EXIT_COMMANDS = {'quit', 'exit', 'q'}

class LoadingSpinner:
    """Simple loading spinner with animated ellipses"""
    
    def __init__(self, message="Searching"):
        self.message = message
        self.running = False
        self.thread = None
    
    def _animate(self):
        """Animate the loading dots"""
        dots = 0
        while self.running:
            # Clear current line and show message with dots
            sys.stdout.write(f'\r{self.message}{"." * (dots % 4)}{" " * (3 - (dots % 4))}')
            sys.stdout.flush()
            dots += 1
            time.sleep(0.5)
    
    def start(self):
        """Start the loading animation"""
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the loading animation and clear the line"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.6)
        # Clear the loading line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 4) + '\r')
        sys.stdout.flush()

def safe_get_metadata(result, key, default="unknown"):
    """Safely extract metadata with default"""
    return result.get("metadata", {}).get(key, default)

def check_error(results):
    """Check for errors in results and display if found"""
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return True
    return False

def display_results(results, show_full_text=False):
    """Display search results in a formatted way"""
    if check_error(results):
        return
    
    if results["total_results"] == 0:
        print("No results found")
        return
    
    print(f"📋 Found {results['total_results']} results:")
    print("-" * 60)
    
    for result in results["results"]:
        rank = result["rank"]
        similarity = result["similarity"]
        filename = safe_get_metadata(result, "filename")
        section = safe_get_metadata(result, "section")
        content_type = safe_get_metadata(result, "content_type")
        
        print(f"\n{rank}. Similarity: {similarity:.4f}")
        print(f"   File: {filename}")
        print(f"   Section: {section}")
        print(f"   Type: {content_type}")
        
        if show_full_text:
            text = result['text'][:300]
            print(f"   Text: {text}{'...' if len(result['text']) > 300 else ''}")
        else:
            print(f"   Preview: {safe_get_metadata(result, 'text_preview', 'No preview')}")
    
    print("-" * 60)

def display_llm_response(results):
    """Display LLM response with sources"""
    if check_error(results):
        return
    
    print("\n")
    print("=" * 60)
    print(results["llm_response"])
    print("=" * 60)
    
    sources_used = results.get("sources_used", 0)
    if sources_used > 0:
        print(f"\n📚 Based on {sources_used} source(s) from your TTRPG database")
        
        if results.get("search_results"):
            print("\n📖 Sources used:")
            for i, result in enumerate(results["search_results"][:sources_used], 1):
                filename = safe_get_metadata(result, "filename")
                section = safe_get_metadata(result, "section")
                similarity = result.get("similarity", 0.0)
                print(f"  {i}. {filename} - {section} (similarity: {similarity:.3f})")
    else:
        print("\nNo relevant sources found in database")

def show_stats(searcher):
    """Display database statistics"""
    stats = searcher.get_stats()
    if check_error(stats):
        return
    
    print(f"\n📊 Database Statistics:")
    print(f"  Total chunks: {stats['total_vectors']}")
    print(f"  Unique files: {stats['unique_files']}")
    print(f"  Embedding model: {stats['model']}")
    print(f"  Vector dimension: {stats['dimension']}")
    print(f"  LLM available: {'✅ Yes' if stats['llm_available'] else '❌ No'}")
    
    print(f"\nContent Types:")
    for ctype, count in stats['content_types'].items():
        print(f"  {ctype}: {count}")
    
    print(f"\nTop Files:")
    sorted_files = sorted(stats['files'].items(), key=lambda x: x[1], reverse=True)
    for filename, count in sorted_files[:10]:  # Show top 10
        print(f"  {filename}: {count} chunks")

def parse_search_command(command):
    """Parse search command and extract parameters"""
    parts = command.split()
    
    # Check if it's a raw search
    use_llm = True
    if parts and parts[0].lower() == "raw":
        use_llm = False
        parts = parts[1:]
    
    # Default parameters
    params = {
        'k': 5,
        'content_type': None,
        'section': None,
        'filename': None,
        'show_full_text': '--full' in parts,
        'use_llm': use_llm
    }
    
    # Extract parameters using regex patterns
    command_str = ' '.join(parts)
    for param, (pattern, converter) in PARAM_PATTERNS.items():
        match = re.search(pattern, command_str)
        if match:
            try:
                params[param] = converter(match.group(1))
            except ValueError:
                pass  # Keep default value
    
    # Remove parameters from command to get query
    query_parts = [part for part in parts 
                   if not (part.startswith('--') or part == '--full')]
    query = ' '.join(query_parts)
    
    return query, params

def execute_search(searcher, query, params):
    """Execute the appropriate search based on parameters"""
    use_llm = params['use_llm'] and searcher.llm_client
    section = params['section']
    filename = params['filename']
    k = params['k']
    content_type = params['content_type']
    show_full_text = params['show_full_text']
    
    spinner = LoadingSpinner("Searching")
    
    try:
        if use_llm:
            # LLM-powered search with fallback to vector search for filtering
            if section:
                print("ℹ️  Using section vector search.")
                spinner.start()
                results = searcher.search(query, k, filter_field='section', filter_value=section)
                spinner.stop()
                display_results(results, show_full_text)
            elif filename:
                print("ℹ️  Using file vector search.")
                spinner.start()
                results = searcher.search(query, k, filter_field='filename', filter_value=filename)
                spinner.stop()
                display_results(results, show_full_text)
            else:
                spinner.start()
                llm_k = max(k, 5)  # Use larger k for LLM context
                results = searcher.search_with_llm(query, llm_k, content_type)
                spinner.stop()
                display_llm_response(results)
        else:
            # Vector search only
            spinner.start()
            if section:
                results = searcher.search(query, k, filter_field='section', filter_value=section)
            elif filename:
                results = searcher.search(query, k, filter_field='filename', filter_value=filename)
            else:
                results = searcher.search(query, k, content_type)
            
            spinner.stop()
            display_results(results, show_full_text)
    
    except Exception as e:
        spinner.stop()
        print(f"❌ Search failed: {e}")
        raise

def handle_list_command(searcher, list_type):
    """Handle list commands (types, sections, files)"""
    handlers = {
        'types': ('content_type', "📝 Available content types"),
        'sections': ('section', "📂 Available sections", 20),
        'files': ('filename', "📄 Available files", 10)
    }
    
    if list_type in handlers:
        handler_info = handlers[list_type]
        field = handler_info[0]
        label = handler_info[1]
        limit = handler_info[2] if len(handler_info) > 2 else None
        
        items = searcher.list_available(field)
        
        if limit and len(items) > limit:
            items_str = ', '.join(items[:limit]) + '...'
        else:
            items_str = ', '.join(items)
        
        print(f"{label}: {items_str}")

def main():
    """Main interactive terminal interface"""
    print("🎲 TTRPG AI Search Terminal")
    print("=" * 50)
    
    # Initialize searcher with loading indicator
    init_spinner = LoadingSpinner("Searching")
    init_spinner.start()
    
    try:
        searcher = TTRPGSearcher()
        init_spinner.stop()
    except Exception as e:
        init_spinner.stop()
        print(f"❌ Failed to initialize searcher: {e}")
        return
    
    if not searcher.vector_store:
        print("❌ Vector database not found!")
        print("Run: python src/page_processor.py --all")
        return
    
    # Show initial stats
    stats = searcher.get_stats()
    if not check_error(stats):
        print(f"✅ Loaded {stats['total_vectors']} chunks from {stats['unique_files']} files")
        if stats['llm_available']:
            print("API ready. Ask questions in natural language.")
        else:
            print("⚠️  LLM not available. Using vector search only.")
    
    print("\nType 'help' for commands, 'quit' to exit")
    print("Default: AI-assisted search | Use 'raw' for vector search only")
    
    # Command handlers
    command_handlers = {
        'help': lambda: print(get_help_text(searcher)),
        'stats': lambda: show_stats(searcher),
        'types': lambda: handle_list_command(searcher, 'types'),
        'sections': lambda: handle_list_command(searcher, 'sections'),
        'files': lambda: handle_list_command(searcher, 'files'),
    }
    
    # Main interaction loop
    while True:
        try:
            command = input("\nSearch: ").strip()
            
            if not command:
                continue
            
            command_lower = command.lower()
            
            if command_lower in EXIT_COMMANDS:
                print("Goodbye!")
                break
            
            elif command_lower in command_handlers:
                command_handlers[command_lower]()
            
            else:
                # Parse and execute search
                query, params = parse_search_command(command)
                
                if not query:
                    print("❌ Please provide a search query")
                    continue
                
                execute_search(searcher, query, params)
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 