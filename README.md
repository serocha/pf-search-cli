# Pathfinder Search CLI

A standalone command line interface for processing Pathfinder rulebooks (or any cleaned text) and generating vector embeddings for semantic search.

## Overview

This CLI tool processes raw pages and generates embeddings for a FAISS vector database. It's designed for updating or generating new embeddings from your rulebook data, but it does contain search functionality for testing.

### A note on embedding
There are storage, RAM, and compute costs with choosing large vectors, and the results don't necessarily scale.

Switch to 1536-dims for most use cases, or 768-dims for particularly large datasets. Your wallet and server compute will thank you. I used gemini-embedding-exp to generate 3072-dim vectors on two Pathfinder 2E rulebooks, and while I found the results to be quite good, the model was free and my dataset was small.

Edit the config.json file to change the embedding model, LLM, the number of results returned, and more.

<br>

```
cli/
├── data/           # Source pages (page_*.txt files)
├── output/         # Generated embeddings and vector store
├── config.json     # Config settings
├── creds/          # API credentials
├── src/            # Code
└── run.py          # CLI entrypoint wrapper
```

## Setup

### How to Format Rulebooks

This tool processes files to produce embeddings from cleaned, paginated text. The more nicely organized and clean your files are, the easier it will be for the vector database and optional LLM to retrieve the results you want.

Place each rulebook in a separate folder in `/data/`. Folders are considered unique datasets, e.g. `/data/my_rulebook/` will produce outputs attributed to "my_rulebook". Individual page files should be formatted as `page_XXX.txt`, e.g. `page_001.txt`.

### API Key

Processing data requires a Google Gemini API key for generating embeddings. The embedding expects 3072 dimensions by default, which is supported by gemini-embedding-exp. Other APIs may or may not be supported in the future.

Docker users must create the following file, which is imported as a Docker secret:
```bash
mkdir -p creds
echo "GEMINI_API_KEY=your-gemini-api-key" > creds/api.txt
```

Local users must create the GEMINI_API_KEY environment variable manually.

## Running

### Docker Usage

1. Build and run:
```bash
docker-compose up -d
```

2. Use the CLI:
```bash
# Get into the container
docker-compose exec pf-cli bash

# List available datasets
docker-compose exec pf-cli python run.py list

# Process data
docker-compose exec pf-cli python run.py process "Pathfinder 2e Player Core - Clean Pages" 1 10

# Start search terminal (for testing)
docker-compose exec pf-cli python run.py search
```
By default, the container is named _`pf-cli`_.

### Local

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Process your data:
```bash
python run.py list                                    # See available datasets
python run.py process "Dataset Name" 1 10             # Process pages 1-10
python run.py process "Dataset Name"                  # Process all pages
```

3. Test searching (optional):
```bash
python run.py search                                  # Search w/ LLM support or raw vector
```

### Available Commands

- `python run.py list` - List available datasets
- `python run.py process 'Dataset Name' [start] [end]` - Process pages
- `python run.py rebuild` - Rebuild vector store with all JSON files in /output/
- `python run.py search` - Start AI search terminal (for testing)
- `python run.py help` - Show help

### Interactive Search

The search terminal supports (for testing purposes):
- AI-assisted search (default): `How do combat rules work?`
- Vector search only: `raw combat rules`
- Filtered searches: `combat --type=rules --k=5`
- Help: `help`
- Statistics: `stats`

