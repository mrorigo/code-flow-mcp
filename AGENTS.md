# CodeFlow - Agent Reference Guide

Quick reference for AI agents working with CodeFlow's codebase and tools.

## Overview

CodeFlow is a code analysis tool providing:
- **CLI Tool**: Batch analysis and semantic querying
- **MCP Server**: Real-time analysis with file watching for AI assistants
- **Unified API**: Automatic Python/TypeScript detection and analysis

## Configuration (v0.3.0+)

Both CLI and MCP server use unified configuration via `code_flow/core/config.py`.

**Default config file**: `codeflow.config.yaml` in current directory

**Precedence**: CLI args > Config file > Defaults

### Config File Example
```yaml
watch_directories: ["."]
ignored_patterns: ["venv", "**/__pycache__", ".git", "node_modules"]
chromadb_path: "./code_vectors_chroma"
max_graph_depth: 3
embedding_model: "all-MiniLM-L6-v2"
max_tokens: 256
language: "python"  # or "typescript"
```

### CLI Usage
```bash
# Use default config
code_flow [directory]

# Custom config
code_flow --config custom.yaml

# Override config values
code_flow --language typescript --embedding-model accurate
```

### MCP Server Usage
```bash
# Use default config
code_flow_mcp_server

# Custom config
code_flow_mcp_server --config custom.yaml
```

## Architecture

### Core Components
- **`core/config.py`**: Central configuration with Pydantic models (NEW in v0.3.0)
- **`core/python_extractor.py`**: Python AST extraction
- **`core/typescript_extractor.py`**: TypeScript regex-based extraction
- **`core/structured_extractor.py`**: JSON/YAML file indexing
- **`core/call_graph_builder.py`**: Call graph construction
- **`core/vector_store.py`**: ChromaDB integration for semantic search
- **`core/__init__.py`**: Unified API with auto-detection

### MCP Server
- **`mcp_server/server.py`**: FastMCP server with tool definitions
- **`mcp_server/analyzer.py`**: Background analysis with file watching
- **`mcp_server/llm.py`**: Optional LLM-based summary generation
- **`mcp_server/__main__.py`**: Entry point with config loading

### CLI Tool
- **`cli/code_flow.py`**: Main analyzer orchestrator and CLI entry point

## Key Features

### Background Analysis (MCP Server)
- Server starts immediately, analysis runs in background
- Queries before full analysis may return partial results
- All tool responses include `analysis_status` field
- Use `ping` tool to check analysis progress

### Error Handling
- Vector store errors logged but don't crash; features disabled gracefully
- Source file reading failures logged but continue processing
- Stale file references automatically cleaned up in background

### Structured Data Indexing
- Indexes JSON/YAML configuration files for semantic search
- Flattens nested structures into searchable chunks
- Configurable ignored filenames (e.g., `package-lock.json`)

### Language Detection
- Auto-detects Python/TypeScript from file extensions
- Unified API: `create_extractor()` returns appropriate extractor
- Defaults to Python if language unclear

## MCP Server Tools

1. **`ping`**: Check server status and analysis progress
2. **`semantic_search`**: Natural language code search
3. **`get_call_graph`**: Export call graph (JSON or Mermaid)
4. **`get_function_metadata`**: Detailed function information
5. **`query_entry_points`**: List all entry points
6. **`generate_mermaid_graph`**: Mermaid diagram generation
7. **`cleanup_stale_references`**: Manual cleanup trigger

## Dependencies

- **Python**: 3.11+
- **Core**: ChromaDB >=1.1.0, sentence-transformers >=5.1.0, pydantic, pyyaml
- **MCP**: mcp[cli], watchdog >=2.0, aiohttp >=3.9.0
- **Optional**: openai >=1.0.0 (for LLM summaries), tqdm (CLI progress bars)
- **TypeScript**: No extra dependencies, regex-based parsing

- **Package Manager**: `uv` (recommended)
- **Virtual Environment**: `.venv` (created by `uv`)

See `pyproject.toml` for complete dependency list.

## Development Notes

### Data Models
- Uses dataclasses for models (`FunctionNode`, `CallEdge`, etc.)
- Pydantic for configuration validation (v0.3.0+)
- No methods on data models, pure data structures

### Code Organization
- Unified API in `core/__init__.py` for both languages
- Factory pattern: `create_extractor()` for language detection
- File watching via `watchdog` for MCP server incremental updates
- Exceptions explicitly listed in function metadata

### Vector Store
- Persistent ChromaDB storage in `<directory>/code_vectors_chroma/`
- Embeddings use SentenceTransformers (configurable model)
- Automatic cleanup of stale references (background task)
- Supports both code functions and structured data elements

### Entry Points
- **CLI**: `code_flow` command (via `project.scripts`)
- **MCP Server**: `code_flow_mcp_server` command (via `project.scripts`)
- Both use `code_flow/core/config.py` for configuration (v0.3.0+)

## Version History

- **v0.3.0**: Unified configuration system, central `core/config.py` module
- **v0.2.0**: Structured data indexing, Meta-RAG summaries
- **v0.1.x**: Initial release with Python/TypeScript support