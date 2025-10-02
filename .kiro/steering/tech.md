---
inclusion: always
---

# CodeFlow Technology Stack

## Build System & Package Management

- **Package Manager**: `uv` (preferred) with `pip` fallback
- **Build System**: setuptools with `pyproject.toml` configuration
- **Python Version**: 3.8+ required

## Core Dependencies

### Analysis & Processing
- **AST Parsing**: Python `ast` module, TypeScript regex-based parsing
- **Vector Store**: ChromaDB (>=1.1.0) for semantic embeddings
- **Embeddings**: sentence-transformers (>=5.1.0), default model: `all-MiniLM-L6-v2`
- **File Watching**: watchdog (>=2.0) for real-time updates

### MCP Server Framework
- **Protocol**: Model Context Protocol (MCP) with FastMCP framework
- **Server**: `mcp[cli]` package for MCP server implementation
- **Configuration**: YAML-based config management with PyYAML

### Testing & Validation
- **Testing**: pytest with pytest-asyncio for async test support
- **Data Validation**: Pydantic for structured data models
- **Mocking**: unittest.mock for test isolation

## Architecture Patterns

### Core Design Principles
- **Cognitive Load Optimization**: Clear, predictable patterns prioritizing human comprehension
- **Unified Interface**: Single API with automatic language detection (Python/TypeScript)
- **Factory Pattern**: `create_extractor()` returns appropriate language-specific extractors
- **Modular Structure**: Separate extractors, builders, and stores with clean interfaces

### Key Components
- **AST Extractors**: Language-specific parsers (`PythonASTExtractor`, `TypeScriptASTExtractor`)
- **Call Graph Builder**: Constructs function relationship graphs with entry point detection
- **Vector Store**: ChromaDB integration with batch operations and cleanup
- **MCP Server**: FastMCP-based server with tool registration and lifecycle management

## Common Commands

### Development Setup
```bash
# Install with uv (recommended)
uv add "mcp[cli]" chromadb sentence-transformers pyyaml watchdog pytest pytest-asyncio pydantic

# Install from source
git clone <repo>
cd codeflow
pip install -e .
```

### CLI Usage
```bash
# Analyze codebase (auto-detects language)
python -m code_flow_graph.cli.code_flow_graph [directory] --output report.json

# Semantic search
python -m code_flow_graph.cli.code_flow_graph [directory] --query "authentication functions"

# Query existing analysis (no re-analysis)
python -m code_flow_graph.cli.code_flow_graph [directory] --no-analyze --query "database operations"

# Generate Mermaid diagrams
python -m code_flow_graph.cli.code_flow_graph [directory] --query "entry points" --mermaid
python -m code_flow_graph.cli.code_flow_graph [directory] --query "api endpoints" --llm-optimized
```

### MCP Server
```bash
# Start MCP server
python -m code_flow_graph.mcp_server

# With custom config
python -m code_flow_graph.mcp_server --config path/to/config.yaml

# Via run script
./run.sh --config codeflow.config.yaml
```

### Testing
```bash
# Run all tests
pytest

# MCP server tests
pytest tests/mcp_server/

# Core functionality tests
pytest tests/core/

# Test with coverage
pytest --cov=code_flow_graph
```

### TypeScript Support
TypeScript analysis is performed using regex-based parsing - no external dependencies required.
All TypeScript features are supported through pattern matching without needing Node.js or TypeScript compiler.

## Configuration Files

- **pyproject.toml**: Package configuration and dependencies
- **codeflow.config.yaml**: MCP server configuration (watch dirs, embedding model, etc.)
- **.kiro/settings/mcp.json**: Workspace-level MCP configuration
- **~/.kiro/settings/mcp.json**: User-level MCP configuration
- **tsconfig.json**: TypeScript project configuration (parsed for project structure)

## Performance Considerations

- **Batch Operations**: Vector store uses batch processing for large codebases
- **Incremental Updates**: File watching enables incremental analysis
- **Persistent Storage**: ChromaDB provides disk-based persistence
- **Memory Management**: Streaming processing for large files
- **Background Cleanup**: Automatic removal of stale references