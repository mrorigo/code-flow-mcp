---
inclusion: always
---

# CodeFlow Project Structure

## Root Directory Organization

```
codeflow/
├── code_flow_graph/           # Main package
│   ├── core/                  # Core analysis components
│   ├── cli/                   # Command-line interface
│   └── mcp_server/            # MCP server implementation
├── tests/                     # Test suite
├── docs/                      # Documentation
├── examples/                  # Usage examples
├── tmp/                       # Temporary files and debugging
├── code_vectors_chroma/       # ChromaDB vector store (generated)
└── configuration files
```

## Core Package Structure (`code_flow_graph/`)

### Core Analysis (`code_flow_graph/core/`)
- **`__init__.py`**: Unified interface with factory functions and auto-detection
- **`models.py`**: Pure data structures (`CodeElement`, `FunctionElement`, `ClassElement`)
- **`python_extractor.py`**: Python AST parsing and metadata extraction
- **`typescript_extractor.py`**: TypeScript AST parsing and metadata extraction
- **`call_graph_builder.py`**: Function relationship graph construction
- **`vector_store.py`**: ChromaDB integration and semantic search
- **`utils.py`**: Shared utilities (gitignore parsing, file matching)

### CLI Tool (`code_flow_graph/cli/`)
- **`code_flow_graph.py`**: Main CLI orchestrator with `CodeGraphAnalyzer` class

### MCP Server (`code_flow_graph/mcp_server/`)
- **`server.py`**: FastMCP server with tool definitions and lifecycle management
- **`analyzer.py`**: Core analysis logic with file watching capabilities
- **`__main__.py`**: Server entry point
- **`config/default.yaml`**: Default server configuration

## Test Organization (`tests/`)

### Core Tests (`tests/core/`)
- **`python/`**: Python-specific functionality tests
  - `unit/`: Individual component tests
  - `integration/`: End-to-end workflow tests
  - `sample_projects/`: Test fixtures (basic, Django, Flask projects)
  - `fixtures/`: Shared test data and utilities

### TypeScript Tests (`tests/typescript/`)
- **`unit/`**: TypeScript extractor unit tests
- **`integration/`**: TypeScript analysis integration tests
- **`sample_projects/`**: Framework-specific test projects
  - `basic/`: Simple TypeScript project
  - `angular/`: Angular application structure
  - `express/`: Express.js backend
  - `nextjs/`: Next.js application
  - `vue/`: Vue.js application

### MCP Server Tests (`tests/mcp_server/`)
- **`test_server.py`**: Server functionality and tool tests
- **`test_analyzer.py`**: Analysis engine tests
- **`test_setup.py`**: Configuration and initialization tests

## Documentation (`docs/`)

### MCP Documentation
- **`MCP-GUIDE.md`**: Comprehensive MCP implementation guide
- **`MCP-PLAN.md`**: Development roadmap and planning
- **`MCP-PRD.md`**: Product requirements document
- **`MCP-BLUEPRINT.md`**: Technical architecture blueprint

### TypeScript Documentation
- **`TYPESCRIPT_API.md`**: TypeScript analysis API reference
- **`TYPESCRIPT_CLI.md`**: TypeScript CLI usage guide
- **`TYPESCRIPT_EXAMPLES.md`**: TypeScript analysis examples
- **`TYPESCRIPT_TROUBLESHOOTING.md`**: Common issues and solutions
- **`features/TYPESCRIPT.md`**: TypeScript feature documentation

## Configuration Files

### Package Configuration
- **`pyproject.toml`**: Python package metadata, dependencies, and build configuration
- **`uv.lock`**: Dependency lock file for reproducible builds

### Application Configuration
- **`codeflow.config.yaml`**: Default MCP server settings
- **`.roo/mcp.json`**: MCP server registration for development

### Development Configuration
- **`.gitignore`**: Git ignore patterns
- **`.vscode/`**: VS Code workspace settings
- **`.kiro/`**: Kiro IDE configuration and steering rules

## Generated/Runtime Directories

### Vector Store
- **`code_vectors_chroma/`**: ChromaDB persistent storage
  - Contains embeddings, metadata, and indices
  - Project-specific, created per analyzed codebase

### Temporary Files
- **`tmp/`**: Development and debugging artifacts
  - Test scenarios and debugging scripts
  - Temporary analysis outputs

## Naming Conventions

### Python Modules
- **Snake_case**: All Python files and directories (`python_extractor.py`)
- **PascalCase**: Class names (`CodeGraphAnalyzer`, `FunctionElement`)
- **Lowercase**: Package names (`code_flow_graph`)

### Test Files
- **Prefix `test_`**: All test files (`test_server.py`, `test_analyzer.py`)
- **Mirror structure**: Test directory structure mirrors source structure
- **Descriptive names**: Test methods describe functionality (`test_semantic_search_tool_success`)

### Configuration Files
- **Lowercase with extensions**: `pyproject.toml`, `codeflow.config.yaml`
- **Dot-prefixed**: Hidden configuration (`.gitignore`, `.kiro/`)

## Import Patterns

### Unified Interface Usage
```python
# Preferred: Use unified interface
from code_flow_graph.core import extract_from_file, create_extractor

# Language-specific when needed
from code_flow_graph.core import extract_python_file, extract_typescript_file
```

### Internal Module Imports
```python
# Relative imports within package
from .models import FunctionElement, ClassElement
from .utils import get_gitignore_patterns

# Absolute imports for cross-module
from code_flow_graph.core.vector_store import CodeVectorStore
```

## File Organization Principles

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **Language Agnostic Core**: Unified interface hides language-specific complexity
3. **Test Mirroring**: Test structure mirrors source structure for clarity
4. **Configuration Centralization**: All config files at appropriate levels (root, package, user)
5. **Generated Content Isolation**: Runtime/generated files in separate directories
6. **Documentation Co-location**: Feature docs near related code when possible