# CodeFlow: Cognitive Load Optimized Code Analysis Tool

<img src="logo.png"/>

## Overview

CodeFlow is a powerful Python-based code analysis tool designed to help developers and autonomous agents understand complex codebases with minimal cognitive overhead. It generates detailed call graphs, identifies critical code elements, and provides semantic search capabilities, all while adhering to principles that prioritize human comprehension.

By extracting rich metadata from Abstract Syntax Trees (ASTs) and leveraging a persistent vector store (ChromaDB), CodeFlow enables efficient querying and visualization of code structure and behavior.

The tool provides two main interfaces:
- **CLI Tool**: A command-line interface for direct analysis and querying of codebases.
- **MCP Server**: A Model Context Protocol server that integrates with AI assistants and IDEs for real-time code analysis.

## Features

### Core Analysis Capabilities
- **Deep AST Metadata Extraction (Python):** Gathers comprehensive details about functions and classes including:
  - Parameters, return types, docstrings
  - Cyclomatic complexity and Non-Comment Lines of Code (NLOC)
  - Applied decorators (e.g., `@app.route`, `@transactional`)
  - Explicitly caught exceptions
  - Locally declared variables
  - Inferred external library/module dependencies
  - Source body hash for efficient change detection
- **Intelligent Call Graph Generation:**
  - Builds a graph of function-to-function calls.
  - Employs multiple heuristics to identify potential entry points in the codebase.
- **Persistent Vector Store (ChromaDB):**
  - Stores all extracted code elements and call edges as semantic embeddings.
  - Enables rapid semantic search and filtered queries over the codebase's functions and their metadata.
  - Persists analysis results to disk, allowing instant querying of previously analyzed projects without re-parsing.

### Visualization and Output
- **Mermaid Diagram Visualization:**
  - Generates text-based Mermaid Flowchart syntax for call graphs.
  - Highlights functions relevant to a semantic query.
  - Includes an **LLM-optimized mode** for concise, token-efficient graph representations suitable for Large Language Model ingestion, providing clear aliases and FQN mappings.

### MCP Server Features
- **Real-time Analysis:** File watching with incremental updates for dynamic codebases.
- **Tool-based API:** Exposes analysis capabilities through MCP tools for AI assistants.
- **Session Context:** Maintains per-session state for complex analysis workflows.
- **Comprehensive Tools:** Semantic search, call graph generation, function metadata retrieval, entry point identification, and Mermaid graph generation.

### CLI Tool Features
- **Batch Analysis:** Complete codebase analysis with report generation.
- **Interactive Querying:** Semantic search against analyzed codebases.
- **Flexible Output:** JSON reports, Mermaid diagrams, and console output.
- **Incremental Updates:** Query existing analyses without full re-processing.

### Cognitive Load Optimization
- Designed with principles to make the tool's output and its own codebase easy to understand and use.
- **Mental Model Simplicity:** Clear, predictable patterns in code and output.
- **Explicit Behavior:** Favor clarity over brevity, making implicit actions visible (e.g., decorators).
- **Information Hiding & Locality:** Well-defined modules, keeping related code together.
- **Minimal Background Knowledge:** Self-describing data, common patterns, reduced need for memorization.
- **Strategic Abstraction:** Layers introduced only when they genuinely reduce overall complexity.
- **Linear Understanding:** Code and output structured for easy top-to-bottom reading.

## Requirements

Before running CodeFlow, ensure you have Python 3.8+ and the following dependencies installed:

```txt
chromadb
sentence-transformers
fastmcp
pyyaml
watchdog>=2.0
pytest
pytest-asyncio
pydantic
```

## Installation

### From Source
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/codeflow.git
cd codeflow
pip install -e .
```

This will install the package in editable mode and make both the CLI tool and MCP server available.

### CLI Tool
The CLI tool is available as a module:

```bash
python -m code_flow_graph.cli.code_flow_graph --help
```

### MCP Server
The MCP server is available as a script:

```bash
code_flow_graph_mcp_server --help
```

## Usage

### CLI Tool

The `code_flow_graph.cli.code_flow_graph` module is the main entry point for command-line analysis. All commands start with:

`python -m code_flow_graph.cli.code_flow_graph [YOUR_CODE_DIRECTORY]`

Replace `[YOUR_CODE_DIRECTORY]` with the path to your project. If omitted, the current directory (`.`) will be used.

#### 1. Analyze a Codebase and Generate a Report

This command will parse your codebase, build the call graph, populate the ChromaDB vector store (persisted in `<YOUR_CODE_DIRECTORY>/code_vectors_chroma/`), and generate a JSON report.

```bash
python -m code_flow_graph.cli.code_flow_graph [YOUR_CODE_DIRECTORY] --language python --output my_analysis_report.json
```

#### 2. Querying the Codebase (Analysis + Query)

Run a full analysis and then immediately perform a semantic search. This will update the vector store if code has changed.

```bash
python -m code_flow_graph.cli.code_flow_graph [YOUR_CODE_DIRECTORY] --language python --query "functions that handle user authentication"
```

#### 3. Querying an Existing Analysis (Query Only)

Once a codebase has been analyzed (i.e., the `code_vectors_chroma/` directory exists in `[YOUR_CODE_DIRECTORY]`), you can query it much faster without re-running the full analysis:

```bash
python -m code_flow_graph.cli.code_flow_graph [YOUR_CODE_DIRECTORY] --no-analyze --query "functions related to data serialization"
```

#### 4. Generating Mermaid Call Graphs

You can generate Mermaid diagrams of the call graph for functions relevant to your query.

**Standard Mermaid (for visual rendering):**
```bash
python -m code_flow_graph.cli.code_flow_graph [YOUR_CODE_DIRECTORY] --query "database connection pooling" --mermaid
```
The output is Mermaid syntax, which can be copied into a Mermaid viewer (e.g., VS Code extension, Mermaid.live) for visualization.

**LLM-Optimized Mermaid (for AI agents):**
```bash
python -m code_flow_graph.cli.code_flow_graph [YOUR_CODE_DIRECTORY] --query "main entry point setup" --llm-optimized
```
This output is stripped of visual styling and uses short aliases for node IDs, with explicit `%% Alias: ShortID = Fully.Qualified.Name` comments. This minimizes token count for LLMs while providing all necessary structural information.

#### Command Line Arguments

- `<directory>`: (Positional, optional) Path to the codebase directory (default: current directory `.`). This is also the base for the persistent ChromaDB store (`<directory>/code_vectors_chroma/`).
- `--language`: Programming language (`python` or `typescript`, default: `python`).
- `--output`: Output file for the analysis report (default: `code_analysis_report.json`). *Only used during full analysis.*
- `--query <QUESTION>`: Perform a semantic query.
- `--no-analyze`: (Flag) Skips AST extraction and graph building. Requires `--query`. Assumes an existing vector store.
- `--mermaid`: (Flag) Generates a Mermaid graph for query results. Requires `--query`.
- `--llm-optimized`: (Flag) Generates Mermaid graph optimized for LLM token count (removes styling). Implies `--mermaid`.

#### Example Report Output

The `code_analysis_report.json` provides a comprehensive JSON structure including a summary, identified entry points, class summaries, and a detailed call graph (functions with all metadata, and edges).

### MCP Server

The MCP server provides programmatic access to CodeFlow's analysis capabilities through the Model Context Protocol. It can be integrated with AI assistants, IDEs, and other MCP-compatible tools.

#### Starting the Server

Start the MCP server with default configuration:

```bash
python -m code_flow_graph.mcp_server
```

Or with a custom configuration file:

```bash
python -m code_flow_graph.mcp_server --config path/to/config.yaml
```

#### Available Tools

The server exposes the following tools through the MCP protocol:

- **`ping`**: Test server connectivity
- **`semantic_search`**: Search functions semantically using natural language queries
- **`get_call_graph`**: Retrieve call graph in JSON or Mermaid format
- **`get_function_metadata`**: Get detailed metadata for a specific function
- **`query_entry_points`**: Get all identified entry points in the codebase
- **`generate_mermaid_graph`**: Generate Mermaid diagram for call graph visualization
- **`update_context`**: Update session context with key-value pairs
- **`get_context`**: Retrieve current session context

#### Testing with Client

Use the included client to test server functionality:

```bash
python client.py
```

This performs a handshake and tests basic tool functionality.

## Configuration

### MCP Server Configuration

The MCP server uses a YAML configuration file (default: `code_flow_graph/mcp_server/config/default.yaml`):

```yaml
watch_directories: ["code_flow_graph"]  # Directories to monitor for changes
ignored_patterns: ["venv", "**/__pycache__"]  # Patterns to ignore during analysis
chromadb_path: "./code_vectors_chroma"  # Path to ChromaDB vector store
max_graph_depth: 3  # Maximum depth for graph traversal
```

Customize these settings by creating your own config file and passing it with `--config`.

## Examples

### CLI Tool Examples

#### Basic Analysis
```bash
# Analyze current directory and generate report
python -m code_flow_graph.cli.code_flow_graph . --output analysis.json

# Analyze a specific project
python -m code_flow_graph.cli.code_flow_graph /path/to/my/project --language python
```

#### Semantic Search
```bash
# Find authentication functions
python -m code_flow_graph.cli.code_flow_graph . --query "user authentication login"

# Search for database operations
python -m code_flow_graph.cli.code_flow_graph . --query "database queries CRUD operations"
```

#### Visualization
```bash
# Generate Mermaid diagram for API endpoints
python -m code_flow_graph.cli.code_flow_graph . --query "API endpoints" --mermaid

# LLM-optimized graph for AI analysis
python -m code_flow_graph.cli.code_flow_graph . --query "error handling" --llm-optimized
```

### MCP Server Examples

#### Semantic Search
```json
{
  "tool": "semantic_search",
  "input": {
    "query": "functions that handle user authentication",
    "n_results": 5,
    "filters": {}
  }
}
```

#### Get Function Metadata
```json
{
  "tool": "get_function_metadata",
  "input": {
    "fqn": "myapp.auth.authenticate_user"
  }
}
```

#### Generate Call Graph
```json
{
  "tool": "get_call_graph",
  "input": {
    "fqns": ["myapp.main"],
    "format": "mermaid"
  }
}
```

#### Update Context
```json
{
  "tool": "update_context",
  "input": {
    "current_focus": "authentication_module",
    "analysis_depth": "detailed"
  }
}
```

## Testing

### MCP Server Tests
Run the MCP server test suite:

```bash
pytest tests/mcp_server/
```

This includes tests for:
- Server initialization and tool registration
- Tool functionality (semantic search, call graphs, etc.)
- Configuration loading
- File watching and incremental updates

### CLI Tool Testing
Test the CLI tool by running analysis on the test files:

```bash
# Test basic functionality
python -m code_flow_graph.cli.code_flow_graph tests/ --output test_report.json

# Test querying
python -m code_flow_graph.cli.code_flow_graph tests/ --query "test functions"
```

### Integration Testing
Use the client script for end-to-end testing:

```bash
python client.py
```

This tests the MCP protocol handshake and basic tool interactions.

## Architecture

The tool is structured into three main components, designed for clarity and maintainability:

### Core Components
1. **AST Extractor** (`core/ast_extractor.py`)
   - Parses source code into Abstract Syntax Trees.
   - Extracts rich metadata for `FunctionElement` and `ClassElement` objects (complexity, decorators, dependencies, etc.).
   - Filters files based on `.gitignore` for relevant analysis.

2. **Call Graph Builder** (`core/call_graph_builder.py`)
   - Constructs a directed graph of function calls based on extracted AST data.
   - Identifies application entry points using multiple heuristics.
   - Provides structured `FunctionNode` and `CallEdge` objects, containing the rich metadata.

3. **Vector Store** (`core/vector_store.py`)
   - Integrates with [ChromaDB](https://www.trychroma.com/) for a persistent, queryable knowledge base.
   - Stores semantic embeddings of functions and edges, along with their detailed metadata.
   - Enables semantic search (`query_functions`) and efficient updates via source code hashing.

### MCP Server Architecture
- **Server** (`mcp_server/server.py`): FastMCP-based server handling MCP protocol and tool registration.
- **Analyzer** (`mcp_server/analyzer.py`): Core analysis logic with file watching for incremental updates.
- **Tools** (`mcp_server/tools.py`): MCP tool implementations with request/response models.
- **Configuration** (`mcp_server/config/`): YAML-based configuration management.

### CLI Tool Architecture
- **CodeGraphAnalyzer** (`cli/code_flow_graph.py`): Main orchestrator for analysis pipeline.
- Command-line argument parsing and output formatting.
- Integration with core components for analysis and querying.

## Contributing

We welcome contributions! Please refer to the [Contributing Guide](CONTRIBUTING.md) (or similar if you create one) for details on how to get involved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- Enhanced TypeScript parsing and feature parity with Python.
- Advanced data flow analysis (beyond simple local variables).
- Integration with other visualization tools (e.g., Graphviz).
- More sophisticated entry point detection for various frameworks.
- Direct IDE integrations for real-time analysis and navigation.
- Support for other programming languages.
- Web-based UI for interactive code exploration.
- Plugin system for custom analysis rules.

## Acknowledgments

This project is built upon the excellent work of:
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Python AST module](https://docs.python.org/3/library/ast.html)
- [Mermaid.js](https://mermaid.js.org/) for diagramming.
- [FastMCP](https://github.com/jlowin/fastmcp) for MCP server framework.
- [Watchdog](https://github.com/gorakhargosh/watchdog) for file monitoring.
