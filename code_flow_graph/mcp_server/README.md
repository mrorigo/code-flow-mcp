# CodeFlowGraph MCP Server

A Model Context Protocol (MCP) server that provides semantic search and analysis capabilities for codebases using vector embeddings and call graph analysis.

## Setup

1. Install dependencies:
```bash
pip install -e .
```

2. Ensure you have the required dependencies:
   - chromadb
   - sentence-transformers
   - fastmcp

## Run

Start the MCP server:
```bash
python -m code_flow_graph.mcp_server
```

## Test

Run the test suite:
```bash
pytest tests/mcp_server/
```

Use the client for handshake and tool testing:
```bash
python client.py
```

## Config

Edit `code_flow_graph/mcp_server/config/default.yaml` to customize:
- Watch directories for file changes
- Vector store settings
- Analysis parameters

## Tools

The server provides the following tools:

### ping
Echo a message back.
```json
{
  "input": {
    "message": "hello"
  },
  "output": {
    "status": "ok",
    "echoed": "hello"
  }
}
```

### semantic_search
Search functions semantically using natural language queries.
```json
{
  "input": {
    "query": "functions that handle user authentication",
    "n_results": 5,
    "filters": {}
  },
  "output": {
    "results": [
      {
        "metadata": {
          "name": "authenticate_user",
          "file_path": "/path/to/auth.py"
        }
      }
    ]
  }
}
```

### get_call_graph
Get call graph in JSON or Mermaid format.
```json
{
  "input": {
    "fqns": ["module.function"],
    "format": "json"
  },
  "output": {
    "graph": {
      "functions": {
        "module.function": {
          "name": "function"
        }
      }
    }
  }
}
```

### get_function_metadata
Get detailed metadata for a specific function.
```json
{
  "input": {
    "fqn": "module.function"
  },
  "output": {
    "name": "function",
    "file_path": "/path/to/file.py",
    "complexity": 5,
    "parameters": ["arg1", "arg2"]
  }
}
```

### query_entry_points
Get all identified entry points in the codebase.
```json
{
  "input": {},
  "output": {
    "entry_points": [
      {
        "name": "main",
        "file_path": "/path/to/main.py"
      }
    ]
  }
}
```

### generate_mermaid_graph
Generate Mermaid diagram for call graph visualization.
```json
{
  "input": {
    "fqns": ["module.function"],
    "llm_optimized": true
  },
  "output": {
    "graph": "graph TD\nA --> B"
  }
}
```

### update_context
Update session context with key-value pairs.
```json
{
  "input": {
    "key": "value"
  },
  "output": {
    "version": 1
  }
}
```

### get_context
Retrieve current session context.
```json
{
  "input": {},
  "output": {
    "key": "value"
  }
}