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
   - mcp[cli]

## Run

Start the MCP server:
```bash
python -m code_flow.mcp_server
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

Edit `code_flow/mcp_server/config/default.yaml` to customize:
- Watch directories for file changes
- Vector store settings
- Analysis parameters
- Drift detection settings

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

### check_drift
Run drift detection against the current analyzed codebase.
```json
{
  "input": {},
  "output": {
    "report": {
      "summary": {
        "structural_findings": 1,
        "topological_findings": 0
      }
    }
  }
}
```
Notes:
- Requires `drift_enabled: true` in config.
- Uses the current analysis state; run after initial indexing completes.

### reinforce_memory
Create or reinforce a memory entry in Cortex.
```json
{
  "input": {
    "content": "Use snake_case for database columns",
    "memory_type": "TRIBAL",
    "tags": ["conventions"],
    "scope": "repo"
  }
}
```

### query_memory
Query Cortex memory with decay-aware ranking.
```json
{
  "input": {
    "query": "database naming conventions",
    "n_results": 5,
    "filters": {"memory_type": "TRIBAL"}
  }
}
```

### list_memory
List Cortex memory entries with filters and pagination.
```json
{
  "input": {
    "filters": {"memory_type": "EPISODIC"},
    "limit": 10,
    "offset": 0
  }
}
```

### forget_memory
Delete a memory by id.
```json
{
  "input": {
    "knowledge_id": "<uuid>"
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
