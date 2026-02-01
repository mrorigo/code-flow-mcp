# CodeFlow User Guide

This guide explains how to install, configure, and use CodeFlow’s CLI and MCP server, including Cortex memory capabilities.

## Table of Contents

- [CodeFlow User Guide](#codeflow-user-guide)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Install](#install)
  - [Quick Start (CLI)](#quick-start-cli)
  - [Quick Start (MCP Server)](#quick-start-mcp-server)
  - [Configuration](#configuration)
    - [Configuration Precedence](#configuration-precedence)
  - [Cortex Memory](#cortex-memory)
    - [Memory Types](#memory-types)
    - [MCP Tools](#mcp-tools)
    - [CLI Commands](#cli-commands)
    - [Memory Configuration](#memory-configuration)
  - [Semantic Search](#semantic-search)
  - [Call Graphs and Mermaid](#call-graphs-and-mermaid)
  - [Entry Points and Function Metadata](#entry-points-and-function-metadata)
  - [Structured Data Indexing](#structured-data-indexing)
  - [Background Analysis and File Watching](#background-analysis-and-file-watching)
  - [LLM Summaries (Optional)](#llm-summaries-optional)
  - [Testing](#testing)
  - [Troubleshooting](#troubleshooting)
    - [Vector store not initialized](#vector-store-not-initialized)
    - [Empty results during MCP startup](#empty-results-during-mcp-startup)
    - [Embedding dimension mismatch](#embedding-dimension-mismatch)
  - [FAQ](#faq)

## Overview

CodeFlow is a code analysis tool that builds call graphs, extracts metadata using Tree-sitter, and indexes the results into a vector store for semantic search. It exposes:

- A **CLI** for analysis, reporting, and querying.
- An **MCP server** for programmatic access (tools for search, call graphs, metadata, and memory).

Key components live in:

- Tree-sitter extractors in [`code_flow_graph/core/treesitter`](code_flow_graph/core/treesitter/__init__.py:1)
- Vector store integration in [`code_flow_graph/core/vector_store.py`](code_flow_graph/core/vector_store.py:1)
- MCP server tooling in [`code_flow_graph/mcp_server/server.py`](code_flow_graph/mcp_server/server.py:1)
- MCP analyzer and file watching in [`code_flow_graph/mcp_server/analyzer.py`](code_flow_graph/mcp_server/analyzer.py:1)
- Cortex memory store in [`code_flow_graph/core/cortex_memory.py`](code_flow_graph/core/cortex_memory.py:1)

## Install

Recommended setup uses `uv`:

```bash
uv sync
```

The project dependencies are defined in [`pyproject.toml`](pyproject.toml:1). You’ll need Python 3.11+ and a working virtual environment.

## Quick Start (CLI)

Analyze a codebase and generate a report:

```bash
python -m code_flow_graph.cli.code_flow_graph .
```

Run a semantic query after analysis:

```bash
python -m code_flow_graph.cli.code_flow_graph . --query "authentication flows" --mermaid
```

Useful flags include:

- `--language` to force Python/TypeScript/Rust.
- `--embedding-model` to pick an embedding model.
- `--max-tokens` to tune chunk size for embeddings.

CLI entry point: [`code_flow_graph/cli/code_flow_graph.py`](code_flow_graph/cli/code_flow_graph.py:1).

## Quick Start (MCP Server)

Start the MCP server with defaults:

```bash
python -m code_flow_graph.mcp_server
```

The server uses the unified config model in [`code_flow_graph/core/config.py`](code_flow_graph/core/config.py:1) and loads `codeflow.config.yaml` by default.

Tool definitions are in [`code_flow_graph/mcp_server/server.py`](code_flow_graph/mcp_server/server.py:1).

## Configuration

All configuration flows through the Pydantic model in [`code_flow_graph/core/config.py`](code_flow_graph/core/config.py:1). The default config file is `codeflow.config.yaml`.

Minimal example:

```yaml
watch_directories: ["."]
ignored_patterns: ["venv", "**/__pycache__", ".git", "node_modules"]
chromadb_path: "./code_vectors_chroma"
max_graph_depth: 3
embedding_model: "all-MiniLM-L6-v2"
max_tokens: 256
language: "python"
```

### Configuration Precedence

1. CLI args
2. Config file
3. Defaults in [`code_flow_graph/core/config.py`](code_flow_graph/core/config.py:1)

## Cortex Memory

Cortex memory stores “tribal” and “episodic” knowledge in a dedicated Chroma collection. It applies decay over time and ranks results with a memory score.

Core implementation: [`code_flow_graph/core/cortex_memory.py`](code_flow_graph/core/cortex_memory.py:1).

### Memory Types

- **TRIBAL**: Long-lived conventions and architectural rules.
- **EPISODIC**: Short-lived, fast-decaying facts.
- **FACT**: Medium-lived facts.

### MCP Tools

Available tools (defined in [`code_flow_graph/mcp_server/server.py`](code_flow_graph/mcp_server/server.py:78)):

- `reinforce_memory`
- `query_memory`
- `list_memory`
- `forget_memory`

### CLI Commands

```bash
# Add memory
python -m code_flow_graph.cli.code_flow_graph memory add \
  --type TRIBAL \
  --content "Use snake_case for DB columns" \
  --tags conventions

# Query memory
python -m code_flow_graph.cli.code_flow_graph memory query \
  --query "DB column naming" \
  --type TRIBAL \
  --limit 5

# Reinforce
python -m code_flow_graph.cli.code_flow_graph memory reinforce --knowledge-id <uuid>

# Delete
python -m code_flow_graph.cli.code_flow_graph memory forget --knowledge-id <uuid>
```

### Memory Configuration

```yaml
memory_enabled: true
memory_collection_name: "cortex_memory_v1"
memory_similarity_weight: 0.7
memory_score_weight: 0.3
memory_min_score: 0.1
memory_cleanup_interval_seconds: 3600
memory_grace_seconds: 86400
memory_half_life_days:
  TRIBAL: 180.0
  EPISODIC: 7.0
  FACT: 30.0
memory_decay_floor:
  TRIBAL: 0.1
  EPISODIC: 0.01
  FACT: 0.05
```

## Semantic Search

Semantic search uses embeddings from SentenceTransformers in the vector store. Queries return code/document metadata and ranked results.

CLI usage:

```bash
python -m code_flow_graph.cli.code_flow_graph . --query "JWT auth" --mermaid
```

MCP tool: `semantic_search` in [`code_flow_graph/mcp_server/server.py`](code_flow_graph/mcp_server/server.py:218).

## Call Graphs and Mermaid

The call graph is derived from Tree-sitter extraction and call graph building. You can retrieve it as JSON or Mermaid.

- MCP tool: `get_call_graph` and `generate_mermaid_graph` in [`code_flow_graph/mcp_server/server.py`](code_flow_graph/mcp_server/server.py:261).
- CLI option: `--mermaid` to generate Mermaid for query results.

## Entry Points and Function Metadata

Entry points are derived from call graph analysis.

- MCP tool: `query_entry_points` in [`code_flow_graph/mcp_server/server.py`](code_flow_graph/mcp_server/server.py:304)
- MCP tool: `get_function_metadata` in [`code_flow_graph/mcp_server/server.py`](code_flow_graph/mcp_server/server.py:276)

## Structured Data Indexing

YAML/JSON files are indexed and searchable as structured data, enabling configuration-aware search.

Implementation: [`code_flow_graph/core/structured_extractor.py`](code_flow_graph/core/structured_extractor.py:1).

## Background Analysis and File Watching

The MCP server runs analysis in the background and watches for file changes using `watchdog`. This enables incremental updates and keeps the index fresh.

Implementation: [`code_flow_graph/mcp_server/analyzer.py`](code_flow_graph/mcp_server/analyzer.py:1).

## LLM Summaries (Optional)

If enabled, CodeFlow can generate summaries of functions via an LLM pipeline. Configure in `llm_config` and enable `summary_generation_enabled` in your config.

LLM processing is implemented in [`code_flow_graph/mcp_server/llm.py`](code_flow_graph/mcp_server/llm.py:1).

## Testing

Run unit and integration tests with `pytest`:

```bash
uv run pytest tests/
```

Memory tests live in [`tests/core/test_cortex_memory.py`](tests/core/test_cortex_memory.py:1).

## Troubleshooting

### Vector store not initialized

If you see messages about the vector store being unavailable, verify:

- `chromadb_path` exists or is writeable.
- Embedding model dependencies are installed.
- Configuration points to the correct directory.

### Empty results during MCP startup

The MCP server returns partial or empty results while the background analysis runs. Use the `ping` tool to check `analysis_status`.

### Embedding dimension mismatch

If you switch embedding models across runs, re-run analysis to rebuild embeddings with a consistent model.

## FAQ

**Q: Do I need Tree-sitter manually installed?**

No. Tree-sitter bindings are bundled and initialized in the extractor modules under [`code_flow_graph/core/treesitter`](code_flow_graph/core/treesitter/__init__.py:1).

**Q: Where is Cortex memory stored?**

In the same ChromaDB persistence path as code embeddings, but isolated to its own collection. See [`code_flow_graph/core/cortex_memory.py`](code_flow_graph/core/cortex_memory.py:1).

**Q: How do I disable Cortex memory?**

Set `memory_enabled: false` in your config file. See defaults in [`code_flow_graph/core/config.py`](code_flow_graph/core/config.py:1).
