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
  - [Drift Detection](#drift-detection)
    - [Enabling Drift](#enabling-drift)
    - [CLI Output](#cli-output)
    - [MCP Tool](#mcp-tool)
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

- Tree-sitter extractors in [`../code_flow/core/treesitter`](code_flow/core/treesitter/__init__.py:1)
- Vector store integration in [`../code_flow/core/vector_store.py`](code_flow/core/vector_store.py:1)
- MCP server tooling in [`../code_flow/mcp_server/server.py`](code_flow/mcp_server/server.py:1)
- MCP analyzer and file watching in [`../code_flow/mcp_server/analyzer.py`](code_flow/mcp_server/analyzer.py:1)
- Cortex memory store in [`../code_flow/core/cortex_memory.py`](code_flow/core/cortex_memory.py:1)

## Install

Recommended setup uses `uv`:

```bash
uv sync
```

The project dependencies are defined in [`pyproject.toml`](pyproject.toml:1). You’ll need Python 3.11+ and a working virtual environment.

Install CodeFlow as a user-level tool (recommended for CLI usage):

```bash
uv tool install code-flow
```

Expose uv's tool bin directory on your PATH:

```bash
export PATH="$(uv tool dir --bin):$PATH"
```

Persist the PATH update in your shell profile (e.g., `~/.zshrc`).

One-off execution without installation:

```bash
uvx --from code-flow code_flow --help
```

## Quick Start (CLI)

Analyze a codebase and generate a report:

```bash
code_flow analyze -- .
```

Run a semantic query after analysis:

```bash
code_flow query -- . --query "authentication flows" --mermaid
```

Useful flags include:

- `--language` to force Python/TypeScript/Rust.
- `--embedding-model` to pick an embedding model.
- `--max-tokens` to tune chunk size for embeddings.

CLI entry point: [`../code_flow/cli/code_flow.py`](code_flow/cli/code_flow.py:1).

## Quick Start (MCP Server)

Start the MCP server with defaults:

```bash
code_flow_mcp_server
```

The server uses the unified config model in [`../code_flow/core/config.py`](code_flow/core/config.py:1) and loads `codeflow.config.yaml` by default.

Tool definitions are in [`../code_flow/mcp_server/server.py`](code_flow/mcp_server/server.py:1).

## Configuration

All configuration flows through the Pydantic model in [`../code_flow/core/config.py`](code_flow/core/config.py:1). The default config file is `codeflow.config.yaml`.

Minimal example:

```yaml
project_root: "/path/to/project"
watch_directories: ["."]
  ignored_patterns: ["venv", "**/__pycache__", ".git", "node_modules"]
  max_graph_depth: 3
  embedding_model: "all-MiniLM-L6-v2"
  max_tokens: 256
  language: "python"
  call_graph_confidence_threshold: 0.8
```

### Configuration Precedence

1. CLI args
2. Config file
3. Defaults in [`../code_flow/core/config.py`](code_flow/core/config.py:1)

## Cortex Memory

Cortex memory stores “tribal” and “episodic” knowledge in a dedicated Chroma collection. It applies decay over time and ranks results with a memory score.

Core implementation: [`../code_flow/core/cortex_memory.py`](code_flow/core/cortex_memory.py:1).

### Memory Types

- **TRIBAL**: Long-lived conventions and architectural rules.
- **EPISODIC**: Short-lived, fast-decaying facts.
- **FACT**: Medium-lived facts.

### MCP Tools

Available tools (defined in [`../code_flow/mcp_server/server.py`](code_flow/mcp_server/server.py:78)):

- `reinforce_memory`
- `query_memory`
- `list_memory`
- `forget_memory`

### CLI Commands

```bash
# Add memory
code_flow memory add \
  --type TRIBAL \
  --content "Use snake_case for DB columns" \
  --tags conventions

# Query memory
code_flow memory query \
  --query "DB column naming" \
  --type TRIBAL \
  --limit 5

# Reinforce
code_flow memory reinforce --knowledge-id <uuid>

# Delete
code_flow memory forget --knowledge-id <uuid>
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
code_flow query -- . --query "JWT auth" --mermaid --min-similarity 0.5
```

MCP tool: `semantic_search` in [`../code_flow/mcp_server/server.py`](code_flow/mcp_server/server.py:218).

## Call Graphs and Mermaid

The call graph is derived from Tree-sitter extraction and call graph building. You can retrieve it as JSON or Mermaid.

- MCP tool: `get_call_graph` and `generate_mermaid_graph` in [`../code_flow/mcp_server/server.py`](code_flow/mcp_server/server.py:261).
- CLI option: `query --mermaid` to generate Mermaid for query results.

Call resolution uses module-local symbol tables and import-aware matching with confidence scores. Drift analysis consumes only edges at or above `call_graph_confidence_threshold`.

Metrics are emitted to `.codeflow/reports/` as:

- `call_graph_metrics.json`
- `call_graph_metrics.md`

## Entry Points and Function Metadata

Entry points are derived from call graph analysis.

- MCP tool: `query_entry_points` in [`../code_flow/mcp_server/server.py`](code_flow/mcp_server/server.py:304)
- MCP tool: `get_function_metadata` in [`../code_flow/mcp_server/server.py`](code_flow/mcp_server/server.py:276)

## Structured Data Indexing

YAML/JSON files are indexed and searchable as structured data, enabling configuration-aware search.

Implementation: [`../code_flow/core/structured_extractor.py`](code_flow/core/structured_extractor.py:1).

## Drift Detection

Drift detection analyzes module-level structure and call-graph topology to surface outliers and layering anomalies.

Core implementation:

- Feature extraction: [`../code_flow/core/drift_features.py`](code_flow/core/drift_features.py:1)
- Structural clustering: [`../code_flow/core/drift_clusterer.py`](code_flow/core/drift_clusterer.py:1)
- Topology analysis: [`../code_flow/core/drift_topology.py`](code_flow/core/drift_topology.py:1)
- Report assembly: [`../code_flow/core/drift_report.py`](code_flow/core/drift_report.py:1)

### Enabling Drift

Set drift configuration in `codeflow.config.yaml`:

```yaml
drift_enabled: true
drift_granularity: "module"  # module | file
drift_min_entity_size: 3
drift_cluster_algorithm: "hdbscan"
drift_confidence_threshold: 0.6
call_graph_confidence_threshold: 0.8
```

### CLI Output

When enabled, drift analysis writes a sibling report next to the main analysis output:

```bash
code_flow analyze -- . --output analysis.json
# writes: analysis.json.drift.json
```

### MCP Tool

Use `check_drift` in [`../code_flow/mcp_server/server.py`](code_flow/mcp_server/server.py:470) to generate a drift report from the current analysis state.

## Background Analysis and File Watching

The MCP server runs analysis in the background and watches for file changes using `watchdog`. This enables incremental updates and keeps the index fresh.

Implementation: [`../code_flow/mcp_server/analyzer.py`](code_flow/mcp_server/analyzer.py:1).

## LLM Summaries (Optional)

If enabled, CodeFlow can generate summaries of functions via an LLM pipeline. Configure in `llm_config` and enable `summary_generation_enabled` in your config.

LLM processing is implemented in [`../code_flow/mcp_server/llm.py`](code_flow/mcp_server/llm.py:1).

## Testing

Run unit and integration tests with `pytest`:

```bash
uv run pytest tests/
```

Memory tests live in [`tests/core/test_cortex_memory.py`](tests/core/test_cortex_memory.py:1).

## Troubleshooting

### Vector store not initialized

If you see messages about the vector store being unavailable, verify:

- `<project_root>/.codeflow/chroma` exists or is writeable.
- Embedding model dependencies are installed.
- Configuration points to the correct directory.

### Empty results during MCP startup

The MCP server returns partial or empty results while the background analysis runs. Use the `ping` tool to check `analysis_status`.

### Embedding dimension mismatch

If you switch embedding models across runs, re-run analysis to rebuild embeddings with a consistent model.

## FAQ

**Q: Do I need Tree-sitter manually installed?**

No. Tree-sitter bindings are bundled and initialized in the extractor modules under [`../code_flow/core/treesitter`](code_flow/core/treesitter/__init__.py:1).

**Q: Where is Cortex memory stored?**

In the same ChromaDB persistence path as code embeddings, but isolated to its own collection. See [`../code_flow/core/cortex_memory.py`](code_flow/core/cortex_memory.py:1).

**Q: How do I disable Cortex memory?**

Set `memory_enabled: false` in your config file. See defaults in [`../code_flow/core/config.py`](code_flow/core/config.py:1).
