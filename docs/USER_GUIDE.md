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
    - [Configuration Keys (Core Defaults)](#configuration-keys-core-defaults)
    - [MCP Server Default Config Example](#mcp-server-default-config-example)
    - [Configuration Precedence](#configuration-precedence)
  - [Cortex Memory](#cortex-memory)
    - [Memory Types](#memory-types)
    - [MCP Tools](#mcp-tools)
    - [MCP Resources (Top Memories)](#mcp-resources-top-memories)
    - [CLI Commands](#cli-commands)
    - [Memory Configuration](#memory-configuration)
  - [Semantic Search](#semantic-search)
  - [Call Graphs and Mermaid](#call-graphs-and-mermaid)
  - [Entry Points and Function Metadata](#entry-points-and-function-metadata)
  - [Impact Analysis (MCP)](#impact-analysis-mcp)
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

Key components include:

- Tree-sitter extractors for Python, TypeScript/TSX, and Rust
- Call-graph construction with entry-point identification
- Vector store integration for semantic search
- MCP server tooling and background analysis with file watching
- Cortex memory store for long/short-lived memory entries

## Install

Recommended setup uses `uv`:

```bash
uv sync
```

The project dependencies are defined in [`pyproject.toml`](pyproject.toml). You’ll need Python 3.11+ and a working virtual environment.

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

CLI entry point: `code_flow`.

## Quick Start (MCP Server)

Start the MCP server with defaults:

```bash
code_flow_mcp_server
```

The server loads [`codeflow.config.yaml`](codeflow.config.yaml) by default.

## Configuration

All configuration flows through a single unified model. The default config file is [`codeflow.config.yaml`](codeflow.config.yaml).

### Configuration Keys (Core Defaults)

These defaults apply when a value is not provided in the config file or CLI args:

```yaml
# NOTE: project_root is required for analysis and must be set in your config.
project_root: "/path/to/project"
watch_directories: ["."]
ignored_patterns: ["venv", "**/__pycache__", ".git", ".idea", ".vscode", "node_modules"]
chromadb_path: "./.codeflow/chroma"
max_graph_depth: 3
embedding_model: "all-MiniLM-L6-v2" # Or `jinaai/jina-embeddings-v2-small-en` or `fast`
max_tokens: 256 # Adapt to model
language: "python"
min_similarity: 0.1
call_graph_confidence_threshold: 0.8

# Summary generation (Meta-RAG)
summary_generation_enabled: false
llm_config: {}

# Cortex memory
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

# Cortex memory resources (MCP)
memory_resources_enabled: true
memory_resources_limit: 10
memory_resources_filters: {}

# Drift detection
drift_enabled: false
drift_granularity: "module"
drift_min_entity_size: 3
drift_cluster_algorithm: "hdbscan"
drift_cluster_eps: 0.5
drift_cluster_min_samples: 5
drift_numeric_features: []
drift_textual_features: []
drift_ignore_path_patterns: []
drift_confidence_threshold: 0.6
```

### MCP Server Default Config Example

The MCP server ships with a default config template at [`code_flow/mcp_server/config/default.yaml`](code_flow/mcp_server/config/default.yaml). This is the full example:

```yaml
watch_directories: ["code_flow"]
ignored_patterns: ["venv", "**/__pycache__"]
ignored_filenames: ["package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock"]
chromadb_path: "./code_vectors_chroma"
max_graph_depth: 3
embedding_model: "all-MiniLM-L6-v2"
max_tokens: 256
language: "python"
call_graph_confidence_threshold: 0.8

# Background cleanup configuration
cleanup_interval_minutes: 30

# Cortex memory configuration
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

# Cortex memory resources (MCP)
memory_resources_enabled: true
memory_resources_limit: 10
memory_resources_filters: {}

# Summary Generation (Meta-RAG)
summary_generation_enabled: false
llm_config:
  api_key_env_var: "OPENAI_API_KEY"
  base_url: "https://openrouter.ai/api/v1"
  model: "x-ai/grok-4.1-fast"
  max_tokens: 256
  concurrency: 5
  min_complexity: 3
  min_nloc: 5
  skip_private: true
  skip_test: true
  prioritize_entry_points: true
  summary_depth: "standard"
  max_input_tokens: 2000

# Drift detection configuration
drift_enabled: false
drift_granularity: "module"
drift_min_entity_size: 3
drift_cluster_algorithm: "hdbscan"
drift_numeric_features:
  - complexity_mean
  - complexity_variance
  - nloc_mean
  - nloc_variance
  - decorator_count_mean
  - dependency_count_mean
  - exception_count_mean
  - incoming_degree_mean
  - outgoing_degree_mean
drift_textual_features:
  - decorators
  - external_dependencies
  - catches_exceptions
drift_ignore_path_patterns:
  - "**/tests/**"
  - "**/__pycache__/**"
  - "**/node_modules/**"
drift_confidence_threshold: 0.6
```

### Configuration Precedence

1. CLI args
2. Config file
3. Built-in defaults

## Cortex Memory

Cortex memory stores “tribal” and “episodic” knowledge in a dedicated Chroma collection. It applies decay over time and ranks results with a memory score.

Core implementation is part of the CodeFlow core library.

### Memory Types

- **TRIBAL**: Long-lived conventions and architectural rules.
- **EPISODIC**: Short-lived, fast-decaying facts.
- **FACT**: Medium-lived facts.

### MCP Tools

Available tools:

- `reinforce_memory`
- `query_memory`
- `list_memory`
- `forget_memory`

### MCP Resources (Top Memories)

CodeFlow can expose the top Cortex memories as MCP resources so clients can pull them
as contextual documents. The server registers resources under the `memory://` URI
scheme and provides a summary resource for the current top list.

**Resource URIs:**

- `memory://top` – summary list of top memories
- `memory://<knowledge_id>` – full memory entry

**Configuration:**

```yaml
memory_resources_enabled: true
memory_resources_limit: 10
memory_resources_filters:
  memory_type: TRIBAL
  tags: ["conventions"]
```

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

MCP tool: `semantic_search`.

## Call Graphs and Mermaid

The call graph is derived from Tree-sitter extraction and call graph building. You can retrieve it as JSON or Mermaid.

- MCP tools: `get_call_graph` and `generate_mermaid_graph`.
- CLI option: `query --mermaid` to generate Mermaid for query results.

Call resolution uses module-local symbol tables and import-aware matching with confidence scores. Drift analysis consumes only edges at or above `call_graph_confidence_threshold`.

Metrics are emitted to `.codeflow/reports/` as:

- `call_graph_metrics.json`
- `call_graph_metrics.md`

## Entry Points and Function Metadata

Entry points are derived from call graph analysis and scored for priority.

- MCP tool: `query_entry_points`
  - Supports pagination with `limit` and `offset`.
  - Returns minimal fields by default; set `include_details=true` for full function metadata.
  - Includes `entry_point_score`, `entry_point_category`, `entry_point_priority`, and `entry_point_signals` in each result.
- MCP tool: `get_function_metadata`

## Impact Analysis (MCP)

The MCP server provides `impact_analysis` to compute impacted nodes from a change set using the call graph. It accepts explicit file lists (primary input) and falls back to files modified since the last analysis if no list is provided.

**Tool:** `impact_analysis`

**Inputs:**
- `changed_files` (list of paths; optional)
- `depth` (int, default 2)
- `direction` (`up` | `down` | `both`)
- `include_paths` (bool, default false)

**Example (explicit files):**

```json
{
  "tool": "impact_analysis",
  "input": {
    "changed_files": ["src/services/user_service.py"],
    "depth": 2,
    "direction": "both",
    "include_paths": false
  }
}
```

**Example (watch events since last analysis):**

```json
{
  "tool": "impact_analysis",
  "input": {
    "depth": 2,
    "direction": "up",
    "include_paths": true
  }
}
```

## Structured Data Indexing

YAML/JSON files are indexed and searchable as structured data, enabling configuration-aware search.

Implementation: part of the CodeFlow core library.

## Drift Detection

Drift detection analyzes module-level structure and call-graph topology to surface outliers and layering anomalies.

Core implementation:

- Feature extraction
- Structural clustering
- Topology analysis
- Report assembly

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

Use `check_drift` to generate a drift report from the current analysis state.

## Background Analysis and File Watching

The MCP server runs analysis in the background and watches for file changes using `watchdog`. This enables incremental updates and keeps the index fresh.

Implementation: part of the MCP server background analyzer.

## LLM Summaries (Optional)

If enabled, CodeFlow can generate summaries of functions via an LLM pipeline. Configure in `llm_config` and enable `summary_generation_enabled` in your config.

LLM processing is implemented in the MCP server LLM pipeline.

## Testing

Run unit and integration tests with `pytest`:

```bash
uv run pytest tests/
```

Memory tests live in the test suite under [`tests/`](tests).

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

No. Tree-sitter bindings are bundled and initialized in the extractor modules.

**Q: Where is Cortex memory stored?**

In the same ChromaDB persistence path as code embeddings, but isolated to its own collection.

**Q: How do I disable Cortex memory?**

Set `memory_enabled: false` in your config file.
