# CodeFlow: A Cognitive-Load-Optimized System for Semantic Code Analysis and Architectural Governance

## Abstract

This whitepaper describes CodeFlow, a system for semantic code analysis that integrates Tree-sitter parsing, call graph construction, vector-based retrieval, and time-decayed memory to reduce cognitive overhead in complex codebases. We present the architecture, data model, and processing pipeline for CodeFlow’s core capabilities, then extend the discussion to governance-oriented intelligence through drift detection. The drift subsystem leverages metadata-derived feature vectors and call-graph topology analysis to surface structural and dependency anomalies without rule-heavy configuration. We formalize the system’s configuration and execution model, outline evaluation methodology, and summarize limitations and future work. The result is an end-to-end blueprint for integrating retrieval, visualization, memory, and governance in a coherent, developer-aligned platform.

## Keywords

Semantic code analysis, Tree-sitter, call graphs, vector search, architectural drift, memory systems, software governance, cognitive load.

## 1. Introduction

Large codebases impose significant cognitive load on engineers and autonomous agents. Traditional static analysis yields precise diagnostics but often fails to provide high-level contextual navigation, while LLM-centric retrieval can be shallow without structured metadata. CodeFlow addresses this gap by combining accurate AST metadata extraction with call graph modeling and vector-based semantic search. The system is intentionally designed for clarity and predictability: metadata is explicit, graphs are inspectable, and the configuration is centralized.

In addition to retrieval, large organizations require architectural governance. Divergences from established patterns—missing decorators, unexpected dependencies, or inverted layering—are expensive to detect manually. To address this, CodeFlow introduces drift detection: a lightweight, unsupervised mechanism that identifies anomalies without requiring bespoke rules.

This paper presents CodeFlow’s full system design, then details the drift subsystem as a governance layer on top of the core analysis pipeline.

## 2. System Overview

CodeFlow provides three interfaces:

1. **CLI** for batch analysis, reporting, and semantic queries.
2. **MCP Server** for tool-based programmatic access.
3. **Unified API** for language-agnostic extraction and analysis.

The system is anchored by four core components:

- **Tree-sitter extraction** for high-fidelity metadata across Python, TypeScript/TSX, and Rust.
- **Call graph builder** for function-level dependency modeling.
- **Vector store** (ChromaDB) for semantic indexing and retrieval.
- **Cortex memory** for decay-aware knowledge retention.

Drift detection extends these components by combining metadata-derived features with graph-theoretic analysis.

## 3. Architecture and Data Flow

### 3.1 Parsing and Metadata Extraction

Tree-sitter parsers generate language-accurate ASTs. CodeFlow extracts:

- Function signatures, parameters, return types.
- Cyclomatic complexity and NLOC.
- Decorators and access modifiers.
- External dependencies and local variables.
- Exception handling patterns.
- Stable hash of body for change detection.

Reference implementation: [`code_flow/core/treesitter`](code_flow/core/treesitter/__init__.py:1) and element models in [`code_flow/core/models.py`](code_flow/core/models.py:1).

### 3.2 Call Graph Construction

Function elements are normalized into `FunctionNode` objects and connected through `CallEdge` relationships. Resolution is constrained to module-local symbol tables with import-aware matching and confidence scoring to reduce false positives. Entry points are inferred through multiple heuristics (naming, incoming degree, and file patterns). The graph is exportable as JSON or Mermaid.

Reference implementation: [`code_flow/core/call_graph_builder.py`](code_flow/core/call_graph_builder.py:1).

### 3.3 Vector Store and Semantic Search

Function and edge metadata is embedded via SentenceTransformers and stored in ChromaDB. This supports semantic search queries that return ranked functions and their metadata. The store persists across runs, enabling rapid query-only workflows.

Reference implementation: [`code_flow/core/vector_store.py`](code_flow/core/vector_store.py:1).

### 3.4 Cortex Memory

Cortex memory stores higher-level knowledge in a dedicated collection. It distinguishes long-lived tribal knowledge from short-lived episodic facts using decay parameters to prevent stale guidance.

Reference implementation: [`code_flow/core/cortex_memory.py`](code_flow/core/cortex_memory.py:1).

## 4. Configuration Model

CodeFlow centralizes configuration through a Pydantic model. Precedence is: CLI overrides, config file, then defaults. Memory and drift settings are included alongside analysis parameters.

Reference: [`code_flow/core/config.py`](code_flow/core/config.py:38).

Example:

```yaml
project_root: "/path/to/project"
watch_directories: ["."]
embedding_model: "all-MiniLM-L6-v2"
memory_enabled: true
drift_enabled: true
drift_granularity: "module"
```

## 5. Drift Detection Subsystem

Drift detection provides governance signals by identifying structural outliers and dependency violations. It is designed as a warning system rather than a rules engine.

### 5.1 Feature Vectors

Entities (modules or files) are represented as vectors derived from constituent functions. Numeric features include mean and variance of complexity, NLOC, and dependency counts. Categorical counts and textual sets capture modifiers, decorators, and exception patterns.

Model: [`code_flow/core/drift_models.py`](code_flow/core/drift_models.py:1).

### 5.2 Structural Drift

Structural drift is identified by clustering in feature space. The current implementation uses a centroid-based baseline while preserving configuration hooks for HDBSCAN.

Reference: [`code_flow/core/drift_clusterer.py`](code_flow/core/drift_clusterer.py:1).

### 5.3 Topological Drift

Topological drift detects cycles and layer inversions in the call graph. Module-level edges are constructed and analyzed with lightweight cycle detection, cycle deduplication, and layer inference. The drift pipeline can gate edges by confidence thresholds to reduce noisy topology.

Reference: [`code_flow/core/drift_topology.py`](code_flow/core/drift_topology.py:1).

### 5.4 Report Assembly

Findings are aggregated into a JSON-compatible report structure that includes counts, evidence, and configuration metadata.

Reference: [`code_flow/core/drift_report.py`](code_flow/core/drift_report.py:1).

## 6. Interfaces and Execution

### 6.1 CLI

The CLI generates an analysis report and, when enabled, a sibling drift report:

```bash
python -m code_flow.cli.code_flow . --output analysis.json
# produces analysis.json.drift.json
```

Reference: [`code_flow/cli/code_flow.py`](code_flow/cli/code_flow.py:1).

### 6.2 MCP Server

The MCP server exposes drift analysis as `check_drift` for on-demand retrieval from the current analysis state.

Reference: [`code_flow/mcp_server/server.py`](code_flow/mcp_server/server.py:470).

## 7. Evaluation Methodology

Evaluation emphasizes false-positive control and cross-language parity:

1. **Synthetic Drift Injection**: introduce missing decorators or improper dependencies to validate detection.
2. **Baseline Stability**: expect low drift counts on canonical sample projects and rising confidence distribution when gating is applied.
3. **Cross-Language Consistency**: compare results across Python and TypeScript corpora.
4. **Manual Audit**: inspect top-ranked findings to calibrate thresholds.

## 8. Limitations

- The clustering baseline is deterministic and does not yet implement full HDBSCAN.
- Dynamic imports and reflection may reduce call graph fidelity.
- Layer inference is heuristic and assumes mostly acyclic graphs.

## 9. Future Work

- Integrate HDBSCAN and higher-dimensional embeddings for textual features.
- Fuse drift findings with Cortex memory to contextualize exceptions.
- Extend drift signals to structured configuration data.
- Provide Mermaid overlays for drift visualization.

## 10. Conclusion

CodeFlow unifies parsing, graph analysis, retrieval, and memory into a coherent system optimized for cognitive load reduction. Drift detection extends this system with governance signals that are explainable, incremental, and configuration-light. Together, these components establish a practical foundation for scalable, AI-assisted software comprehension and architectural integrity.
