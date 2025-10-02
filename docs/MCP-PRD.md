# Product Requirements Document (PRD)

## Document Information

- **Product Name:** CodeFlowGraph MCP Server
- **Version:** 1.0 (Revised Draft)
- **Date:** [Insert Current Date]
- **Author:** Sonoma (AI Assistant)
- **Stakeholders:** AI Agent Developers, LLM Integrators, Autonomous Coding Teams
- **Status:** Revised Draft - Focused on MCP Protocol

## Executive Summary

The CodeFlowGraph MCP Server extends the existing CodeFlowGraph analysis engine into a dedicated Model Context Protocol (MCP) server, optimized exclusively for AI agents and LLMs. By leveraging the MCP SDK, the server exposes a suite of specialized tools that allow models to query, traverse, and analyze a codebase's structure and metadata in real-time (via on-demand tool calls). The server maintains a persistent index of the codebase using incremental updates, ensuring agents always have access to current code insights without manual intervention.

This revision prioritizes MCP protocol compliance and agent interaction, treating the server as an "AI tool provider." Human-facing features (e.g., CLI querying, WebSocket notifications) are deprioritized or out of scope, as agents handle their own interaction patterns and notifications. The server focuses on efficient, low-latency tool execution for tasks like semantic search and graph extraction.

**Business Value:**
- Empowers AI agents (e.g., in LangChain, Auto-GPT) with deep, context-aware codebase understanding via standardized MCP tools.
- Reduces agent hallucination on code-related tasks by providing verifiable, structured responses (e.g., Mermaid graphs, metadata).
- Enables seamless integration into agent workflows for code generation, debugging, and refactoring assistance.

**High-Level Deliverables:**
- An MCP-compliant server using the MCP SDK (stdio/SSE transports).
- A set of MCP tools for code analysis (semantic search, call graphs, metadata queries).
- Incremental indexing backend based on existing CodeFlowGraph components.
- Configuration for codebase paths and indexing options.

## Objectives

### Primary Objectives
1. **MCP Protocol Implementation:** Create a minimal, MCP-compliant server that exposes CodeFlowGraph's analysis capabilities as callable tools, allowing AI agents to interact via standard MCP transports (stdio for local, SSE for remote).
2. **Agent-Centric Tooling:** Define tools that return structured, LLM-friendly outputs (e.g., JSON metadata, token-efficient Mermaid graphs) for semantic querying, graph traversal, and metadata extraction.
3. **Persistent, Incremental Indexing:** Automatically maintain an up-to-date codebase index using file watchers and source hashing, ensuring tool responses reflect the latest code state.
4. **Token Efficiency for LLMs:** Optimize tool outputs (e.g., aliased Mermaid graphs) to minimize token usage while preserving essential information.
5. **Cognitive Load Principles:** Ensure tool responses are explicit, linear, and self-describing, aligning with CodeFlowGraph's design philosophy.

### Secondary Objectives
- Reuse existing CodeFlowGraph components (AST extractor, vector store, call graph builder) for zero-rewrite compatibility.
- Support Python codebases initially, with extensibility for other languages.
- Provide configurable indexing (e.g., watch paths, ignore patterns).

## Scope

### In Scope
- MCP server implementation using the MCP SDK (`mcp.server.lowlevel.Server`).
- File monitoring and incremental re-indexing for Python files.
- MCP tools for core analysis functions (detailed below).
- Stdio and SSE transports for agent integration.
- Configuration via YAML/CLI for watch directories and indexing options.
- Logging for tool execution and indexing events.

### Out of Scope (for v1.0)
- Human-facing interfaces (CLI querying, REST API, GUI dashboard).
- Real-time notifications (agents manage their own context updates).
- Multi-language support beyond Python (TypeScript placeholder).
- Advanced security (e.g., authentication; assume local/trusted environments).
- Code execution or dynamic analysis (static AST-based only).
- Integration with specific agent frameworks (but design for compatibility).

## User Stories and Use Cases

### Primary Users
- **AI Agents/LLMs:** Call MCP tools during reasoning workflows to query codebase structure, retrieve metadata, or generate visualizations.
- **Agent Developers:** Integrate the server into custom AI pipelines for code-aware tasks.

### Key User Stories

1. **As an AI Agent, I want to call a semantic search tool so that I can find functions related to a specific concept (e.g., "database migration").**
   - Acceptance: Tool receives query, returns top-N functions with metadata (FQN, complexity, decorators) as `types.TextContent` or JSON.

2. **As an AI Agent, I want to request a call graph subgraph for relevant functions so that I can understand dependencies and flow.**
   - Acceptance: Tool generates Mermaid syntax (LLM-optimized if specified), returns as `types.TextContent` with alias mappings.

3. **As an AI Agent, I want to retrieve detailed metadata for a specific function so that I can analyze its complexity, exceptions, or dependencies.**
   - Acceptance: Tool takes FQN, returns structured JSON (e.g., parameters, NLOC, external deps) as content.

4. **As an AI Agent, I want the server to automatically update its index on file changes so that my tool calls always reflect current code.**
   - Acceptance: Agent calls a tool before/after editing; responses show updated elements without explicit re-indexing.

5. **As an Agent Developer, I want to list available MCP tools so that I can discover and integrate them into my workflow.**
   - Acceptance: `@app.list_tools()` returns schema with tool names, descriptions, and input schemas.

### Use Cases

1. **Code Generation Assistance:** Agent queries "functions handling user sessions," gets results + Mermaid subgraph, then generates code that correctly calls those functions.
2. **Debugging Support:** Agent asks "functions catching `ValueError` in auth module," retrieves metadata, suggests fixes based on local variables and complexity.
3. **Refactoring Planning:** Agent calls "get_call_graph" for a function, analyzes the subgraph to propose safe refactoring steps.
4. **Onboarding/Exploration:** Agent lists tools, calls "query_entry_points," generates overview graph for new codebase.

## Functional Requirements

### 1. Server Core
- **FR-1.1:** Implement using `mcp.server.lowlevel.Server` with stdio (default) and SSE transports (via FastAPI/Starlette as in example).
- **FR-1.2:** Start server via CLI (`python -m code_flow_graph.mcp_server --watch <dir> --transport stdio|sse`).
- **FR-1.3:** On startup, perform initial full indexing (using existing `analyze()` pipeline).
- **FR-1.4:** Use file watchers (`watchdog`) to detect Python file changes (create/modify/delete) in watch directories.
- **FR-1.5:** On change: Incrementally re-extract elements from affected file, update call graph, re-embed in ChromaDB (skip unchanged via `hash_body`).
- **FR-1.6:** Handle deletions: Remove associated elements/edges from ChromaDB and graph.
- **FR-1.7:** Graceful shutdown: Persist state (e.g., last index timestamp).

### 2. MCP Tools
Define tools using `@app.call_tool()` and `@app.list_tools()`, returning `list[types.ContentBlock]` (primarily `types.TextContent` with JSON or Mermaid strings).

- **FR-2.1: `list_tools()` Tool**
  - Returns list of all available tools with schemas (name, title, description, inputSchema).
  - Example Tools:
    - `semantic_search`: Query codebase semantically. Inputs: `query` (str), `n_results` (int, default 5), `filters` (dict, e.g., {"complexity": {"$gt": 10}}). Output: JSON list of functions with metadata.
    - `get_call_graph`: Retrieve subgraph for given FQNs. Inputs: `fqns` (list[str]), `depth` (int, default 1), `format` ("mermaid" or "json"). Output: Mermaid string or JSON edges/nodes.
    - `get_function_metadata`: Detailed info for a function. Inputs: `fqn` (str). Output: JSON with all attributes (complexity, decorators, etc.).
    - `query_entry_points`: List all entry points. Inputs: None. Output: JSON list with metadata.
    - `generate_mermaid_graph`: Full/partial graph visualization. Inputs: `fqns` (list[str], optional), `llm_optimized` (bool, default False). Output: Mermaid string with aliases if optimized.

- **FR-2.2:** Tools must handle errors gracefully (e.g., "Function not found" as `types.TextContent`).
- **FR-2.3:** All outputs use JSON for structured data (e.g., `{"functions": [...], "graph": {...}}`) wrapped in `types.TextContent(text=json.dumps(...))`.

### 3. Indexing Pipeline
- **FR-3.1:** Reuse existing `PythonASTExtractor`, `CallGraphBuilder`, `CodeVectorStore`.
- **FR-3.2:** Incremental mode: On file change, extract only new/changed elements (compare `hash_body`), update graph edges, upsert in ChromaDB.
- **FR-3.3:** Batch updates: Queue multiple changes, process asynchronously to avoid blocking tool calls.
- **FR-3.4:** Logging: Emit events (e.g., "Re-indexed 2 functions from file X").

### 4. Configuration
- **FR-4.1:** YAML config (`config.yaml`): `watch_directories` (list), `ignored_patterns` (list, integrates `.gitignore`), `chromadb_path` (str), `max_graph_depth` (int).
- **FR-4.2:** CLI flags: `--watch <dir>` (multiple), `--transport <stdio|sse>`, `--config <path>`, `--port <int>` (for SSE).

## Non-Functional Requirements

- **Performance:** Tool calls < 2s latency; incremental indexing < 1s per file. Handle 5k+ functions.
- **Scalability:** Support up to 5 concurrent tool calls; queue for indexing.
- **Reliability:** Tools handle missing data gracefully; server restarts from persisted index.
- **Security:** Local-only by default (stdio); SSE with basic origin checks. No code execution.
- **Usability:** Tools have clear descriptions/schemas; outputs are self-contained JSON.
- **Compatibility:** Python 3.8+; MCP SDK v0.x; stdio for local agents, SSE for remote.
- **Token Efficiency:** LLM-optimized graphs < 1k tokens for 30-node subgraphs; use aliases and minimal whitespace.

## Technical Architecture

### High-Level Components
1. **MCP Server (MCP SDK):** Core using `mcp.server.lowlevel.Server`. Define tools with decorators.
2. **File Watcher (watchdog):** Monitors directories for Python changes.
3. **Indexing Queue (asyncio.Queue):** Async processing of file events to update index without blocking tools.
4. **Analysis Backend:** Existing CodeFlowGraph pipeline, adapted for incremental use (e.g., partial re-build of call graph).
5. **Storage:** ChromaDB for vectors/metadata; no additional DB needed.
6. **Transport Layer:** Stdio (local) or SSE (remote) as per MCP SDK example.

### Data Flow
- **Startup:** Load config → Initial full indexing → Start watcher and MCP server.
- **File Change:** Watcher detects → Queue task → Incremental extract/update → Re-embed if changed.
- **Tool Call:** Agent calls MCP tool → Execute analysis (query ChromaDB/graph) → Return `ContentBlock` with JSON/Mermaid.

### Tech Stack
- **MCP Framework:** `mcp` SDK (as in reference example).
- **Server Transport:** Stdio (default); SSE via Starlette/FastAPI (as in example).
- **File Watching:** watchdog.
- **Async Processing:** asyncio (built-in).
- **Dependencies:** Existing (ChromaDB, etc.) + `mcp`, `watchdog`, `starlette` (for SSE).

## Dependencies

### Internal
- CodeFlowGraph core: `ast_extractor.py`, `call_graph_builder.py`, `vector_store.py`.
- Enhanced Mermaid generation with aliasing.

### External
- **mcp:** MCP SDK for protocol implementation.
- **watchdog:** File system monitoring.
- **starlette & uvicorn:** For SSE transport (if enabled).
- **anyio:** Async utilities (as in example).

## Risks and Assumptions

### Risks
- **MCP SDK Changes:** Protocol evolution could require updates. *Mitigation:* Pin SDK version; test against example.
- **Incremental Graph Updates:** Updating call edges on partial changes may be complex. *Mitigation:* Re-build subgraph for affected files.
- **Tool Latency:** Complex queries (e.g., deep graphs) may exceed 2s. *Mitigation:* Async execution; limit depth.
- **Alias Conflicts:** Short aliases may collide in large graphs. *Mitigation:* Robust uniqueness checks.

### Assumptions
- Agents use standard MCP clients (e.g., from SDK) for tool calls.
- Server runs locally or in trusted environments (no auth needed initially).
- File changes are infrequent enough for incremental processing (<100/hour).
- Existing ChromaDB handles concurrent reads during indexing.

## Timeline and Phases

### Phase 1: MCP Server Skeleton (1-2 weeks)
- Implement basic MCP server with stdio transport (based on example).
- Add one simple tool (e.g., `list_tools` and `semantic_search` stub).
- Integrate initial full indexing on startup.

### Phase 2: Core MCP Tools (2-3 weeks)
- Implement all defined tools (`semantic_search`, `get_call_graph`, etc.).
- Add SSE transport support.
- Integrate incremental indexing with file watcher.

### Phase 3: Optimization & Validation (1-2 weeks)
- Add aliasing and LLM-optimized outputs.
- Test tool schemas and responses with MCP clients.
- Edge case handling (e.g., missing functions, large graphs).

### Phase 4: Documentation & Release (1 week)
- Tool schemas and example agent integrations.
- v1.0 release with stdio focus.

**Total Estimated Timeline:** 5-8 weeks.

## Success Metrics

- **Technical:** All tools respond in <2s; 100% MCP schema compliance (validated via SDK tests).
- **Integration:** Successfully integrated into 1-2 agent workflows (e.g., Claude Code); LLM-optimized graphs <800 tokens for typical queries.
- **Agent Performance:** Agents using tools show 20%+ accuracy improvement on code tasks (measured via benchmarks).
- **Coverage:** 90% of existing CodeFlowGraph features exposed as MCP tools.
