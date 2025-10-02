# Implementation Plan for CodeFlowGraph MCP Server

## Document Information

- **Plan Name:** MCP Server Implementation Plan
- **Version:** 1.0 (Standalone, FastMCP-Based)
- **Date:** [Insert Current Date]
- **Author:** Sonoma (AI Assistant)
- **Branch:** `mcp-server-v1` (New feature branch; no backward compatibility required)
- **Status:** Draft - Ready for Execution

## Introduction

This implementation plan outlines the step-by-step process to build a production-grade Model Context Protocol (MCP) server for CodeFlowGraph, leveraging the existing codebase (AST extractor, call graph builder, and vector store from `core/`). The server will expose CodeFlowGraph's analysis capabilities as MCP-compliant tools for AI clients, enabling semantic querying, graph traversal, and metadata extraction over JSON-RPC 2.0. It will use FastMCP (the MCP SDK) for simplified protocol handling, supporting stdio and HTTP(S)/SSE transports.

The server maintains a persistent index of Python codebases, with incremental updates via file watching. Tools will return structured JSON or Mermaid outputs optimized for LLMs. All 11 MCP features (10 core + elicitation) will be implemented for full compliance with the June 2025 spec, including OAuth2 authentication, streaming responses, and elicitation for multi-turn inputs.

### Key Principles Guiding Implementation
- **MCP Compliance:** Full support for handshake, resource discovery, tool invocation, streaming, state exchange, error handling, OAuth2 auth, version negotiation, logging/metrics/health, shutdown/concurrency, and elicitation. Use FastMCP decorators (@register_tool, @server.on_handshake, etc.) to simplify.
- **Leveraging Existing Codebase:** Reuse `core.ast_extractor.PythonASTExtractor`, `core.call_graph_builder.CallGraphBuilder`, and `core.vector_store.CodeVectorStore` without modifications. Wrap in a new `MCPAnalyzer` for indexing.
- **Simplifications for Speed:** FastMCP handles much of the low-level JSON-RPC (e.g., no manual message parsing). Start with stdio transport; add HTTP/SSE later. Use async for concurrency. Elicitation for tools needing clarification (e.g., ambiguous queries). This reduces boilerplate vs. raw lowlevel.Server, enabling faster market entry (estimated 4-6 weeks total).
- **Cognitive Load Optimization:** Linear structure; explicit dependencies and success criteria per task. Favor explicit code (e.g., clear error messages) over implicit behaviors.
- **Assumptions:**
  - Python 3.8+ environment with existing deps (ChromaDB, sentence-transformers).
  - Local/trusted deployment initially; OAuth2 for enterprise auth.
  - Testing: Pytest for units; MCP SDK client for integration.
  - No code execution; static analysis only.
- **Total Estimated Effort:** 4-6 weeks (simplified by FastMCP), assuming 1-2 developers.
- **Success Metrics:** Tools respond <2s; full MCP compliance (handshake succeeds, tools invoke correctly); indexing handles 5k+ functions; token-efficient outputs (<1k tokens for graphs).

### Repository Structure Changes
- New directory: `codeflowgraph/code_flow_graph/mcp_server/` (server code, analyzer, tools, config).
- Updated: `pyproject.toml` for new deps: `fastmcp`, `watchdog>=2.0`, `pyyaml`, `starlette`, `uvicorn`, `pydantic[email]`, `prometheus-client`, `opentelemetry-api`.
- Entrypoint: `python -m code_flow_graph.mcp_server` (CLI with argparse for config, transport).
- Config: `mcp_server/config/default.yaml` for watch dirs, ignored patterns, ChromaDB path.
- Reuse: Direct imports from `core/` (e.g., `from core.ast_extractor import PythonASTExtractor`).
- Tests: New `tests/mcp_server/` for units/integration.

## High-Level Phases

Phases build incrementally: skeleton first, then features/tools, optimizations, and release. Each includes objectives, dependencies, tasks (sequential, with files/effort/success), and milestones.

### Phase 1: MCP Server Skeleton with Core Protocol (Estimated: 1 week)
**Objectives:** Implement FastMCP base with handshake, version negotiation, resource discovery, basic auth stub, logging/health, and initial indexing. Expose one stub tool.

**Dependencies:**
- External: `fastmcp`, `pyyaml`, `watchdog` (stub), `prometheus-client`.
- Internal: Existing `core/` modules.

**Tasks:**
1. **Create Branch and Setup (0.5 days):**
   - Git: `git checkout -b mcp-server-v1`.
   - Add `mcp_server/__init__.py` (empty); `mcp_server/__main__.py` for entrypoint.
   - Update `pyproject.toml`: Add deps as listed.
   - Create `mcp_server/config/default.yaml`:
     ```yaml
     watch_directories: ["."]
     ignored_patterns: []
     chromadb_path: "./code_vectors_chroma"
     max_graph_depth: 3
     oauth_issuer: "https://auth.example.com"
     ```
   - CLI in `__main__.py`: Use `argparse` for `--config <path> --transport <stdio|http> --port <int> --host <str>`.
   - Success: Branch created; `pip install -e .` installs deps; CLI parses args.

2. **Implement FastMCP Base and Protocol Core (1-2 days):**
   - File: `mcp_server/server.py`.
   - Init: `from fastmcp import MCPServer; server = MCPServer(name="CodeFlowGraphMCP", version="2025.6")`.
   - Handshake (Feature 1/8): `@server.on_handshake async def on_handshake(params):` – Validate client version/capabilities (e.g., reject if != "2025.6"); respond with supported features: `["streaming", "resourceDiscovery", "elicitation", "oauth2"]`.
   - Resource Discovery (Feature 2): `@register_tool` for a stub; implement `listResources` via `@server.list_resources` returning list of tools (name, type="tool", description, input/output schemas via Pydantic).
   - Auth Stub (Feature 7): `@server.auth_middleware async def auth_check(request):` – Check `Authorization: Bearer <token>` header; stub `validate_token(token)` (placeholder JWT/introspection); raise `server.UnauthorizedError` if invalid. Expose `/.well-known/mcp-resource-server` endpoint with JSON (issuer, jwks_uri, resource_indicators_supported=True).
   - Version Negotiation (Feature 8): In handshake, if incompatible, raise `server.ProtocolError`.
   - Logging/Metrics/Health (Feature 9): Use `logging` with correlation IDs; add `/health`, `/readiness` (return {"status": "ok"}), `/metrics` (Prometheus gauge for tool calls/indexed functions).
   - Shutdown/Concurrency (Feature 10): `@server.on_shutdown async def shutdown():` – Close analyzer resources. Use asyncio for concurrency; add timeouts (e.g., 30s per tool).
   - Transports: Stdio default; stub HTTP via `server.run(host, port)`.
   - Success: Run server; simulated client handshake succeeds; `/health` returns OK; logs show trace IDs.

3. **Integrate Initial Full Indexing (1 day):**
   - File: `mcp_server/analyzer.py`.
   - Class `MCPAnalyzer`: Init with config (load YAML); set `project_root`, create `PythonASTExtractor`, `CallGraphBuilder`, `CodeVectorStore`.
   - `async def analyze(self):` – Run extractor on watch dirs (respect ignored patterns via existing gitignore logic), build graph, populate store. Use `hash_body` for dedup.
   - In `server.py`: `@server.on_startup async def startup(): analyzer = MCPAnalyzer(config); await analyzer.analyze(); self.analyzer = analyzer`.
   - Context Exchange Stub (Feature 5): `@server.update_context`, `@server.get_context` – Store session state (e.g., {"user_id": str}) in dict; versioned with timestamp.
   - Success: Startup logs "Indexed X functions"; query store returns data.

4. **Stub Tool and Error Handling (0.5 days):**
   - File: `mcp_server/tools.py` (import in server.py).
   - Stub Tool: `@register_tool("ping", input_model=PingRequest, output_model=PingResponse)` where models are Pydantic BaseModel (e.g., `class PingRequest(BaseModel): message: str`).
   - Handler: `async def ping_tool(req: PingRequest) -> PingResponse: return PingResponse(status="ok", echoed=req.message)`.
   - Errors (Feature 6): Use `server.InvalidParamsError(code=4001, message="...", data={"hint": "..."})` for validation; retryable for transients (e.g., indexing busy: code=499, data={"retry_after": 5}).
   - Update `listResources`: Include "ping" with schemas.
   - Success: Client calls "callTool" on "ping"; returns valid response; invalid input raises typed error.

**Milestones:**
- FastMCP skeleton with handshake, discovery, auth stub, one tool.
- Initial indexing on startup; health/metrics endpoints.
- Test: Pytest for handshake/error; MCP SDK client invokes "ping".

### Phase 2: Full MCP Tools and Advanced Features (Estimated: 1-2 weeks)
**Objectives:** Implement all CodeFlowGraph tools as MCP tools; add streaming, elicitation, stateful context, incremental indexing, full auth.

**Dependencies:**
- External: `starlette`, `uvicorn` for HTTP/SSE; `pyjwt` for OAuth2 stub.
- Internal: Analyzer uses core modules for tools.

**Tasks:**
1. **Implement Core Tools with Schemas (2-3 days):**
   - In `tools.py`: Define Pydantic models for inputs/outputs (e.g., `class SemanticSearchRequest(BaseModel): query: str; n_results: int = 5; filters: dict = {}`).
   - Tool 1: `@register_tool("semantic_search") async def semantic_search(req: SemanticSearchRequest) -> SearchResponse:`
     - Use `self.analyzer.vector_store.query_functions(req.query, req.n_results, req.filters)`; return JSON list of functions (metadata: FQN, complexity, etc.) as `SearchResponse(results: list[dict])`.
   - Tool 2: `@register_tool("get_call_graph")` – Inputs: `fqns: list[str] = [], depth: int = 1, format: str = "mermaid"`. Use `analyzer.graph_builder.export_graph()` or `export_mermaid_graph(fqns, depth)`; return `GraphResponse(graph: str)` (Mermaid/JSON).
   - Tool 3: `@register_tool("get_function_metadata")` – Input: `fqn: str`. Fetch from `analyzer.graph_builder.functions[fqn]`; return `MetadataResponse(complexity: int, decorators: list[dict], ...)` as JSON.
   - Tool 4: `@register_tool("query_entry_points")` – No input; return `EntryPointsResponse(entry_points: list[dict])` from `get_entry_points()`.
   - Tool 5: `@register_tool("generate_mermaid_graph")` – Similar to Tool 2; support `llm_optimized: bool` for aliases/minimal styling.
   - Auth/Scope: In middleware, check token scopes (e.g., "code:read" for tools); reject with 403 if insufficient.
   - Update `listResources`: Dynamically list all tools with schemas, permissions (e.g., ["read"]).
   - Success: Each tool validates inputs; outputs JSON/Mermaid; integrated in discovery.

2. **Add Streaming and Elicitation (1 day):**
   - Streaming (Feature 4): For large graphs (e.g., `generate_mermaid_graph`), declare "streaming" in handshake; use `async for chunk in generate_chunks(): await server.stream_chunk(chunk)`. Handle disconnects with try-except.
   - Elicitation (Feature 11): In tools like "semantic_search", if query ambiguous (e.g., len(query)<3), send `"elicitation/create"` notification with flat schema (e.g., {"query": {"type": "string", "minLength": 3}}). Await client response (`elicitation/accept/reject/cancel`); merge into req if accepted, else error. Use Pydantic for schemas.
   - Backpressure: If indexing busy, raise retryable error.
   - Success: Tool streams Mermaid chunks; elicitation flow tested (mock client responses).

3. **Full OAuth2 Auth and Context (1 day):**
   - Enhance `validate_token`: Use `pyjwt` to verify JWT (audience=resource indicator from token); introspect scopes. Publish discovery endpoint.
   - Context (Feature 5): Store per-session (e.g., {"last_query": str, "user_scopes": list}); `updateContext` merges safely; `getContext` returns versioned dict.
   - Success: Invalid token rejects with 401; valid token with scopes allows tool access.

4. **Implement Incremental Indexing (1 day):**
   - In `analyzer.py`: Use `watchdog` in background task (`@server.on_startup` starts observer).
   - On event (create/modify/delete .py): Async queue process – Extract from file, update graph/store (use `hash_body` to skip unchanged); remove for deletes.
   - Batch/debounce: Queue up to 10; wait 1s.
   - Log: "Updated X functions from file Y".
   - Success: Touch .py file; tool query reflects change without full re-index.

**Milestones:**
- All 5 tools registered/invokable with schemas.
- Streaming/elicitation work; full auth enforces scopes.
- Incremental updates; concurrent calls (asyncio) handled.

### Phase 3: Optimization, Testing, and Production Readiness (Estimated: 1 week)
**Objectives:** Optimize for performance/token efficiency; full testing; add concurrency/shutdown polish.

**Dependencies:** None new.

**Tasks:**
1. **Optimizations and Gotchas (1 day):**
   - Token Efficiency: In Mermaid tools, use aliases (enhance core `_generate_short_alias` if needed); limit nodes (<50).
   - Concurrency (Feature 10): Use `asyncio.gather` for parallel tool calls; pool connections to ChromaDB.
   - Sanitize inputs: Pydantic auto-validates; escape JSON/Mermaid.
   - Tradeoffs: Streaming adds complexity but reduces latency for large outputs; elicitation for UX but optional (fallback to error).
   - Risks: Client disconnects (handle in streams); resource leaks (use `async with` for analyzer).
   - Success: 30-node graph <800 tokens; latency <2s benchmarked.

2. **Testing (2 days):**
   - Units (`tests/mcp_server/test_server.py`): Handshake/version reject; tool validation; auth (valid/invalid tokens); elicitation flows (accept/reject/cancel); errors (codes/messages/hints); context merge.
   - Integration (`tests/mcp_server/test_integration.py`): Mock MCP client (FastMCP client lib); invoke tools, assert JSON outputs; simulate file change/index update; concurrent 5 calls.
   - Edge: Empty index; invalid FQN; overload (backpressure error); shutdown drains requests.
   - Coverage: >80%; include Prometheus metrics test.
   - Gotchas: Test flat schemas only for elicitation; retry logic (e.g., exponential backoff in client mocks).
   - Success: All tests pass; MCP compliance verified.

3. **Deployment Polish (1 day):**
   - Dockerfile: Base on python:3.11-slim; copy code, install deps, expose port 8080.
   - K8s Manifests: Deployment with liveness/readiness probes (/health); env vars for config/secrets.
   - CI: GitHub Actions – Run pytest, build Docker.
   - Config Best Practices: Env vars override YAML (e.g., OAUTH_ISSUER); no hardcodes.
   - Monitoring: OpenTelemetry traces for tool calls.
   - Success: Docker builds/runs; K8s deploys with probes.

**Milestones:**
- Optimizations applied; benchmarks meet metrics.
- Full test suite (units/integration); no leaks.
- Deployment artifacts ready.

### Phase 4: Documentation and Release (Estimated: 0.5-1 week)
**Objectives:** Document features, tools, deployment; release v1.0.

**Tasks:**
1. **Documentation (0.5 days):**
   - `mcp_server/README.md`: Setup (pip install, python -m ...), transports, tool examples (JSON requests/responses), auth (OAuth2 setup), config.
   - Tool Schemas: Export from `listResources` to `tools_schema.json`.
   - Gotchas: Section on disconnects, schema flatness, retry codes.
   - Deployment: Instructions for Docker/K8s; env vars (e.g., TOKEN_VALIDATOR_URL).
   - Success: Docs self-contained; examples runnable.

2. **Release (0.5 days):**
   - Git: PR to main; tag `v1.0-mcp`.
   - Packaging: Ensure subpackage installable.
   - Success: `pip install -e .`; server runs in container.

**Milestones:**
- Docs complete; v1.0 released.

## Potential Challenges and Mitigations
- **Challenge:** FastMCP API changes. *Mitigation:* Pin `fastmcp==latest-stable`; follow examples.
- **Challenge:** OAuth2 integration complexity. *Mitigation:* Stub with pyjwt; real provider (e.g., Auth0) in docs.
- **Challenge:** Streaming backpressure. *Mitigation:* Async queues; client-side handling.
- **Monitoring:** Bi-weekly reviews; adjust for FastMCP simplifications (e.g., reduced Phase 2 effort).

## Next Steps
1. Create branch/setup (immediate).
2. Phase 1 review after 1 week.
3. Track in GitHub issues ("mcp-server").
4. Post-Release: Benchmark with AI clients (e.g., LLM tool calls).
