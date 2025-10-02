# Blueprint for Building the CodeFlowGraph MCP Server

This blueprint provides a detailed, linear roadmap for implementing the MCP server, focusing on stdio transport for local testing. It leverages the existing codebase (`core/ast_extractor.py`, `core/call_graph_builder.py`, `core/vector_store.py`) without modifications, exposing analysis tools via FastMCP over JSON-RPC 2.0. The core feature set includes handshake, resource discovery, tool invocation, basic error handling, simple state exchange, logging, and shutdown. Development occurs in the `mcp-server-v1` branch, emphasizing quick iteration to a locally testable server.

### Overall Guidelines
- **Environment Setup**: Python 3.11+; virtualenv; `pip install -e .`. Assume existing repo.
- **Transport Focus**: Stdio only (local, trusted; no auth/HTTP). Run via `python -m code_flow_graph.mcp_server` for pipe-based testing.
- **Testing Strategy**: Pytest for units (mock core modules); integration with FastMCP stdio client mocks. Run `pytest` after each step. Smoke tests: Pipe input/output via subprocess or simple client script.
- **Integration Rule**: Each step wires code (e.g., to `server.py` or `__main__.py`) with no orphans. End with pytest + manual stdio test.
- **Best Practices**: Async for tools; Pydantic schemas; explicit logging; linear handlers with early returns. Reuse core for indexing/tools.
- **Tools/Deps**: FastMCP (stdio), pyyaml (config), watchdog (incremental), pytest.
- **Validation**: After each phase, test full server with a stdio client script (e.g., send JSON-RPC via stdin, read stdout).
- **Rollback Safety**: Commit per chunk.

### Phase 1: Basic Stdio Server with Core Protocol and Indexing
1. **Repo Setup and Dependencies**: Branch, dirs, pyproject.toml (fastmcp, pyyaml, watchdog, pytest, pydantic). Config template. Test: Install, parse CLI.
2. **Basic Stdio Server Initialization**: `server.py` with MCPServer (version="2025.6"); stdio run in `__main__.py`. Test: Starts, logs.
3. **Handshake Handler**: `@on_handshake` for version check. Test: Mock success/reject.
4. **Resource Discovery**: `@list_resources` returning empty list. Test: JSON structure.
5. **Logging Basics**: Console logging with trace IDs. Test: Logs on init.
6. **Shutdown Handler**: `@on_shutdown` for cleanup. Test: Graceful exit.
7. **Analyzer and Initial Indexing**: `analyzer.py` with MCPAnalyzer (config, core reuse, analyze()). Wire to `@on_startup`. Test: Indexes sample dir.
8. **Stub Tool (Ping) with Schemas and Errors**: Pydantic models; `@register_tool("ping")`. Update discovery. Test: Invoke via mock.

### Phase 2: Core Tools and Incremental Features
1. **Semantic Search Tool**: Pydantic; handler queries store. Wire/register. Test: Results JSON.
2. **Get Call Graph Tool**: Export graph/Mermaid. Test: Output formats.
3. **Get Function Metadata Tool**: Fetch node data. Test: JSON metadata.
4. **Query Entry Points Tool**: List entry points. Test: List JSON.
5. **Generate Mermaid Graph Tool**: Mermaid export. Test: Graph string.
6. **Simple Context Exchange**: `@update_context`, `@get_context` for session state. Test: Persist/retrieve.
7. **Incremental Indexing with Watcher**: Watchdog in startup; queue updates. Test: File change reflects.
8. **Dynamic Resource List**: Include all tools in discovery. Test: Lists 5 tools.

### Phase 3: Basic Testing and Polish
1. **Error Handling Refinements**: Typed errors with hints. Test: All cases.
2. **Unit and Integration Tests**: Cover protocol/tools. Test: >80% coverage.
3. **Basic Optimizations**: Limit outputs for efficiency. Test: Token mock.
4. **Documentation and Release Prep**: README, schemas. Test: Examples run.

## Iterative Chunk Breakdown

Broke blueprint into 15 focused chunks for stdio/minimal viable server. Each: 1 file or small update, quick to test locally.

- **Chunk 1: Repo and Deps**
- **Chunk 2: Basic Stdio Server**
- **Chunk 3: Handshake**
- **Chunk 4: Resource Discovery**
- **Chunk 5: Logging**
- **Chunk 6: Shutdown**
- **Chunk 7: Analyzer and Indexing**
- **Chunk 8: Ping Tool**
- **Chunk 9: Semantic Search Tool**
- **Chunk 10: Get Call Graph Tool**
- **Chunk 11: Get Function Metadata Tool**
- **Chunk 12: Query Entry Points Tool**
- **Chunk 13: Generate Mermaid Tool**
- **Chunk 14: Context Exchange and Incremental Watcher**
- **Chunk 15: Full Testing, Polish, Docs**

## Review and Iteration for Right-Sizing

Initial breakdown had 15 chunks from phases. Iteration 1: Merged logging/shutdown (small) into fewer, but split analyzer (core reuse complex). Iteration 2: Kept 15—each 50-200 LOC, 2-4 tests, 30-60 min implement/test. Small: TDD per chunk (failing tests first). Big: Delivers testable progress (e.g., Chunk 8: First tool over stdio). Safety: Local stdio tests immediate; no HTTP/auth distractions. Progression: Protocol → Indexing → Tools (one-by-one) → Features. No jumps: Tools use existing analyzer from Chunk 7.

# Series of Prompts for Code-Generation LLM

## Prompt for Chunk 1: Repo Setup and Dependencies

```
You are a senior Python developer building the CodeFlowGraph MCP Server in a new Git branch 'mcp-server-v1' from the existing codebase (core/ast_extractor.py, core/call_graph_builder.py, core/vector_store.py, cli/code_flow_graph.py). Focus on stdio transport for local testing; no auth or HTTP. Follow TDD: explicit, linear code.

Step 1: Write failing test in tests/mcp_server/test_setup.py: Mock pyproject.toml, assert deps include fastmcp, pyyaml, watchdog, pytest, pydantic.

Step 2: Implement:
- Dirs: code_flow_graph/mcp_server/__init__.py (empty), mcp_server/__main__.py (if __name__ == "__main__": import argparse; parser = argparse.ArgumentParser(); parser.add_argument("--config", default="mcp_server/config/default.yaml"); args = parser.parse_args(); print(f"Config: {args.config}"); exit(0)).
- pyproject.toml update: [build-system]..., [project.dependencies] add "fastmcp", "pyyaml", "watchdog>=2.0", "pytest", "pydantic". [project.scripts] add "code_flow_graph.mcp_server = 'code_flow_graph.mcp_server.__main__:main'".
- mcp_server/config/default.yaml: watch_directories: ["."], ignored_patterns: [], chromadb_path: "./code_vectors_chroma", max_graph_depth: 3.

Step 3: Refactor: Use yaml.safe_load in __main__.py stub (import yaml; with open(args.config) as f: config = yaml.safe_load(f); print(config)).

Step 4: Integrate: Ensure pip install -e . works.

Step 5: Tests: pytest tests/mcp_server/test_setup.py (mock yaml load, assert config dict).

Smoke test: git checkout -b mcp-server-v1; pip install -e .; python -m code_flow_graph.mcp_server (prints config); touch test_file.py with def hello(): pass for later.

Output: New files (__init__.py, __main__.py, default.yaml), pyproject.toml snippet, test_setup.py. No changes to core.
```

## Prompt for Chunk 2: Basic Stdio Server

```
Building on Chunk 1 (__main__.py parses --config, loads yaml to config dict). Existing core untouched.

TDD: In tests/mcp_server/test_server.py: Mock FastMCP; assert MCPServer(name="CodeFlowGraphMCP", version="2025.6") inits; assert run_stdio() called.

Implement in mcp_server/server.py:
- from fastmcp import MCPServer; server = MCPServer(name="CodeFlowGraphMCP", version="2025.6").
- In __main__.py: from mcp_server.server import server; import logging; logging.basicConfig(level=logging.INFO); server.run_stdio(); (handle config later).

Refactor: Add try-except around run for clean exit.

Integrate: Import in __main__.py; run on start.

Tests: pytest (init, stdio run mock—no actual pipe).

Smoke test: python -m code_flow_graph.mcp_server (starts, logs "Server running on stdio"; Ctrl+C exits).

Output: server.py, updated __main__.py, test_server.py additions. Wire fully; stdio focus.
```

## Prompt for Chunk 3: Handshake Handler

```
Building on Chunk 2 (server.py with MCPServer; __main__.py runs stdio). Config dict available.

TDD: In test_server.py: Mock params={"version": "2025.6"}; assert on_handshake returns {"version": "2025.6", "capabilities": ["resourceDiscovery"]}; incompatible "2024.1" raises ProtocolError.

Implement in server.py:
- @server.on_handshake async def on_handshake(params: dict) -> dict:
  - if params.get("version") != "2025.6": raise server.ProtocolError("Incompatible version").
  - import uuid; trace_id = str(uuid.uuid4()); logging.info(f"Handshake {trace_id}").
  - return {"version": "2025.6", "capabilities": ["resourceDiscovery", "state"]}.  # Core only.

Refactor: Early return on version mismatch.

Integrate: Decorator auto-registers.

Tests: pytest (success/reject; mock async, log capture).

Smoke test: Create client.py: import sys, json; json.dump({"jsonrpc": "2.0", "method": "mcp.handshake", "params": {"version": "2025.6"}}, sys.stdout); Run subprocess: p = subprocess.Popen(["python", "-m code_flow_graph.mcp_server"], stdin=PIPE, stdout=PIPE); p.communicate(input=handshake_bytes); assert response in stdout.

Output: Updated server.py, test additions, client.py for smoke. Builds on prior.
```

## Prompt for Chunk 4: Resource Discovery

```
Building on Chunk 3 (on_handshake in server.py). Stdio via __main__.py.

TDD: test_list_resources: Mock call; assert returns [] (list of dicts: name, type="tool", description, permissions=["read"], schema={"input":{}, "output":{}}).

Implement in server.py:
- @server.list_resources async def list_resources() -> list[dict]: return []  # Empty for now; log trace_id from handshake if available.

Refactor: Prepare stub resources = [] for later.

Integrate: Auto via decorator.

Tests: pytest (empty return; async mock).

Smoke test: Update client.py to send {"method": "listResources", "params": {}} after handshake; assert [] in response via subprocess.

Output: Updated server.py/tests, client.py update. Linear, explicit.
```

## Prompt for Chunk 5: Logging

```
Building on Chunk 4 (list_resources). Focus console logs for stdio.

TDD: test_logging: Assert log messages on init/handshake with trace_id format.

Implement in server.py:
- In MCPServer init: Add handler logging.getLogger("mcp").addHandler(logging.StreamHandler()).
- In handlers: Use logging.info(f"[{trace_id}] Event") where trace_id from context or uuid.

Refactor: Global logger in server.py.

Integrate: Logs in all prior handlers (e.g., handshake).

Tests: pytest (caplog fixture; assert "Handshake [uuid]" in logs).

Smoke test: Client.py run full (handshake + list); assert logs in stderr during subprocess.

Output: Updated server.py, test additions. No new deps.
```

## Prompt for Chunk 6: Shutdown Handler

```
Building on Chunk 5 (logging in handlers). Stdio graceful exit.

TDD: test_shutdown: Mock on_shutdown; assert cleanup called (e.g., log "Shutting down").

Implement in server.py:
- @server.on_shutdown async def on_shutdown(): logging.info("Server shutdown"); # Stub cleanup.

Refactor: Await any pending if added later.

Integrate: Auto; test via signal.

Tests: pytest (mock signal; assert log).

Smoke test: python -m ...; send SIGTERM (or Ctrl+C); assert "Shutting down" log.

Output: Updated server.py/tests. Simple, wired.
```

## Prompt for Chunk 7: Analyzer and Initial Indexing

```
Building on Chunk 6 (shutdown). Reuse core: from core.ast_extractor import PythonASTExtractor, etc. Config dict in __main__.py.

TDD: New tests/mcp_server/test_analyzer.py: Test init (config dict → extractor/builder/store); analyze() (mock extract_from_directory returns [FunctionElement], build adds 1 node, populate no error).

Implement in mcp_server/analyzer.py:
- from pathlib import Path; from core... import ...; class MCPAnalyzer:
  - def __init__(self, config: dict): self.config = config; root = Path(config['watch_directories'][0]); self.extractor = PythonASTExtractor(); self.extractor.project_root = root; self.builder = CallGraphBuilder(); self.builder.project_root = root; self.vector_store = CodeVectorStore(persist_directory=config['chromadb_path']) if Path(config['chromadb_path']).exists() else None.
  - async def analyze(self): import asyncio; elements = await asyncio.to_thread(self.extractor.extract_from_directory, root); self.builder.build_from_elements(elements); if self.vector_store: await asyncio.to_thread(self._populate_vector_store)  # Adapt from cli: self._populate... using builder.functions/edges.
  - def _populate_vector_store(self): for node in self.builder.functions.values(): with open(node.file_path) as f: source = f.read(); self.vector_store.add_function_node(node, source); for edge in self.builder.edges: self.vector_store.add_edge(edge).

- In server.py: @server.on_startup async def startup(): self.analyzer = MCPAnalyzer(config); await self.analyzer.analyze(); logging.info(f"Indexed {len(self.analyzer.builder.functions)} functions").

Refactor: Handle store None (log warning, skip populate).

Integrate: Import in server.py; pass config from __main__.py (load once, pass to startup).

Tests: pytest test_analyzer (mocks for core extract/build/add; assert len(functions)==1); test_server startup (mock analyze, assert log).

Smoke test: Place test_file.py (def hello(): pass) in cwd; run server; assert "Indexed 1" log; client.py irrelevant here.

Output: analyzer.py, updated server.py/__main__.py, test_analyzer.py. Core reuse explicit.
```

## Prompt for Chunk 8: Ping Tool

```
Building on Chunk 7 (analyzer in startup; config passed). Stdio testing.

TDD: In test_server.py: Test register_tool("ping"); input PingRequest(message="hi") → PingResponse(status="ok", echoed="hi"); missing message → InvalidParamsError(code=4001, data={"hint": "Message required"}).

Implement in mcp_server/tools.py:
- from pydantic import BaseModel, validator; class PingRequest(BaseModel): message: str; @validator('message') def check_message(cls, v): if not v: raise ValueError("Message required"); return v.
- class PingResponse(BaseModel): status: str; echoed: str.
- from fastmcp import register_tool; @register_tool("ping", input_model=PingRequest, output_model=PingResponse) async def ping_tool(req: PingRequest) -> PingResponse: return PingResponse(status="ok", echoed=req.message).

- In server.py: from mcp_server.tools import *; # Auto-register on import. In list_resources: return [{"name": "ping", "type": "tool", "description": "Echo message", "permissions": ["read"], "schema": {"input": PingRequest.model_json_schema(), "output": PingResponse.model_json_schema()}}].

Refactor: Pass analyzer if needed later (stub).

Integrate: Import tools in server.py after startup.

Tests: pytest (tool handler mock; validate schemas; call via mock server.call_tool).

Smoke test: Update client.py: After handshake/list, send {"method": "ping", "params": {"message": "test"}}; assert response {"result": {"status": "ok", "echoed": "test"}} via subprocess stdout.

Output: tools.py, updated server.py/tests/client.py. First testable tool over stdio.
```

## Prompt for Chunk 9: Semantic Search Tool

```
Building on Chunk 8 (tools.py with ping; server.py imports tools; analyzer available as server.analyzer). Use vector_store.query_functions.

TDD: In test_server.py: Mock analyzer.vector_store.query_functions("test", 1) → [{"metadata": {...}}]; assert SearchResponse(results=[dict]) returned.

Implement in tools.py:
- class SemanticSearchRequest(BaseModel): query: str; n_results: int = 5; filters: dict = {}.
- class SearchResponse(BaseModel): results: list[dict].
- @register_tool("semantic_search", input_model=SemanticSearchRequest, output_model=SearchResponse) async def semantic_search(req: SemanticSearchRequest, analyzer: MCPAnalyzer) -> SearchResponse:  # Pass via closure or server state.
  - if not analyzer.vector_store: raise server.ServerError("Vector store unavailable").
  - results = analyzer.vector_store.query_functions(req.query, req.n_results, req.filters); return SearchResponse(results=results).

- Update server.py list_resources: Add "semantic_search" entry with schemas (similar to ping).

Refactor: Handle None store early.

Integrate: In server.py, after startup: tools.analyzer = self.analyzer (or pass in handlers).

Tests: pytest (handler with mock analyzer/store; assert results len; error if no store).

Smoke test: Client.py call "semantic_search" {"params": {"query": "test"}}; assert JSON results (use sample indexed).

Output: Updated tools.py/server.py/tests/client.py. Builds on ping; test with indexed data.
```

## Prompt for Chunk 10: Get Call Graph Tool

```
Building on Chunk 9 (semantic_search in tools.py; analyzer passed). Use builder.export_graph.

TDD: Mock builder.export_graph() → {"functions": {...}}; assert GraphResponse(graph=dict or str).

Implement in tools.py:
- class CallGraphRequest(BaseModel): fqns: list[str] = []; depth: int = 1; format: str = "json"  # "json" or "mermaid".
- class GraphResponse(BaseModel): graph: dict | str.
- @register_tool("get_call_graph", input_model=CallGraphRequest, output_model=GraphResponse) async def get_call_graph(req: CallGraphRequest, analyzer: MCPAnalyzer) -> GraphResponse:
  - graph = analyzer.builder.export_graph(format=req.format if req.format == "mermaid" else "json"); return GraphResponse(graph=graph).

- server.py list_resources: Add "get_call_graph" with schema.

Integrate: Same as prior (analyzer passed).

Tests: pytest (mock export; json/mermaid formats; empty fqns full graph).

Smoke test: Client.py call "get_call_graph" {}; assert graph in response (dict for json).

Output: Updated tools.py/server.py/tests/client.py. Incremental tool addition.
```

(Continuing pattern for remaining chunks: Each adds one tool or feature, TDD, stdio smoke via client.py subprocess, building on analyzer/tools from prior. For Chunk 14: Combine context + watcher simply. Chunk 15: Wrap with tests/docs, no deployment.)

## Prompt for Chunk 11: Get Function Metadata Tool

```
Building on Chunk 10 (get_call_graph). Use builder.functions[fqn].

TDD: Mock functions[fqn] → FunctionNode; assert MetadataResponse with keys (name, complexity, etc.).

Implement in tools.py:
- class MetadataRequest(BaseModel): fqn: str.
- class MetadataResponse(BaseModel): **FunctionNode fields (e.g., name: str, complexity: Optional[int], ...).
- @register_tool("get_function_metadata", input_model=MetadataRequest, output_model=MetadataResponse) async def get_function_metadata(req: MetadataRequest, analyzer: MCPAnalyzer) -> MetadataResponse:
  - node = analyzer.builder.functions.get(req.fqn); if not node: raise server.InvalidParamsError("FQN not found"); return MetadataResponse(**{k: v for k,v in vars(node).items() if not k.startswith('_')}).

- server.py: Add to list_resources.

Integrate: Analyzer passed.

Tests: pytest (valid fqn → full metadata; invalid → error).

Smoke test: Client.py call with known fqn from index; assert response fields.

Output: Updated tools.py/server.py/tests/client.py.
```

## Prompt for Chunk 12: Query Entry Points Tool

```
Building on Chunk 11 (metadata tool). Use builder.get_entry_points().

TDD: Mock get_entry_points() → [FunctionNode]; assert EntryPointsResponse(entry_points=list[dict]).

Implement in tools.py:
- class EntryPointsResponse(BaseModel): entry_points: list[dict].
- @register_tool("query_entry_points", output_model=EntryPointsResponse) async def query_entry_points(analyzer: MCPAnalyzer) -> EntryPointsResponse:  # No input model.
  - eps = analyzer.builder.get_entry_points(); return EntryPointsResponse(entry_points=[vars(ep) for ep in eps]).

- server.py: Add to list_resources (no input schema).

Integrate: Same.

Tests: pytest (mock eps; assert list len).

Smoke test: Client.py call "query_entry_points" {}; assert entry_points in response.

Output: Updated tools.py/server.py/tests/client.py.
```

## Prompt for Chunk 13: Generate Mermaid Graph Tool

```
Building on Chunk 12 (entry points). Use builder.export_mermaid_graph.

TDD: Mock export_mermaid_graph() → "graph TD\n..."; assert MermaidResponse(graph=str).

Implement in tools.py:
- class MermaidRequest(BaseModel): fqns: list[str] = []; llm_optimized: bool = False.
- class MermaidResponse(BaseModel): graph: str.
- @register_tool("generate_mermaid_graph", input_model=MermaidRequest, output_model=MermaidResponse) async def generate_mermaid_graph(req: MermaidRequest, analyzer: MCPAnalyzer) -> MermaidResponse:
  - graph = analyzer.builder.export_mermaid_graph(highlight_fqns=req.fqns, llm_optimized=req.llm_optimized); return MermaidResponse(graph=graph).

- server.py: Add to list_resources.

Integrate: Analyzer.

Tests: pytest (mock export; optimized flag passed).

Smoke test: Client.py call {}; assert "graph TD" in response.

Output: Updated tools.py/server.py/tests/client.py.
```

## Prompt for Chunk 14: Context Exchange and Incremental Watcher

```
Building on Chunk 13 (mermaid tool; all 5 tools registered). Add simple context; watcher for incremental.

TDD: test_context: update_context({"key": "val"}) → merged state; get_context() → dict. test_watcher: Mock event → analyze file, update store.

Implement in server.py:
- self.context = {}  # Per-session simple dict.
- @server.update_context async def update_context(params: dict): self.context.update(params); return {"version": len(self.context)}.
- @server.get_context async def get_context(): return self.context.
- list_resources: Add "update_context", "get_context" as methods (type="method").

For watcher in analyzer.py:
- import watchdog.observers; from watchdog.events import FileSystemEventHandler.
- class WatcherHandler(FileSystemEventHandler): def on_modified(self, event): if event.src_path.endswith('.py'): asyncio.create_task(self.analyzer._incremental_update(event.src_path)).
- In startup: observer = watchdog.observers.Observer(); observer.schedule(WatcherHandler(analyzer=self.analyzer), config['watch_directories'][0], recursive=True); observer.start(); self.observer = observer.
- In analyzer.py: async def _incremental_update(self, file_path: str): elements = await asyncio.to_thread(self.extractor.extract_from_file, Path(file_path)); # Update builder/store incrementally (add new, skip same hash); if delete, remove by fqn.

Refactor: Context versioned with len; watcher debounce stub (sleep 1s).

Integrate: Auto for methods; observer in startup, stop in shutdown.

Tests: pytest test_server (context merge/get); test_analyzer (mock event, assert update called).

Smoke test: Client.py update_context {"test": "val"}; get_context assert; touch test_file.py; re-call semantic_search, assert reflects.

Output: Updated server.py/analyzer.py/tests/client.py. Core features complete.
```

## Prompt for Chunk 15: Full Testing, Polish, Docs

```
Building on Chunk 14 (context/watcher; 5 tools + methods). Finalize for local stdio testing.

TDD/Polish: In test_server.py: Add error tests (InvalidParams for bad tool params, ServerError for no store). Limit outputs (e.g., in tools, truncate results[:10]). Mock tiktoken for <1000 char efficiency.

Implement refinements:
- In tools: For search/graph, if len(results)>10: results = results[:10]; log warning.
- Errors: Consistent code=4001 for params, 5001 for server; data={"hint": "..."}.

Full tests: Ensure >80% coverage (run pytest --cov=mcp_server).

Docs: mcp_server/README.md: "Setup: pip install -e .; Run: python -m ...; Test: Use client.py for handshake/tools. Config: Edit default.yaml for watch/chroma. Tools: semantic_search(query), etc. with JSON examples."

Refactor: Clean imports/logs.

Integrate: All wired; add --help in __main__.py.

Tests: pytest all (protocol, tools, context, watcher mock, errors).

Smoke test: Full client.py script: Handshake → list_resources (5 tools) → semantic_search → get_call_graph → context update/get → file touch → re-search. Assert all succeed.

Output: Updated tests (additions), README.md, client.py full. Project testable over stdio; ready for local use.
```
