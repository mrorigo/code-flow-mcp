from mcp.server.fastmcp import FastMCP
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
from code_flow_graph.mcp_server.analyzer import MCPAnalyzer

# Pydantic models for tools
class MCPError(Exception):
    """Custom MCP error with code and hint."""

    def __init__(self, code: int, message: str, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = {"hint": hint} if hint else {}


class PingResponse(BaseModel):
    status: str
    echoed: str


class SearchResponse(BaseModel):
    results: list[dict]


class GraphResponse(BaseModel):
    graph: dict | str


class MetadataResponse(BaseModel):
    name: str
    fully_qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    parameters: List[str]
    incoming_edges: List[dict]  # Simplified as dict for serialization
    outgoing_edges: List[dict]  # Simplified as dict for serialization
    return_type: Optional[str]
    is_entry_point: bool
    is_exported: bool
    is_async: bool
    is_static: bool
    access_modifier: Optional[str]
    docstring: Optional[str]
    is_method: bool
    class_name: Optional[str]
    complexity: Optional[int]
    nloc: Optional[int]
    external_dependencies: List[str]
    decorators: List[Dict[str, Any]]
    catches_exceptions: List[str]
    local_variables_declared: List[str]
    hash_body: Optional[str]


class EntryPointsResponse(BaseModel):
    entry_points: list[dict]


class MermaidResponse(BaseModel):
    graph: str

# Global logger for MCP
logger = logging.getLogger("mcp")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

@dataclass
class AppContext:
    analyzer: MCPAnalyzer = None

async def on_shutdown():
    logger.info("Server shutdown")
    # Stop file watcher
    if hasattr(server, 'analyzer') and server.analyzer and server.analyzer.observer:
        server.analyzer.observer.stop()
        server.analyzer.observer.join()

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # Startup
    config = getattr(server, 'config', None)
    if config:
        server.analyzer = MCPAnalyzer(config)
        await server.analyzer.analyze()
        logger.info(f"Indexed {len(server.analyzer.builder.functions)} functions")
        # Analyzer is now accessible via server.analyzer
    else:
        logger.warning("No config provided to server, skipping analysis")

    server.context = {}  # Per-session simple dict.

    try:
        yield AppContext()
    finally:
        # Shutdown
        await on_shutdown()

server = FastMCP("CodeFlowGraphMCP", lifespan=lifespan)

# Tool functions with decorators
@server.tool(name="ping")
async def ping_tool(message: str) -> PingResponse:
    """
    Simple ping tool to echo a message.

    Args:
        message: The message to echo.
    Returns:
        PingResponse with status and echoed message.
    """
    return PingResponse(status="ok", echoed=message)


@server.tool(name="semantic_search")
async def semantic_search(query: str, n_results: int = 5, filters: dict = {}) -> SearchResponse:
    # Validate parameters
    if n_results < 1:
        raise ValueError("n_results must be positive")

    if not server.analyzer or not server.analyzer.vector_store:
        raise MCPError(5001, "Vector store unavailable", "Ensure the vector store is properly initialized")
    try:
        results = server.analyzer.vector_store.query_functions(query, n_results, filters)
        if len(results) > 10:
            results = results[:10]
            logger.warning(f"Truncated semantic search results from {len(results)} to 10")
        return SearchResponse(results=results)
    except Exception as e:
        if "Invalid parameters" in str(e) or isinstance(e, ValueError):
            raise MCPError(4001, "Invalid parameters", "Check parameter values and ensure they are valid") from e
        raise


@server.tool(name="get_call_graph")
async def get_call_graph(fqns: list[str] = [], depth: int = 1, format: str = "json") -> GraphResponse:
    if not server.analyzer or not server.analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    graph = server.analyzer.builder.export_graph(format=format if format == "mermaid" else "json")
    return GraphResponse(graph=graph)


@server.tool(name="get_function_metadata")
async def get_function_metadata(fqn: str) -> MetadataResponse:
    if not server.analyzer or not server.analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    node = server.analyzer.builder.functions.get(fqn)
    if not node:
        raise MCPError(4001, f"FQN not found: {fqn}", "Check the fully qualified name and ensure it exists in the codebase")
    # Convert edges to dicts for serialization
    node_dict = {k: v for k, v in vars(node).items() if not k.startswith('_')}
    # Convert edges to simple dicts
    node_dict['incoming_edges'] = [vars(edge) for edge in node.incoming_edges]
    node_dict['outgoing_edges'] = [vars(edge) for edge in node.outgoing_edges]
    return MetadataResponse(**node_dict)


@server.tool(name="query_entry_points")
async def query_entry_points() -> EntryPointsResponse:
    if not server.analyzer or not server.analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    eps = server.analyzer.builder.get_entry_points()
    return EntryPointsResponse(entry_points=[vars(ep) for ep in eps])


@server.tool(name="generate_mermaid_graph")
async def generate_mermaid_graph(fqns: list[str] = [], llm_optimized: bool = False) -> MermaidResponse:
    if not server.analyzer or not server.analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    graph = server.analyzer.builder.export_mermaid_graph(highlight_fqns=fqns, llm_optimized=llm_optimized)
    return MermaidResponse(graph=graph)

# Stub for resources
resources = []

@server.tool()
async def on_handshake(version: str) -> dict:
    if version != "2025.6":
        raise MCPError(4001, "Incompatible version", "Server requires version 2025.6")
    import uuid
    trace_id = str(uuid.uuid4())
    logger.info(f"[{trace_id}] Handshake")
    # Store trace_id for later use
    server.trace_id = trace_id
    return {"version": "2025.6", "capabilities": ["resourceDiscovery", "state"]}

@server.tool()
async def list_resources() -> list[dict]:
    # Log trace_id from handshake if available
    if hasattr(server, 'trace_id'):
        logger.info(f"[{server.trace_id}] List resources")
    return [
        {
            "name": "ping",
            "type": "tool",
            "description": "Echo message",
            "permissions": ["read"],
            "schema": {
                "input": {"type": "object", "properties": {"message": {"type": "string"}}},
                "output": PingResponse.model_json_schema()
            }
        },
        {
            "name": "semantic_search",
            "type": "tool",
            "description": "Search functions semantically",
            "permissions": ["read"],
            "schema": {
                "input": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "n_results": {"type": "integer", "default": 5},
                        "filters": {"type": "object", "default": {}}
                    }
                },
                "output": SearchResponse.model_json_schema()
            }
        },
        {
            "name": "get_call_graph",
            "type": "tool",
            "description": "Get call graph in JSON or Mermaid format",
            "permissions": ["read"],
            "schema": {
                "input": {
                    "type": "object",
                    "properties": {
                        "fqns": {"type": "array", "items": {"type": "string"}, "default": []},
                        "depth": {"type": "integer", "default": 1},
                        "format": {"type": "string", "default": "json"}
                    }
                },
                "output": GraphResponse.model_json_schema()
            }
        },
        {
            "name": "get_function_metadata",
            "type": "tool",
            "description": "Get metadata for a function by FQN",
            "permissions": ["read"],
            "schema": {
                "input": {"type": "object", "properties": {"fqn": {"type": "string"}}},
                "output": MetadataResponse.model_json_schema()
            }
        },
        {
            "name": "query_entry_points",
            "type": "tool",
            "description": "Get all identified entry points",
            "permissions": ["read"],
            "schema": {
                "input": {},
                "output": EntryPointsResponse.model_json_schema()
            }
        },
        {
            "name": "generate_mermaid_graph",
            "type": "tool",
            "description": "Generate Mermaid graph for call graph",
            "permissions": ["read"],
            "schema": {
                "input": {
                    "type": "object",
                    "properties": {
                        "fqns": {"type": "array", "items": {"type": "string"}, "default": []},
                        "llm_optimized": {"type": "boolean", "default": False}
                    }
                },
                "output": MermaidResponse.model_json_schema()
            }
        },
        {
            "name": "update_context",
            "type": "method",
            "description": "Update session context",
            "permissions": ["write"],
            "schema": {
                "input": {"type": "object", "additionalProperties": True},
                "output": {"type": "object", "properties": {"version": {"type": "integer"}}}
            }
        },
        {
            "name": "get_context",
            "type": "method",
            "description": "Get session context",
            "permissions": ["read"],
            "schema": {
                "input": {},
                "output": {"type": "object", "additionalProperties": True}
            }
        }
    ]

@server.tool()
async def update_context(params: dict):
    server.context.update(params)
    return {"version": len(server.context)}

@server.tool()
async def get_context():
    return server.context

# Tools are now registered via decorators