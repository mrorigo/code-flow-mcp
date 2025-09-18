from fastmcp import FastMCP
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from code_flow_graph.mcp_server.analyzer import MCPAnalyzer
from code_flow_graph.mcp_server.tools import PingRequest, PingResponse, ping_tool, SemanticSearchRequest, SearchResponse, semantic_search, CallGraphRequest, GraphResponse, get_call_graph, MetadataRequest, MetadataResponse, get_function_metadata, EntryPointsResponse, query_entry_points, MermaidRequest, MermaidResponse, generate_mermaid_graph
from code_flow_graph.mcp_server.tools import *  # Auto-register on import

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
        # Set analyzer in tools module
        import code_flow_graph.mcp_server.tools as tools
        tools.analyzer = server.analyzer
    else:
        logger.warning("No config provided to server, skipping analysis")

    server.context = {}  # Per-session simple dict.

    try:
        yield AppContext()
    finally:
        # Shutdown
        await on_shutdown()

server = FastMCP(name="CodeFlowGraphMCP", version="2025.6", lifespan=lifespan)

# Stub for resources
resources = []

async def on_handshake(version: str) -> dict:
    if version != "2025.6":
        from code_flow_graph.mcp_server.tools import MCPError
        raise MCPError(4001, "Incompatible version", "Server requires version 2025.6")
    import uuid
    trace_id = str(uuid.uuid4())
    logger.info(f"[{trace_id}] Handshake")
    # Store trace_id for later use
    server.trace_id = trace_id
    return {"version": "2025.6", "capabilities": ["resourceDiscovery", "state"]}

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
                "input": PingRequest.model_json_schema(),
                "output": PingResponse.model_json_schema()
            }
        },
        {
            "name": "semantic_search",
            "type": "tool",
            "description": "Search functions semantically",
            "permissions": ["read"],
            "schema": {
                "input": SemanticSearchRequest.model_json_schema(),
                "output": SearchResponse.model_json_schema()
            }
        },
        {
            "name": "get_call_graph",
            "type": "tool",
            "description": "Get call graph in JSON or Mermaid format",
            "permissions": ["read"],
            "schema": {
                "input": CallGraphRequest.model_json_schema(),
                "output": GraphResponse.model_json_schema()
            }
        },
        {
            "name": "get_function_metadata",
            "type": "tool",
            "description": "Get metadata for a function by FQN",
            "permissions": ["read"],
            "schema": {
                "input": MetadataRequest.model_json_schema(),
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
                "input": MermaidRequest.model_json_schema(),
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

async def update_context(params: dict):
    server.context.update(params)
    return {"version": len(server.context)}

async def get_context():
    return server.context

# Register with server
server.tool()(on_handshake)
server.tool(name="listResources")(list_resources)
server.tool(name="ping")(ping_tool)
server.tool(name="semantic_search")(semantic_search)
server.tool(name="get_call_graph")(get_call_graph)
server.tool(name="get_function_metadata")(get_function_metadata)
server.tool(name="query_entry_points")(query_entry_points)
server.tool(name="generate_mermaid_graph")(generate_mermaid_graph)
server.tool(name="update_context")(update_context)
server.tool(name="get_context")(get_context)