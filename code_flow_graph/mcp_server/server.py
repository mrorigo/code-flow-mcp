from mcp.server.fastmcp import FastMCP
import mcp.types as types
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pydantic import BaseModel, Field
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
async def ping_tool(message: str = Field(description="Message to echo")) -> PingResponse:
    """
    Simple ping tool to echo a message.
    """
    return PingResponse(status="ok", echoed=message)


@server.tool(name="semantic_search")
async def semantic_search(query: str = Field(description="Search query string"),
                          n_results: int = Field(default=5, description="Number of results to return"),
                          filters: dict = Field(default={}, description="Optional filters to apply to the search results")
                          ) -> SearchResponse:
    """
    Perform semantic search in codebase using vector similarity.
    """
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
async def get_call_graph(fqns: list[str] = Field(default=[], description="List of fully qualified names to include in the graph"),
                         depth: int = Field(default=1, description="Depth of the call graph to export"),
                         format: str = Field(default="json", description="Output format, either 'json' or 'mermaid'")) -> GraphResponse:
    """
    Export the call graph in specified format.
    """
    if not server.analyzer or not server.analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    graph = server.analyzer.builder.export_graph(format=format if format == "mermaid" else "json")
    return GraphResponse(graph=graph)


@server.tool(name="get_function_metadata")
async def get_function_metadata(fqn: str = Field(description="Fully qualified name of the function")) -> MetadataResponse:
    """
    Retrieve metadata for a specific function by its fully qualified name.

    Args:
        fqn: The fully qualified name of the function to retrieve metadata for.
    Returns:
        MetadataResponse containing detailed function metadata.
    """
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
    """
    Retrieve all identified entry points in the codebase.
    """
    if not server.analyzer or not server.analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    eps = server.analyzer.builder.get_entry_points()
    return EntryPointsResponse(entry_points=[vars(ep) for ep in eps])

@server.tool(name="generate_mermaid_graph")
async def generate_mermaid_graph(fqns: list[str] = Field(default=[], description="List of fully qualified names to highlight in the graph"),
                                 llm_optimized: bool = Field(description="Whether to optimize the graph for LLM consumption")) -> MermaidResponse:
    """
    Generate a Mermaid diagram for the call graph.
    """
    if not server.analyzer or not server.analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    graph = server.analyzer.builder.export_mermaid_graph(highlight_fqns=fqns, llm_optimized=llm_optimized)
    return MermaidResponse(graph=graph)
