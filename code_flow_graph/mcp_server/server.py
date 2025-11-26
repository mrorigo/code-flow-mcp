from mcp.server.fastmcp import FastMCP
import mcp.types as types
import logging
import sys
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
    analysis_status: Optional[str] = None
    indexed_functions: Optional[int] = None


class SearchResponse(BaseModel):
    results: list[dict] | str
    analysis_status: Optional[str] = None


class GraphResponse(BaseModel):
    graph: dict | str
    analysis_status: Optional[str] = None


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
    summary: Optional[str]
    analysis_status: Optional[str] = None


class EntryPointsResponse(BaseModel):
    entry_points: list[dict]
    analysis_status: Optional[str] = None


class MermaidResponse(BaseModel):
    graph: str
    analysis_status: Optional[str] = None

# Global logger for MCP
logger = logging.getLogger("mcp")
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)

@dataclass
class AppContext:
    analyzer: MCPAnalyzer = None

async def on_shutdown():
    logger.info("Server shutdown")
    # Shutdown analyzer (which handles file watcher and background cleanup)
    if hasattr(server, 'analyzer') and server.analyzer:
        server.analyzer.shutdown()

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # Startup
    config = getattr(server, 'config', None)
    if config:
        server.analyzer = MCPAnalyzer(config)
        # Start analysis in background instead of blocking
        await server.analyzer.start_analysis()
        logger.info("Server started, analysis running in background")
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
    Simple ping tool to echo a message and report analysis status.
    """
    analysis_status = None
    indexed_functions = None
    
    if hasattr(server, 'analyzer') and server.analyzer:
        state = server.analyzer.analysis_state
        analysis_status = state.value
        indexed_functions = len(server.analyzer.builder.functions) if server.analyzer.builder else 0
    
    return PingResponse(
        status="ok", 
        echoed=message,
        analysis_status=analysis_status,
        indexed_functions=indexed_functions
    )


import json

def format_search_results_as_markdown(results: list[dict]) -> str:
    """
    Helper function to format search results as a markdown string optimized for LLM ingestion.
    """
    if not results:
        return "No results found."

    def _parse_list_field(value: Any) -> List[str]:
        """Helper to safely parse list fields that might be JSON strings."""
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        if isinstance(value, list):
            return value
        return []

    markdown = "# Semantic Search Results\n\n"
    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        markdown += f"## Result {i}\n"
        markdown += f"- **Distance**: {result.get('distance', 'N/A')}\n"
        
        if metadata.get('type') == 'structured_data':
             markdown += f"- **File**: {metadata.get('file_path', 'N/A')}\n"
             markdown += f"- **Key**: {metadata.get('key_name', 'N/A')}\n"
             markdown += f"- **Path**: {metadata.get('json_path', 'N/A')}\n"
             markdown += f"- **Value Type**: {metadata.get('value_type', 'N/A')}\n"
             markdown += f"- **Content**: {result.get('document', 'N/A')}\n"
        else:
            markdown += f"- **Fully Qualified Name**: {metadata.get('fully_qualified_name', 'N/A')}\n"
            markdown += f"- **Name**: {metadata.get('name', 'N/A')}\n"
            markdown += f"- **Type**: {metadata.get('type', 'N/A')}\n"
            markdown += f"- **Class**: {metadata.get('class_name', 'N/A')}\n"
            markdown += f"- **File Path**: {metadata.get('file_path', 'N/A')}\n"
            markdown += f"- **Line Start**: {metadata.get('line_start', 'N/A')}\n"
            markdown += f"- **Line End**: {metadata.get('line_end', 'N/A')}\n"
            # markdown += f"- **Parameters**: {metadata.get('parameter_count', 'N/A')} ({', '.join(metadata.get('parameters', []))})\n"
            markdown += f"- **Return Type**: {metadata.get('return_type', 'N/A')}\n"
            markdown += f"- **Is Method**: {metadata.get('is_method', 'N/A')}\n"
            markdown += f"- **Is Async**: {metadata.get('is_async', 'N/A')}\n"
            markdown += f"- **Is Entry Point**: {metadata.get('is_entry_point', 'N/A')}\n"
            markdown += f"- **Access Modifier**: {metadata.get('access_modifier', 'N/A')}\n" if metadata.get('access_modifier') else ""
            markdown += f"- **Complexity**: {metadata.get('complexity', 'N/A')}\n"
            # markdown += f"- **NLOC**: {metadata.get('nloc', 'N/A')}\n"
            markdown += f"- **Incoming Degree**: {metadata.get('incoming_degree', 'N/A')}\n"
            markdown += f"- **Outgoing Degree**: {metadata.get('outgoing_degree', 'N/A')}\n"
            
            ext_deps = _parse_list_field(metadata.get('external_dependencies'))
            markdown += f"- **External Deps**: {', '.join(ext_deps)}\n" if ext_deps else ""
            
            decorators = _parse_list_field(metadata.get('decorators'))
            # Decorators might be dicts or strings, handle accordingly if needed, but for now join string representation
            # If decorators are stored as list of dicts in JSON, we might want to extract names.
            # Based on vector_store.py: "decorators": json.dumps(node.decorators) where node.decorators is List[Dict[str, Any]]
            # So parsed decorators will be a list of dicts.
            if decorators and isinstance(decorators[0], dict):
                 decorator_names = [d.get('name', str(d)) for d in decorators]
                 markdown += f"- **Decorators**: {', '.join(decorator_names)}\n"
            elif decorators:
                 markdown += f"- **Decorators**: {', '.join([str(d) for d in decorators])}\n"

            catches = _parse_list_field(metadata.get('catches_exceptions'))
            markdown += f"- **Catches**: {', '.join(catches)}\n" if catches else ""
            
            local_vars = _parse_list_field(metadata.get('local_variables_declared'))
            markdown += f"- **Local Variables**: {', '.join(local_vars)}\n"
            
            markdown += f"- **Has Docstring**: {metadata.get('has_docstring', 'N/A')}\n"
            markdown += f"- **Summary**: {metadata.get('summary', 'N/A')}\n"
            # markdown += f"- **Hash Body**: {metadata.get('hash_body', 'N/A')}\n"
            markdown += f"- **Document**: {result.get('document', 'N/A')}\n"
        markdown += "\n"
    return markdown


@server.tool(name="semantic_search")
async def semantic_search(query: str = Field(description="Search query string"),
                          n_results: int = Field(default=5, description="Number of results to return"),
                          filters: dict = Field(default={}, description="Optional filters to apply to the search results"),
                          format: str = Field(default="markdown", description="Output format: 'markdown' or 'json'")
                          ) -> SearchResponse:
    """
    Perform semantic search in codebase using vector similarity.
    """
    # Get analysis status
    analysis_status = server.analyzer.analysis_state.value if hasattr(server, 'analyzer') and server.analyzer else None
    
    # Validate parameters
    if n_results < 1:
        raise ValueError("n_results must be positive")

    if not server.analyzer or not server.analyzer.vector_store:
        raise MCPError(5001, "Vector store unavailable", "Ensure the vector store is properly initialized")
    try:
        results = server.analyzer.vector_store.query_codebase(query, n_results, filters)
        if len(results) > 10:
            results = results[:10]
            logger.warning(f"Truncated semantic search results from {len(results)} to 10")
        if format == "markdown":
            formatted_results = format_search_results_as_markdown(results)
            return SearchResponse(results=formatted_results, analysis_status=analysis_status)
        else:
            return SearchResponse(results=results, analysis_status=analysis_status)
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
    analysis_status = server.analyzer.analysis_state.value if hasattr(server, 'analyzer') and server.analyzer else None
    
    if not server.analyzer or not server.analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    graph = server.analyzer.builder.export_graph(format=format if format == "mermaid" else "json")
    return GraphResponse(graph=graph, analysis_status=analysis_status)


@server.tool(name="get_function_metadata")
async def get_function_metadata(fqn: str = Field(description="Fully qualified name of the function")) -> MetadataResponse:
    """
    Retrieve metadata for a specific function by its fully qualified name.

    Args:
        fqn: The fully qualified name of the function to retrieve metadata for.
    Returns:
        MetadataResponse containing detailed function metadata.
    """
    analysis_status = server.analyzer.analysis_state.value if hasattr(server, 'analyzer') and server.analyzer else None
    
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
    node_dict['analysis_status'] = analysis_status
    return MetadataResponse(**node_dict)


@server.tool(name="query_entry_points")
async def query_entry_points() -> EntryPointsResponse:
    """
    Retrieve all identified entry points in the codebase.
    """
    analysis_status = server.analyzer.analysis_state.value if hasattr(server, 'analyzer') and server.analyzer else None
    
    if not server.analyzer or not server.analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    eps = server.analyzer.builder.get_entry_points()
    return EntryPointsResponse(entry_points=[vars(ep) for ep in eps], analysis_status=analysis_status)

@server.tool(name="generate_mermaid_graph")
async def generate_mermaid_graph(fqns: list[str] = Field(default=[], description="List of fully qualified names to highlight in the graph"),
                                  llm_optimized: bool = Field(description="Whether to optimize the graph for LLM consumption")) -> MermaidResponse:
    """
    Generate a Mermaid diagram for the call graph.
    """
    analysis_status = server.analyzer.analysis_state.value if hasattr(server, 'analyzer') and server.analyzer else None
    
    if not server.analyzer or not server.analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    graph = server.analyzer.builder.export_mermaid_graph(highlight_fqns=fqns, llm_optimized=llm_optimized)
    return MermaidResponse(graph=graph, analysis_status=analysis_status)


class CleanupResponse(BaseModel):
    removed_documents: int
    errors: int
    stale_paths: int
    message: str


@server.tool(name="cleanup_stale_references")
async def cleanup_stale_references() -> CleanupResponse:
    """
    Manually trigger cleanup of stale file references in the vector store.
    Removes documents that reference files that no longer exist on the filesystem.
    """
    if not server.analyzer:
        raise MCPError(5001, "Analyzer unavailable", "Ensure the analyzer is properly initialized")

    try:
        cleanup_stats = await server.analyzer.cleanup_stale_references()
        message = f"Cleanup completed: removed {cleanup_stats['removed_documents']} documents"
        if cleanup_stats['errors'] > 0:
            message += f", {cleanup_stats['errors']} errors"
        if cleanup_stats['stale_paths'] > 0:
            message += f", found {cleanup_stats['stale_paths']} stale paths"

        return CleanupResponse(
            removed_documents=cleanup_stats['removed_documents'],
            errors=cleanup_stats['errors'],
            stale_paths=cleanup_stats.get('stale_paths', 0),
            message=message
        )
    except Exception as e:
        raise MCPError(5001, f"Cleanup failed: {str(e)}", "Check server logs for details")
