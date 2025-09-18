from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Global analyzer instance (set by server after startup)
analyzer = None


class MCPError(Exception):
    """Custom MCP error with code and hint."""

    def __init__(self, code: int, message: str, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = {"hint": hint} if hint else {}


class PingRequest(BaseModel):
    message: str

    @field_validator('message')
    @classmethod
    def check_message(cls, v):
        if not v:
            raise ValueError("Message required")
        return v


class PingResponse(BaseModel):
    status: str
    echoed: str


class SemanticSearchRequest(BaseModel):
    query: str
    n_results: int = 5
    filters: dict = {}

    @field_validator('n_results')
    @classmethod
    def validate_n_results(cls, v):
        if v < 1:
            raise ValueError("n_results must be positive")
        return v


class SearchResponse(BaseModel):
    results: list[dict]


class CallGraphRequest(BaseModel):
    fqns: list[str] = []
    depth: int = 1
    format: str = "json"  # "json" or "mermaid"


class GraphResponse(BaseModel):
    graph: dict | str


class MetadataRequest(BaseModel):
    fqn: str


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


class MermaidRequest(BaseModel):
    fqns: list[str] = []
    llm_optimized: bool = False


class MermaidResponse(BaseModel):
    graph: str


async def ping_tool(req: PingRequest) -> PingResponse:
    return PingResponse(status="ok", echoed=req.message)


async def semantic_search(req: SemanticSearchRequest) -> SearchResponse:
    if not analyzer or not analyzer.vector_store:
        raise MCPError(5001, "Vector store unavailable", "Ensure the vector store is properly initialized")
    try:
        results = analyzer.vector_store.query_functions(req.query, req.n_results, req.filters)
        if len(results) > 10:
            results = results[:10]
            logger.warning(f"Truncated semantic search results from {len(results)} to 10")
        return SearchResponse(results=results)
    except Exception as e:
        if "Invalid parameters" in str(e) or isinstance(e, ValueError):
            raise MCPError(4001, "Invalid parameters", "Check parameter values and ensure they are valid") from e
        raise


async def get_call_graph(req: CallGraphRequest) -> GraphResponse:
    if not analyzer or not analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    graph = analyzer.builder.export_graph(format=req.format if req.format == "mermaid" else "json")
    return GraphResponse(graph=graph)


async def get_function_metadata(req: MetadataRequest) -> MetadataResponse:
    if not analyzer or not analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    node = analyzer.builder.functions.get(req.fqn)
    if not node:
        raise MCPError(4001, f"FQN not found: {req.fqn}", "Check the fully qualified name and ensure it exists in the codebase")
    # Convert edges to dicts for serialization
    node_dict = {k: v for k, v in vars(node).items() if not k.startswith('_')}
    # Convert edges to simple dicts
    node_dict['incoming_edges'] = [vars(edge) for edge in node.incoming_edges]
    node_dict['outgoing_edges'] = [vars(edge) for edge in node.outgoing_edges]
    return MetadataResponse(**node_dict)


async def query_entry_points() -> EntryPointsResponse:
    if not analyzer or not analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    eps = analyzer.builder.get_entry_points()
    return EntryPointsResponse(entry_points=[vars(ep) for ep in eps])


async def generate_mermaid_graph(req: MermaidRequest, analyzer_param=None) -> MermaidResponse:
    if not analyzer or not analyzer.builder:
        raise MCPError(5001, "Builder unavailable", "Ensure the call graph builder is properly initialized")
    graph = analyzer.builder.export_mermaid_graph(highlight_fqns=req.fqns, llm_optimized=req.llm_optimized)
    return MermaidResponse(graph=graph)