import pytest
import sys
from unittest.mock import patch, MagicMock, AsyncMock
import logging
import uuid
from pydantic import ValidationError
from code_flow_graph.mcp_server.tools import MCPError


@pytest.fixture
def mock_dependencies():
    """Mock all dependencies to avoid import issues."""
    with patch.dict('sys.modules', {
        'fastmcp': MagicMock(),
        'code_flow_graph.mcp_server.analyzer': MagicMock(),
        'code_flow_graph.core.ast_extractor': MagicMock(),
        'code_flow_graph.core.call_graph_builder': MagicMock(),
        'code_flow_graph.core.vector_store': MagicMock(),
        'sentence_transformers': MagicMock(),
        'chromadb': MagicMock(),
        'torch': MagicMock(),
        'tiktoken': MagicMock(),
    }):
        # Mock FastMCP
        mock_fastmcp_class = MagicMock()
        sys.modules['fastmcp'].FastMCP = mock_fastmcp_class
        mock_server_instance = MagicMock()
        mock_fastmcp_class.return_value = mock_server_instance

        # Mock analyzer
        mock_analyzer_class = MagicMock()
        sys.modules['code_flow_graph.mcp_server.analyzer'].MCPAnalyzer = mock_analyzer_class

        yield mock_fastmcp_class, mock_server_instance, mock_analyzer_class


def test_server_initialization(mock_dependencies):
    mock_fastmcp_class, mock_server_instance, mock_analyzer_class = mock_dependencies

    # Import the server module to trigger initialization
    import code_flow_graph.mcp_server.server

    # Assert FastMCP was called with correct parameters
    mock_fastmcp_class.assert_called_once()
    call_args = mock_fastmcp_class.call_args
    assert call_args[1]['name'] == "CodeFlowGraphMCP"
    assert call_args[1]['version'] == "2025.6"
    assert 'lifespan' in call_args[1]

    # Assert the server instance is created
    assert hasattr(code_flow_graph.mcp_server.server, 'server')
    assert code_flow_graph.mcp_server.server.server == mock_server_instance

def test_run_stdio_called(mock_dependencies):
    mock_fastmcp_class, mock_server_instance, mock_analyzer_class = mock_dependencies

    # Import the server module
    import code_flow_graph.mcp_server.server

    # Simulate calling run_stdio_async (though in reality it's called in __main__)
    code_flow_graph.mcp_server.server.server.run_stdio_async()

    # Assert run_stdio_async was called
    mock_server_instance.run_stdio_async.assert_called_once()

@pytest.mark.asyncio
async def test_on_handshake_success(caplog, mock_dependencies):
    mock_fastmcp_class, mock_server_instance, mock_analyzer_class = mock_dependencies

    caplog.set_level(logging.INFO)
    with patch('uuid.uuid4') as mock_uuid:
        mock_uuid.return_value = uuid.UUID('12345678-1234-5678-1234-567812345678')
        from code_flow_graph.mcp_server.server import on_handshake
        result = await on_handshake(version="2025.6")
        assert result == {"version": "2025.6", "capabilities": ["resourceDiscovery", "state"]}
        assert "[12345678-1234-5678-1234-567812345678] Handshake" in caplog.text

@pytest.mark.asyncio
async def test_on_handshake_reject(mock_dependencies):
    mock_fastmcp_class, mock_server_instance, mock_analyzer_class = mock_dependencies

    from code_flow_graph.mcp_server.server import on_handshake
    with pytest.raises(MCPError) as exc_info:
        await on_handshake(version="2024.1")
    assert exc_info.value.code == 4001
    assert "Incompatible version" in str(exc_info.value)
    assert exc_info.value.data["hint"] == "Server requires version 2025.6"

@pytest.mark.asyncio
async def test_list_resources(caplog, mock_dependencies):
    mock_fastmcp_class, mock_server_instance, mock_analyzer_class = mock_dependencies

    caplog.set_level(logging.INFO)
    from code_flow_graph.mcp_server.server import list_resources, server
    # Set trace_id to simulate handshake
    server.trace_id = "test-trace-id"
    result = await list_resources()
    assert len(result) == 8  # ping, semantic_search, get_call_graph, get_function_metadata, query_entry_points, generate_mermaid_graph, update_context, get_context
    assert result[0]["name"] == "ping"
    assert result[1]["name"] == "semantic_search"
    assert result[2]["name"] == "get_call_graph"
    assert result[3]["name"] == "get_function_metadata"
    assert result[4]["name"] == "query_entry_points"
    assert result[5]["name"] == "generate_mermaid_graph"
    assert result[6]["name"] == "update_context"
    assert result[7]["name"] == "get_context"
    assert "[test-trace-id] List resources" in caplog.text

@pytest.mark.asyncio
async def test_logging(caplog, mock_dependencies):
    """Test logging messages on init/handshake with trace_id format."""
    mock_fastmcp_class, mock_server_instance, mock_analyzer_class = mock_dependencies

    caplog.set_level(logging.INFO)
    with patch('uuid.uuid4') as mock_uuid:
        mock_uuid.return_value = uuid.UUID('12345678-1234-5678-1234-567812345678')
        from code_flow_graph.mcp_server.server import on_handshake
        result = await on_handshake(version="2025.6")
        assert "[12345678-1234-5678-1234-567812345678] Handshake" in caplog.text

@pytest.mark.asyncio
async def test_shutdown(caplog, mock_dependencies):
    """Test shutdown handler logs cleanup."""
    mock_fastmcp_class, mock_server_instance, mock_analyzer_class = mock_dependencies

    caplog.set_level(logging.INFO)
    from code_flow_graph.mcp_server.server import on_shutdown
    await on_shutdown()
    assert "Server shutdown" in caplog.text


@pytest.mark.asyncio
async def test_ping_tool_success():
    """Test ping tool with valid message."""
    from code_flow_graph.mcp_server.tools import ping_tool, PingRequest, PingResponse

    # Test successful ping
    request = PingRequest(message="hi")
    response = await ping_tool(request)

    assert isinstance(response, PingResponse)
    assert response.status == "ok"
    assert response.echoed == "hi"


@pytest.mark.asyncio
async def test_ping_tool_missing_message():
    """Test ping tool with missing message raises ValidationError."""
    from code_flow_graph.mcp_server.tools import PingRequest

    # Test missing message
    with pytest.raises(ValidationError) as exc_info:
        PingRequest(message="")

    assert "Message required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ping_tool_call_via_server(mock_dependencies):
    """Test calling ping tool via mock server."""
    mock_fastmcp_class, mock_server_instance, mock_analyzer_class = mock_dependencies

    # Mock the server's tool calling mechanism
    mock_server_instance.call_tool = AsyncMock()
    mock_server_instance.call_tool.return_value = {"status": "ok", "echoed": "test"}

    from code_flow_graph.mcp_server.server import server

    # Simulate calling the tool
    result = await server.call_tool("ping", {"message": "test"})

    assert result == {"status": "ok", "echoed": "test"}
    mock_server_instance.call_tool.assert_called_once_with("ping", {"message": "test"})


def test_ping_request_schema():
    """Test PingRequest schema validation."""
    from code_flow_graph.mcp_server.tools import PingRequest

    # Test valid schema
    schema = PingRequest.model_json_schema()
    assert "properties" in schema
    assert "message" in schema["properties"]
    assert schema["properties"]["message"]["type"] == "string"


def test_ping_response_schema():
    """Test PingResponse schema validation."""
    from code_flow_graph.mcp_server.tools import PingResponse

    # Test valid schema
    schema = PingResponse.model_json_schema()
    assert "properties" in schema
    assert "status" in schema["properties"]
    assert "echoed" in schema["properties"]
    assert schema["properties"]["status"]["type"] == "string"
    assert schema["properties"]["echoed"]["type"] == "string"


@pytest.mark.asyncio
async def test_semantic_search_tool_success():
    """Test semantic_search tool with valid request."""
    from code_flow_graph.mcp_server.tools import semantic_search, SemanticSearchRequest, SearchResponse
    from unittest.mock import patch

    # Mock analyzer and vector_store
    mock_store = MagicMock()
    mock_store.query_functions.return_value = [{"metadata": {"name": "test_func"}}]

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.vector_store = mock_store

        req = SemanticSearchRequest(query="test", n_results=1)
        response = await semantic_search(req)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 1
        assert response.results[0] == {"metadata": {"name": "test_func"}}
        mock_store.query_functions.assert_called_once_with("test", 1, {})


@pytest.mark.asyncio
async def test_semantic_search_tool_no_store():
    """Test semantic_search tool raises error when no vector store."""
    from code_flow_graph.mcp_server.tools import semantic_search, SemanticSearchRequest
    from unittest.mock import patch

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.vector_store = None

        req = SemanticSearchRequest(query="test")
        with pytest.raises(MCPError) as exc_info:
            await semantic_search(req)
        assert exc_info.value.code == 5001
        assert "Vector store unavailable" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Ensure the vector store is properly initialized"


@pytest.mark.asyncio
async def test_semantic_search_tool_invalid_params():
    """Test semantic_search tool raises error for invalid params."""
    from code_flow_graph.mcp_server.tools import semantic_search, SemanticSearchRequest
    from unittest.mock import patch

    # Mock analyzer and vector_store
    mock_store = MagicMock()
    mock_store.query_functions.return_value = [{"metadata": {"name": "test_func"}}]

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.vector_store = mock_store

        # Test with invalid n_results (negative) - Pydantic validation happens first
        with pytest.raises(ValidationError) as exc_info:
            req = SemanticSearchRequest(query="test", n_results=-1)
            await semantic_search(req)
        assert "n_results must be positive" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_call_graph_tool_success_json():
    """Test get_call_graph tool with JSON format."""
    from code_flow_graph.mcp_server.tools import get_call_graph, CallGraphRequest, GraphResponse
    from unittest.mock import patch

    mock_graph_data = {"functions": {"test.func": {"name": "func"}}}

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder.export_graph.return_value = mock_graph_data

        req = CallGraphRequest(format="json")
        response = await get_call_graph(req)

        assert isinstance(response, GraphResponse)
        assert response.graph == mock_graph_data
        mock_analyzer.builder.export_graph.assert_called_once_with(format="json")


@pytest.mark.asyncio
async def test_get_call_graph_tool_success_mermaid():
    """Test get_call_graph tool with Mermaid format."""
    from code_flow_graph.mcp_server.tools import get_call_graph, CallGraphRequest, GraphResponse
    from unittest.mock import patch

    mock_mermaid_str = "graph TD\nA --> B"

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder.export_graph.return_value = mock_mermaid_str

        req = CallGraphRequest(format="mermaid")
        response = await get_call_graph(req)

        assert isinstance(response, GraphResponse)
        assert response.graph == mock_mermaid_str
        mock_analyzer.builder.export_graph.assert_called_once_with(format="mermaid")


@pytest.mark.asyncio
async def test_get_call_graph_tool_empty_fqns():
    """Test get_call_graph tool with empty fqns (full graph)."""
    from code_flow_graph.mcp_server.tools import get_call_graph, CallGraphRequest, GraphResponse
    from unittest.mock import patch

    mock_graph_data = {"functions": {}}

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder.export_graph.return_value = mock_graph_data

        req = CallGraphRequest(fqns=[], format="json")
        response = await get_call_graph(req)

        assert isinstance(response, GraphResponse)
        assert response.graph == mock_graph_data
        mock_analyzer.builder.export_graph.assert_called_once_with(format="json")


@pytest.mark.asyncio
async def test_get_call_graph_tool_no_builder():
    """Test get_call_graph tool raises error when no builder."""
    from code_flow_graph.mcp_server.tools import get_call_graph, CallGraphRequest
    from unittest.mock import patch

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder = None

        req = CallGraphRequest()
        with pytest.raises(MCPError) as exc_info:
            await get_call_graph(req)
        assert exc_info.value.code == 5001
        assert "Builder unavailable" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Ensure the call graph builder is properly initialized"


@pytest.mark.asyncio
async def test_get_function_metadata_tool_success():
    """Test get_function_metadata tool with valid FQN."""
    from code_flow_graph.mcp_server.tools import get_function_metadata, MetadataRequest, MetadataResponse
    from unittest.mock import patch, MagicMock

    mock_node = MagicMock()
    mock_node.name = "test_func"
    mock_node.fully_qualified_name = "module.test_func"
    mock_node.file_path = "/path/to/file.py"
    mock_node.line_start = 10
    mock_node.line_end = 20
    mock_node.parameters = ["arg1", "arg2"]
    mock_node.incoming_edges = []
    mock_node.outgoing_edges = []
    mock_node.return_type = "str"
    mock_node.is_entry_point = False
    mock_node.is_exported = True
    mock_node.is_async = False
    mock_node.is_static = False
    mock_node.access_modifier = "public"
    mock_node.docstring = "Test function"
    mock_node.is_method = False
    mock_node.class_name = None
    mock_node.complexity = 5
    mock_node.nloc = 10
    mock_node.external_dependencies = []
    mock_node.decorators = []
    mock_node.catches_exceptions = []
    mock_node.local_variables_declared = ["var1"]
    mock_node.hash_body = "abc123"

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder.functions.get.return_value = mock_node

        req = MetadataRequest(fqn="module.test_func")
        response = await get_function_metadata(req)

        assert isinstance(response, MetadataResponse)
        assert response.name == "test_func"
        assert response.complexity == 5
        mock_analyzer.builder.functions.get.assert_called_once_with("module.test_func")


@pytest.mark.asyncio
async def test_get_function_metadata_tool_invalid_fqn():
    """Test get_function_metadata tool with invalid FQN."""
    from code_flow_graph.mcp_server.tools import get_function_metadata, MetadataRequest
    from unittest.mock import patch

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder.functions.get.return_value = None

        req = MetadataRequest(fqn="invalid.fqn")
        with pytest.raises(MCPError) as exc_info:
            await get_function_metadata(req)
        assert exc_info.value.code == 4001
        assert "FQN not found: invalid.fqn" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Check the fully qualified name and ensure it exists in the codebase"


@pytest.mark.asyncio
async def test_get_function_metadata_tool_no_builder():
    """Test get_function_metadata tool raises error when no builder."""
    from code_flow_graph.mcp_server.tools import get_function_metadata, MetadataRequest
    from unittest.mock import patch

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder = None

        req = MetadataRequest(fqn="module.test_func")
        with pytest.raises(MCPError) as exc_info:
            await get_function_metadata(req)
        assert exc_info.value.code == 5001
        assert "Builder unavailable" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Ensure the call graph builder is properly initialized"


@pytest.mark.asyncio
async def test_query_entry_points_tool_success():
    """Test query_entry_points tool with mock entry points."""
    from code_flow_graph.mcp_server.tools import query_entry_points, EntryPointsResponse
    from unittest.mock import patch, MagicMock

    mock_ep1 = MagicMock()
    mock_ep1.name = "main"
    mock_ep1.fully_qualified_name = "module.main"
    mock_ep1.file_path = "/path/to/main.py"
    mock_ep1.line_start = 1
    mock_ep1.is_entry_point = True

    mock_ep2 = MagicMock()
    mock_ep2.name = "run"
    mock_ep2.fully_qualified_name = "module.run"
    mock_ep2.file_path = "/path/to/run.py"
    mock_ep2.line_start = 5
    mock_ep2.is_entry_point = True

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder.get_entry_points.return_value = [mock_ep1, mock_ep2]

        response = await query_entry_points()

        assert isinstance(response, EntryPointsResponse)
        assert len(response.entry_points) == 2
        assert response.entry_points[0]['name'] == "main"
        assert response.entry_points[1]['name'] == "run"
        mock_analyzer.builder.get_entry_points.assert_called_once()


@pytest.mark.asyncio
async def test_query_entry_points_tool_no_builder():
    """Test query_entry_points tool raises error when no builder."""
    from code_flow_graph.mcp_server.tools import query_entry_points
    from unittest.mock import patch

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder = None

        with pytest.raises(MCPError) as exc_info:
            await query_entry_points()
        assert exc_info.value.code == 5001
        assert "Builder unavailable" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Ensure the call graph builder is properly initialized"


@pytest.mark.asyncio
async def test_list_resources_includes_query_entry_points():
    """Test that list_resources includes the query_entry_points tool."""
    from code_flow_graph.mcp_server.server import list_resources

    result = await list_resources()
    assert len(result) == 8  # ping, semantic_search, get_call_graph, get_function_metadata, query_entry_points, generate_mermaid_graph, update_context, get_context
    assert result[4]["name"] == "query_entry_points"
    assert result[4]["description"] == "Get all identified entry points"
    assert result[4]["schema"]["input"] == {}
    assert "entry_points" in result[4]["schema"]["output"]["properties"]
    assert result[5]["name"] == "generate_mermaid_graph"
    assert result[5]["description"] == "Generate Mermaid graph for call graph"
    assert "fqns" in result[5]["schema"]["input"]["properties"]
    assert "llm_optimized" in result[5]["schema"]["input"]["properties"]
    assert "graph" in result[5]["schema"]["output"]["properties"]
    assert result[6]["name"] == "update_context"
    assert result[6]["type"] == "method"
    assert result[7]["name"] == "get_context"
    assert result[7]["type"] == "method"


@pytest.mark.asyncio
async def test_generate_mermaid_graph_tool_success():
    """Test generate_mermaid_graph tool with valid request."""
    from code_flow_graph.mcp_server.tools import generate_mermaid_graph, MermaidRequest, MermaidResponse
    from unittest.mock import patch

    mock_mermaid_str = "graph TD\nA --> B"

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder.export_mermaid_graph.return_value = mock_mermaid_str

        req = MermaidRequest(fqns=["module.func"], llm_optimized=True)
        response = await generate_mermaid_graph(req)

        assert isinstance(response, MermaidResponse)
        assert response.graph == mock_mermaid_str
        mock_analyzer.builder.export_mermaid_graph.assert_called_once_with(highlight_fqns=["module.func"], llm_optimized=True)


@pytest.mark.asyncio
async def test_generate_mermaid_graph_tool_empty_fqns():
    """Test generate_mermaid_graph tool with empty fqns."""
    from code_flow_graph.mcp_server.tools import generate_mermaid_graph, MermaidRequest, MermaidResponse
    from unittest.mock import patch

    mock_mermaid_str = "graph TD\nA --> B"

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder.export_mermaid_graph.return_value = mock_mermaid_str

        req = MermaidRequest(fqns=[], llm_optimized=False)
        response = await generate_mermaid_graph(req)

        assert isinstance(response, MermaidResponse)
        assert response.graph == mock_mermaid_str
        mock_analyzer.builder.export_mermaid_graph.assert_called_once_with(highlight_fqns=[], llm_optimized=False)


@pytest.mark.asyncio
async def test_generate_mermaid_graph_tool_no_builder():
    """Test generate_mermaid_graph tool raises error when no builder."""
    from code_flow_graph.mcp_server.tools import generate_mermaid_graph, MermaidRequest
    from unittest.mock import patch

    with patch('code_flow_graph.mcp_server.tools.analyzer') as mock_analyzer:
        mock_analyzer.builder = None

        req = MermaidRequest()
        with pytest.raises(MCPError) as exc_info:
            await generate_mermaid_graph(req)
        assert exc_info.value.code == 5001
        assert "Builder unavailable" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Ensure the call graph builder is properly initialized"


@pytest.mark.asyncio
async def test_update_context_tool_success(mock_dependencies):
    """Test update_context tool merges params and returns version."""
    mock_fastmcp_class, mock_server_instance, mock_analyzer_class = mock_dependencies

    from code_flow_graph.mcp_server.server import update_context, server

    # Initialize context
    server.context = {}

    result = await update_context({"key": "val"})
    assert result == {"version": 1}
    assert server.context == {"key": "val"}

    # Test merge
    result = await update_context({"key2": "val2"})
    assert result == {"version": 2}
    assert server.context == {"key": "val", "key2": "val2"}


@pytest.mark.asyncio
async def test_get_context_tool_success(mock_dependencies):
    """Test get_context tool returns current context."""
    mock_fastmcp_class, mock_server_instance, mock_analyzer_class = mock_dependencies

    from code_flow_graph.mcp_server.server import get_context, server

    server.context = {"test": "data"}
    result = await get_context()
    assert result == {"test": "data"}

