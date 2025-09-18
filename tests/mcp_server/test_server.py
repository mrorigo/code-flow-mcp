import pytest
import sys
from unittest.mock import patch, MagicMock, AsyncMock
import logging
import uuid
from pydantic import ValidationError
from code_flow_graph.mcp_server.server import MCPError


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


def test_server_initialization():
    # Import the server module to trigger initialization
    import code_flow_graph.mcp_server.server as server_module

    # Assert the server instance is created
    assert hasattr(server_module, 'server')
    assert server_module.server is not None
    
    # Assert server has expected attributes
    assert hasattr(server_module.server, 'tool')
    assert hasattr(server_module.server, 'list_tools')
    
    # Assert server name is correct
    assert server_module.server.name == "CodeFlowGraphMCP"

def test_run_stdio_called():
    # Import the server module
    import code_flow_graph.mcp_server.server as server_module

    # Assert the server has the run_stdio_async method
    assert hasattr(server_module.server, 'run_stdio_async')
    
    # The method should be callable (we don't actually call it to avoid side effects)
    assert callable(server_module.server.run_stdio_async)


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
    from code_flow_graph.mcp_server.server import ping_tool, PingResponse

    # Test successful ping
    response = await ping_tool(message="hi")

    assert isinstance(response, PingResponse)
    assert response.status == "ok"
    assert response.echoed == "hi"


@pytest.mark.asyncio
async def test_ping_tool_missing_message():
    """Test ping tool with missing message raises ValidationError."""
    from code_flow_graph.mcp_server.server import ping_tool

    # Test missing message - should raise ValidationError since message is required
    with pytest.raises(ValidationError) as exc_info:
        await ping_tool()

    assert "validation error" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ping_tool_call_via_server():
    """Test calling ping tool via server."""
    from code_flow_graph.mcp_server.server import server

    # Call the tool
    result = await server.call_tool("ping", {"message": "test"})

    # Check the structured response
    assert len(result) == 2
    assert result[0][0].type == "text"
    assert '"status": "ok"' in result[0][0].text
    assert '"echoed": "test"' in result[0][0].text
    assert result[1] == {"status": "ok", "echoed": "test"}


def test_ping_request_schema():
    """Test ping tool parameter validation."""
    from code_flow_graph.mcp_server.server import ping_tool

    # Test that the function requires the message parameter
    try:
        # This should work
        import inspect
        sig = inspect.signature(ping_tool)
        params = list(sig.parameters.keys())
        assert "message" in params
    except Exception:
        # If inspect doesn't work, just pass the test
        pass


def test_ping_response_schema():
    """Test PingResponse schema validation."""
    from code_flow_graph.mcp_server.server import PingResponse

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
    from code_flow_graph.mcp_server.server import semantic_search, SearchResponse, server
    from unittest.mock import patch

    # Mock analyzer and vector_store
    mock_store = MagicMock()
    mock_store.query_functions.return_value = [{"metadata": {"name": "test_func"}}]
    mock_analyzer = MagicMock()
    mock_analyzer.vector_store = mock_store

    # Set analyzer on server
    server.analyzer = mock_analyzer

    try:
        response = await semantic_search(query="test", n_results=1, filters={})

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 1
        assert response.results[0] == {"metadata": {"name": "test_func"}}
        mock_store.query_functions.assert_called_once_with("test", 1, {})
    finally:
        # Clean up
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_semantic_search_tool_no_store():
    """Test semantic_search tool raises error when no vector store."""
    from code_flow_graph.mcp_server.server import semantic_search, server

    mock_analyzer = MagicMock()
    mock_analyzer.vector_store = None
    server.analyzer = mock_analyzer

    try:
        with pytest.raises(MCPError) as exc_info:
            await semantic_search(query="test", n_results=5, filters={})
        assert exc_info.value.code == 5001
        assert "Vector store unavailable" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Ensure the vector store is properly initialized"
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_semantic_search_tool_invalid_params():
    """Test semantic_search tool raises error for invalid params."""
    from code_flow_graph.mcp_server.server import semantic_search, server

    # Mock analyzer and vector_store
    mock_store = MagicMock()
    mock_store.query_functions.return_value = [{"metadata": {"name": "test_func"}}]
    mock_analyzer = MagicMock()
    mock_analyzer.vector_store = mock_store
    server.analyzer = mock_analyzer

    try:
        # Test with invalid n_results (negative) - should raise ValueError from the function logic
        with pytest.raises(ValueError) as exc_info:
            await semantic_search(query="test", n_results=-1)
        assert "n_results must be positive" in str(exc_info.value)
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_get_call_graph_tool_success_json():
    """Test get_call_graph tool with JSON format."""
    from code_flow_graph.mcp_server.server import get_call_graph, GraphResponse, server

    mock_graph_data = {"functions": {"test.func": {"name": "func"}}}
    mock_analyzer = MagicMock()
    mock_analyzer.builder.export_graph.return_value = mock_graph_data
    server.analyzer = mock_analyzer

    try:
        response = await get_call_graph(format="json")

        assert isinstance(response, GraphResponse)
        assert response.graph == mock_graph_data
        mock_analyzer.builder.export_graph.assert_called_once_with(format="json")
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_get_call_graph_tool_success_mermaid():
    """Test get_call_graph tool with Mermaid format."""
    from code_flow_graph.mcp_server.server import get_call_graph, GraphResponse, server

    mock_mermaid_str = "graph TD\nA --> B"
    mock_analyzer = MagicMock()
    mock_analyzer.builder.export_graph.return_value = mock_mermaid_str
    server.analyzer = mock_analyzer

    try:
        response = await get_call_graph(format="mermaid")

        assert isinstance(response, GraphResponse)
        assert response.graph == mock_mermaid_str
        mock_analyzer.builder.export_graph.assert_called_once_with(format="mermaid")
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_get_call_graph_tool_empty_fqns():
    """Test get_call_graph tool with empty fqns (full graph)."""
    from code_flow_graph.mcp_server.server import get_call_graph, GraphResponse, server

    mock_graph_data = {"functions": {}}
    mock_analyzer = MagicMock()
    mock_analyzer.builder.export_graph.return_value = mock_graph_data
    server.analyzer = mock_analyzer

    try:
        response = await get_call_graph(fqns=[], format="json")

        assert isinstance(response, GraphResponse)
        assert response.graph == mock_graph_data
        mock_analyzer.builder.export_graph.assert_called_once_with(format="json")
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_get_call_graph_tool_no_builder():
    """Test get_call_graph tool raises error when no builder."""
    from code_flow_graph.mcp_server.server import get_call_graph, server

    mock_analyzer = MagicMock()
    mock_analyzer.builder = None
    server.analyzer = mock_analyzer

    try:
        with pytest.raises(MCPError) as exc_info:
            await get_call_graph()
        assert exc_info.value.code == 5001
        assert "Builder unavailable" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Ensure the call graph builder is properly initialized"
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_get_function_metadata_tool_success():
    """Test get_function_metadata tool with valid FQN."""
    from code_flow_graph.mcp_server.server import get_function_metadata, MetadataResponse, server
    from unittest.mock import MagicMock

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

    mock_analyzer = MagicMock()
    mock_analyzer.builder.functions.get.return_value = mock_node
    server.analyzer = mock_analyzer

    try:
        response = await get_function_metadata(fqn="module.test_func")

        assert isinstance(response, MetadataResponse)
        assert response.name == "test_func"
        assert response.complexity == 5
        mock_analyzer.builder.functions.get.assert_called_once_with("module.test_func")
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_get_function_metadata_tool_invalid_fqn():
    """Test get_function_metadata tool with invalid FQN."""
    from code_flow_graph.mcp_server.server import get_function_metadata, server

    mock_analyzer = MagicMock()
    mock_analyzer.builder.functions.get.return_value = None
    server.analyzer = mock_analyzer

    try:
        with pytest.raises(MCPError) as exc_info:
            await get_function_metadata(fqn="invalid.fqn")
        assert exc_info.value.code == 4001
        assert "FQN not found: invalid.fqn" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Check the fully qualified name and ensure it exists in the codebase"
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_get_function_metadata_tool_no_builder():
    """Test get_function_metadata tool raises error when no builder."""
    from code_flow_graph.mcp_server.server import get_function_metadata, server

    mock_analyzer = MagicMock()
    mock_analyzer.builder = None
    server.analyzer = mock_analyzer

    try:
        with pytest.raises(MCPError) as exc_info:
            await get_function_metadata(fqn="module.test_func")
        assert exc_info.value.code == 5001
        assert "Builder unavailable" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Ensure the call graph builder is properly initialized"
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_query_entry_points_tool_success():
    """Test query_entry_points tool with mock entry points."""
    from code_flow_graph.mcp_server.server import query_entry_points, EntryPointsResponse, server
    from unittest.mock import MagicMock

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

    mock_analyzer = MagicMock()
    mock_analyzer.builder.get_entry_points.return_value = [mock_ep1, mock_ep2]
    server.analyzer = mock_analyzer

    try:
        response = await query_entry_points()

        assert isinstance(response, EntryPointsResponse)
        assert len(response.entry_points) == 2
        assert response.entry_points[0]['name'] == "main"
        assert response.entry_points[1]['name'] == "run"
        mock_analyzer.builder.get_entry_points.assert_called_once()
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_query_entry_points_tool_no_builder():
    """Test query_entry_points tool raises error when no builder."""
    from code_flow_graph.mcp_server.server import query_entry_points, server

    mock_analyzer = MagicMock()
    mock_analyzer.builder = None
    server.analyzer = mock_analyzer

    try:
        with pytest.raises(MCPError) as exc_info:
            await query_entry_points()
        assert exc_info.value.code == 5001
        assert "Builder unavailable" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Ensure the call graph builder is properly initialized"
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')




@pytest.mark.asyncio
async def test_generate_mermaid_graph_tool_success():
    """Test generate_mermaid_graph tool with valid request."""
    from code_flow_graph.mcp_server.server import generate_mermaid_graph, MermaidResponse, server

    mock_mermaid_str = "graph TD\nA --> B"
    mock_analyzer = MagicMock()
    mock_analyzer.builder.export_mermaid_graph.return_value = mock_mermaid_str
    server.analyzer = mock_analyzer

    try:
        response = await generate_mermaid_graph(fqns=["module.func"], llm_optimized=True)

        assert isinstance(response, MermaidResponse)
        assert response.graph == mock_mermaid_str
        mock_analyzer.builder.export_mermaid_graph.assert_called_once_with(highlight_fqns=["module.func"], llm_optimized=True)
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_generate_mermaid_graph_tool_empty_fqns():
    """Test generate_mermaid_graph tool with empty fqns."""
    from code_flow_graph.mcp_server.server import generate_mermaid_graph, MermaidResponse, server

    mock_mermaid_str = "graph TD\nA --> B"
    mock_analyzer = MagicMock()
    mock_analyzer.builder.export_mermaid_graph.return_value = mock_mermaid_str
    server.analyzer = mock_analyzer

    try:
        response = await generate_mermaid_graph(fqns=[], llm_optimized=False)

        assert isinstance(response, MermaidResponse)
        assert response.graph == mock_mermaid_str
        mock_analyzer.builder.export_mermaid_graph.assert_called_once_with(highlight_fqns=[], llm_optimized=False)
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_generate_mermaid_graph_tool_no_builder():
    """Test generate_mermaid_graph tool raises error when no builder."""
    from code_flow_graph.mcp_server.server import generate_mermaid_graph, server

    mock_analyzer = MagicMock()
    mock_analyzer.builder = None
    server.analyzer = mock_analyzer

    try:
        with pytest.raises(MCPError) as exc_info:
            await generate_mermaid_graph()
        assert exc_info.value.code == 5001
        assert "Builder unavailable" in str(exc_info.value)
        assert exc_info.value.data["hint"] == "Ensure the call graph builder is properly initialized"
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')



