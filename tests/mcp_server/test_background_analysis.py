import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from code_flow_graph.mcp_server.analyzer import MCPAnalyzer, AnalysisState


@pytest.fixture
def mock_core_components():
    """Mock core components to avoid actual initialization."""
    with patch('code_flow_graph.mcp_server.analyzer.PythonASTExtractor') as mock_extractor, \
         patch('code_flow_graph.mcp_server.analyzer.CallGraphBuilder') as mock_builder, \
         patch('code_flow_graph.mcp_server.analyzer.CodeVectorStore') as mock_store:
        yield mock_extractor, mock_builder, mock_store


def test_analyzer_initial_state(mock_core_components):
    """Test that analyzer starts in NOT_STARTED state."""
    config = {
        'watch_directories': ['.'],
        'chromadb_path': './test_chroma'
    }
    
    analyzer = MCPAnalyzer(config)
    
    assert analyzer.analysis_state == AnalysisState.NOT_STARTED
    assert analyzer.analysis_task is None
    assert analyzer.analysis_error is None


def test_is_ready_before_analysis(mock_core_components):
    """Test is_ready returns False before analysis."""
    config = {
        'watch_directories': ['.'],
        'chromadb_path': './test_chroma'
    }
    
    analyzer = MCPAnalyzer(config)
    
    assert analyzer.is_ready() is False


@pytest.mark.asyncio
async def test_start_analysis_creates_task(mock_core_components):
    """Test that start_analysis creates a background task."""
    mock_extractor, mock_builder, mock_store = mock_core_components
    
    config = {
        'watch_directories': ['.'],
        'chromadb_path': './test_chroma'
    }
    
    analyzer = MCPAnalyzer(config)
    
    # Mock the analyze method to avoid actual analysis
    analyzer.analyze = AsyncMock()
    
    # Start analysis
    await analyzer.start_analysis()
    
    # Check that task was created
    assert analyzer.analysis_task is not None
    
    # Give the task a moment to start
    await asyncio.sleep(0.01)
    
    # State should be IN_PROGRESS or COMPLETED
    assert analyzer.analysis_state in [AnalysisState.IN_PROGRESS, AnalysisState.COMPLETED]


@pytest.mark.asyncio
async def test_analysis_state_transitions(mock_core_components):
    """Test that analysis state transitions correctly."""
    mock_extractor, mock_builder, mock_store = mock_core_components
    
    config = {
        'watch_directories': ['.'],
        'chromadb_path': './test_chroma'
    }
    
    analyzer = MCPAnalyzer(config)
    
    # Mock the analyze method to complete quickly
    async def mock_analyze():
        pass
    
    analyzer.analyze = mock_analyze
    
    # Initial state
    assert analyzer.analysis_state == AnalysisState.NOT_STARTED
    
    # Start analysis
    await analyzer.start_analysis()
    
    # Wait for task to complete
    if analyzer.analysis_task:
        await analyzer.analysis_task
    
    # Should be completed
    assert analyzer.analysis_state == AnalysisState.COMPLETED


@pytest.mark.asyncio
async def test_analysis_failure_handling(mock_core_components):
    """Test that analysis failures are handled correctly."""
    mock_extractor, mock_builder, mock_store = mock_core_components
    
    config = {
        'watch_directories': ['.'],
        'chromadb_path': './test_chroma'
    }
    
    analyzer = MCPAnalyzer(config)
    
    # Mock the analyze method to raise an error
    async def mock_analyze():
        raise ValueError("Test error")
    
    analyzer.analyze = mock_analyze
    
    # Start analysis
    await analyzer.start_analysis()
    
    # Wait for task to complete
    if analyzer.analysis_task:
        await analyzer.analysis_task
    
    # Should be in FAILED state
    assert analyzer.analysis_state == AnalysisState.FAILED
    assert analyzer.analysis_error is not None
    assert isinstance(analyzer.analysis_error, ValueError)


@pytest.mark.asyncio
async def test_wait_for_analysis(mock_core_components):
    """Test wait_for_analysis method."""
    mock_extractor, mock_builder, mock_store = mock_core_components
    
    config = {
        'watch_directories': ['.'],
        'chromadb_path': './test_chroma'
    }
    
    analyzer = MCPAnalyzer(config)
    
    # Mock the analyze method
    async def mock_analyze():
        await asyncio.sleep(0.1)  # Simulate some work
    
    analyzer.analyze = mock_analyze
    
    # Start analysis
    await analyzer.start_analysis()
    
    # Wait for analysis to complete
    result = await analyzer.wait_for_analysis(timeout=1.0)
    
    assert result is True
    assert analyzer.is_ready() is True


@pytest.mark.asyncio
async def test_ping_tool_includes_status():
    """Test that ping tool includes analysis status."""
    from code_flow_graph.mcp_server.server import ping_tool, server
    
    # Mock analyzer with IN_PROGRESS state
    mock_analyzer = MagicMock()
    mock_analyzer.analysis_state = AnalysisState.IN_PROGRESS
    mock_analyzer.builder.functions = {'test.func': MagicMock()}
    
    server.analyzer = mock_analyzer
    
    try:
        response = await ping_tool(message="test")
        
        assert response.status == "ok"
        assert response.echoed == "test"
        assert response.analysis_status == "in_progress"
        assert response.indexed_functions == 1
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_semantic_search_includes_status():
    """Test that semantic_search includes analysis status."""
    from code_flow_graph.mcp_server.server import semantic_search, server
    
    # Mock analyzer with COMPLETED state
    mock_store = MagicMock()
    mock_store.query_functions.return_value = [{"metadata": {"name": "test_func"}}]
    mock_analyzer = MagicMock()
    mock_analyzer.vector_store = mock_store
    mock_analyzer.analysis_state = AnalysisState.COMPLETED
    
    server.analyzer = mock_analyzer
    
    try:
        response = await semantic_search(query="test", n_results=1, filters={}, format="json")
        
        assert response.analysis_status == "completed"
        assert len(response.results) == 1
    finally:
        if hasattr(server, 'analyzer'):
            delattr(server, 'analyzer')


@pytest.mark.asyncio
async def test_start_analysis_only_once(mock_core_components):
    """Test that start_analysis warns if called multiple times."""
    mock_extractor, mock_builder, mock_store = mock_core_components
    
    config = {
        'watch_directories': ['.'],
        'chromadb_path': './test_chroma'
    }
    
    analyzer = MCPAnalyzer(config)
    
    # Mock analyze to take some time
    async def slow_analyze():
        await asyncio.sleep(0.1)
    
    analyzer.analyze = slow_analyze
    
    # Start analysis first time
    await analyzer.start_analysis()
    first_task = analyzer.analysis_task
    
    # Give it a moment to start
    await asyncio.sleep(0.01)
    
    # Try to start again - should warn and not create new task
    await analyzer.start_analysis()
    second_task = analyzer.analysis_task
    
    # Should be the same task (second call should be ignored)
    assert first_task is second_task
    
    # Clean up
    if analyzer.analysis_task:
        await analyzer.analysis_task
