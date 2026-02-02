import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import asyncio
from code_flow.core.models import FunctionElement
from code_flow.mcp_server.analyzer import MCPAnalyzer


@pytest.fixture
def mock_core_components():
    """Mock core components to avoid actual initialization."""
    with patch('code_flow.mcp_server.analyzer.TreeSitterPythonExtractor') as mock_extractor, \
         patch('code_flow.mcp_server.analyzer.CallGraphBuilder') as mock_builder, \
         patch('code_flow.mcp_server.analyzer.CodeVectorStore') as mock_store:
        yield mock_extractor, mock_builder, mock_store


@pytest.fixture
def mock_core_components_rust():
    """Mock core components for Rust analyzer initialization."""
    with patch('code_flow.mcp_server.analyzer.TreeSitterRustExtractor') as mock_extractor, \
         patch('code_flow.mcp_server.analyzer.CallGraphBuilder') as mock_builder, \
         patch('code_flow.mcp_server.analyzer.CodeVectorStore') as mock_store:
        yield mock_extractor, mock_builder, mock_store


def test_mcp_analyzer_init(mock_core_components):
    """Test MCPAnalyzer initialization with config dict."""
    mock_extractor, mock_builder, mock_store = mock_core_components

    config = {
        'project_root': '.',
        'watch_directories': ['.'],
        'chroma_dir': './.codeflow/chroma',
        'memory_dir': './.codeflow/memory',
        'reports_dir': './.codeflow/reports',
        'chromadb_path': './test_chroma'
    }

    analyzer = MCPAnalyzer(config)

    assert analyzer.config == config
    assert analyzer.extractor == mock_extractor.return_value
    assert analyzer.builder == mock_builder.return_value
    assert analyzer.vector_store is None  # Since path doesn't exist


def test_mcp_analyzer_init_rust(mock_core_components_rust):
    """Test MCPAnalyzer initialization for Rust."""
    mock_extractor, mock_builder, _ = mock_core_components_rust
    config = {
        'project_root': '.',
        'watch_directories': ['.'],
        'chroma_dir': './.codeflow/chroma',
        'memory_dir': './.codeflow/memory',
        'reports_dir': './.codeflow/reports',
        'chromadb_path': './test_chroma',
        'language': 'rust'
    }

    analyzer = MCPAnalyzer(config)

    assert analyzer.config == config
    assert analyzer.extractor == mock_extractor.return_value
    assert analyzer.builder == mock_builder.return_value
    assert analyzer.vector_store is None


def test_mcp_analyzer_init_with_existing_store():
    """Test MCPAnalyzer initialization when ChromaDB path exists."""
    config = {
        'project_root': '.',
        'watch_directories': ['.'],
        'chroma_dir': './.codeflow/chroma',
        'memory_dir': './.codeflow/memory',
        'reports_dir': './.codeflow/reports',
        'chromadb_path': './code_vectors_chroma',  # legacy field
        'embedding_model': 'all-MiniLM-L6-v2'
    }

    with patch('code_flow.mcp_server.analyzer.Path.exists', return_value=True):
        with patch('code_flow.mcp_server.analyzer.CodeVectorStore') as mock_store:
            analyzer = MCPAnalyzer(config)
            mock_store.assert_called_once_with(
                persist_directory='./.codeflow/chroma',
                embedding_model_name='all-MiniLM-L6-v2',
                max_tokens=256
            )
            assert analyzer.vector_store is not None


@pytest.mark.asyncio
async def test_analyze(mock_core_components):
    """Test analyze method with mocked core components."""
    mock_extractor, mock_builder, mock_store = mock_core_components

    config = {
        'project_root': '.',
        'watch_directories': ['.'],
        'chroma_dir': './.codeflow/chroma',
        'memory_dir': './.codeflow/memory',
        'reports_dir': './.codeflow/reports',
        'chromadb_path': './test_chroma'
    }

    # Mock elements
    mock_elements = [
        FunctionElement(
            name='test_func',
            kind='function',
            file_path='test.py',
            line_start=1,
            line_end=5,
            full_source='def test_func(): pass',
            parameters=[],
            return_type=None,
            is_async=False,
            docstring=None,
            is_method=False,
            class_name=None
        )
    ]

    analyzer = MCPAnalyzer(config)

    # Mock the extractor
    analyzer.extractor.extract_from_directory = MagicMock(return_value=mock_elements)

    # Mock builder
    analyzer.builder.build_from_elements = MagicMock()
    analyzer.builder.functions = {'test.test_func': MagicMock()}

    # Mock vector store if present
    if analyzer.vector_store:
        analyzer.vector_store.add_function_node = MagicMock()
        analyzer.vector_store.add_edge = MagicMock()

    await analyzer.analyze()

    # Verify calls
    analyzer.extractor.extract_from_directory.assert_called_once()
    analyzer.builder.build_from_elements.assert_called_once_with(mock_elements)

    # If vector store exists, check populate was called
    if analyzer.vector_store:
        analyzer.vector_store.add_function_node.assert_called()
        analyzer.vector_store.add_edge.assert_called()


@pytest.mark.asyncio
async def test_analyze_no_vector_store(mock_core_components):
    """Test analyze when vector store is None."""
    mock_extractor, mock_builder, mock_store = mock_core_components

    config = {
        'project_root': '.',
        'watch_directories': ['.'],
        'chroma_dir': './.codeflow/chroma',
        'memory_dir': './.codeflow/memory',
        'reports_dir': './.codeflow/reports',
        'chromadb_path': './nonexistent'
    }

    analyzer = MCPAnalyzer(config)
    assert analyzer.vector_store is None

    # Mock extractor and builder
    analyzer.extractor.extract_from_directory = MagicMock(return_value=[])
    analyzer.builder.build_from_elements = MagicMock()

    # Should not raise error
    await analyzer.analyze()

    analyzer.extractor.extract_from_directory.assert_called_once()
    analyzer.builder.build_from_elements.assert_called_once()


@pytest.mark.asyncio
async def test_populate_vector_store(mock_core_components):
    """Test _populate_vector_store method."""
    mock_extractor, mock_builder, mock_store = mock_core_components

    config = {
        'project_root': '.',
        'watch_directories': ['.'],
        'chroma_dir': './.codeflow/chroma',
        'memory_dir': './.codeflow/memory',
        'reports_dir': './.codeflow/reports',
        'chromadb_path': './code_vectors_chroma',
        'embedding_model': 'all-MiniLM-L6-v2'
    }

    # Mock the store initialization
    mock_store_instance = MagicMock()
    mock_store.return_value = mock_store_instance

    with patch('code_flow.mcp_server.analyzer.Path.exists', return_value=True):
        analyzer = MCPAnalyzer(config)

    # Mock builder with functions and edges
    mock_func_node = MagicMock()
    mock_func_node.file_path = 'test.py'
    analyzer.builder.functions = {'test.func': mock_func_node}
    analyzer.builder.edges = [MagicMock()]

    # Mock file reading
    with patch('builtins.open', MagicMock()) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = 'def func(): pass'
        mock_open.return_value.__enter__.return_value = mock_file

        analyzer._populate_vector_store()

        mock_store_instance.add_function_nodes_batch.assert_called_once()
        mock_store_instance.add_edges_batch.assert_called_once()


@pytest.mark.asyncio
async def test_watcher_handler_on_modified(mock_core_components):
    """Test WatcherHandler on_modified triggers incremental update."""
    mock_extractor, mock_builder, mock_store = mock_core_components

    config = {
        'project_root': '.',
        'watch_directories': ['.'],
        'chroma_dir': './.codeflow/chroma',
        'memory_dir': './.codeflow/memory',
        'reports_dir': './.codeflow/reports',
        'chromadb_path': './test_chroma'
    }

    analyzer = MCPAnalyzer(config)
    analyzer.loop = asyncio.get_running_loop()

    # Mock the extractor for incremental update
    mock_elements = [
        FunctionElement(
            name='new_func',
            kind='function',
            file_path='modified.py',
            line_start=1,
            line_end=5,
            full_source='def new_func(): pass',
            parameters=[],
            return_type=None,
            is_async=False,
            docstring=None,
            is_method=False,
            class_name=None
        )
    ]
    # Set fqn for the element
    mock_elements[0].fqn = 'new_func'
    analyzer.extractor.extract_from_file = MagicMock(return_value=mock_elements)
    analyzer.builder.functions = {}  # Empty initially

    # Mock vector store
    analyzer.vector_store = MagicMock()
    analyzer.vector_store.get_all_nodes.return_value = []

    # Create handler
    from code_flow.mcp_server.analyzer import WatcherHandler
    handler = WatcherHandler(analyzer=analyzer)

    # Mock event
    mock_event = MagicMock()
    mock_event.src_path = 'test.py'
    mock_event.is_directory = False

    # Mock file reading
    with patch('builtins.open', MagicMock()) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = 'def new_func(): pass'
        mock_open.return_value.__enter__.return_value = mock_file

        # Call on_modified
        handler.on_modified(mock_event)

        # Wait for the task to complete
        await asyncio.sleep(2)  # Since debounce sleeps 1s

        # Assert _incremental_update was called (via the task)
        analyzer.extractor.extract_from_file.assert_called_once_with(Path('test.py'))
        # Check that new function was added
        assert 'new_func' in analyzer.builder.functions
        # Since vector store is mocked, check add_function_node was called
        analyzer.vector_store.add_function_node.assert_called_once()
