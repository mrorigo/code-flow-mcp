import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import asyncio
from code_flow_graph.mcp_server.analyzer import MCPAnalyzer
from code_flow_graph.core.models import StructuredDataElement

@pytest.fixture
def mock_dependencies():
    with patch('code_flow_graph.mcp_server.analyzer.TreeSitterPythonExtractor') as mock_py_ext, \
         patch('code_flow_graph.mcp_server.analyzer.TreeSitterTypeScriptExtractor') as mock_ts_ext, \
         patch('code_flow_graph.mcp_server.analyzer.CallGraphBuilder') as mock_builder, \
         patch('code_flow_graph.mcp_server.analyzer.CodeVectorStore') as mock_store, \
         patch('code_flow_graph.mcp_server.analyzer.StructuredDataExtractor') as mock_struct_ext:
        yield mock_py_ext, mock_ts_ext, mock_builder, mock_store, mock_struct_ext

@pytest.mark.asyncio
async def test_analyze_with_structured_data(mock_dependencies):
    mock_py_ext, mock_ts_ext, mock_builder, mock_store, mock_struct_ext = mock_dependencies
    
    config = {
        'watch_directories': ['.'],
        'chromadb_path': './test_chroma',
        'ignored_filenames': ['ignore.json']
    }
    
    # Mock structured extractor
    mock_struct_instance = mock_struct_ext.return_value
    mock_struct_instance.extract_from_directory.return_value = [
        StructuredDataElement(
            name='key', kind='structured_data', file_path='config.json',
            line_start=1, line_end=1, full_source='{"key": "value"}',
            json_path='key', value_type='str', key_name='key', content='key: value'
        )
    ]
    
    # Mock vector store
    mock_store_instance = mock_store.return_value
    # Mock get_all_nodes to return empty list to avoid iteration error in _populate_vector_store fallback
    mock_store_instance.get_all_nodes.return_value = []
    
    with patch('pathlib.Path.exists', return_value=True):
        analyzer = MCPAnalyzer(config)
    
    # Assert StructuredDataExtractor init with ignored filenames
    mock_struct_ext.assert_called_with(ignored_filenames={'ignore.json'})
    
    # Inject the mock store instance into the analyzer since we mocked the class but init might have created a new mock
    analyzer.vector_store = mock_store_instance
    
    await analyzer.analyze()
    
    # Assert extraction called
    mock_struct_instance.extract_from_directory.assert_called()
    
    # Assert added to vector store
    mock_store_instance.add_structured_elements_batch.assert_called()

@pytest.mark.asyncio
async def test_incremental_update_structured(mock_dependencies):
    mock_py_ext, mock_ts_ext, mock_builder, mock_store, mock_struct_ext = mock_dependencies
    
    config = {'watch_directories': ['.'], 'chromadb_path': './test_chroma'}
    
    with patch('pathlib.Path.exists', return_value=True):
        analyzer = MCPAnalyzer(config)
    
    analyzer.loop = asyncio.get_running_loop()
    
    # Ensure vector store is set to our mock
    mock_store_instance = mock_store.return_value
    analyzer.vector_store = mock_store_instance
    
    mock_struct_instance = mock_struct_ext.return_value
    mock_struct_instance.extract_from_file.return_value = [
        StructuredDataElement(
            name='key', kind='structured_data', file_path='config.json',
            line_start=1, line_end=1, full_source='{"key": "value"}',
            json_path='key', value_type='str', key_name='key', content='key: value'
        )
    ]
    
    mock_store_instance = mock_store.return_value
    
    # Trigger incremental update for a json file
    await analyzer._incremental_update('config.json')
    
    mock_struct_instance.extract_from_file.assert_called_with(Path('config.json'))
    mock_store_instance.add_structured_elements_batch.assert_called()
    
    # Trigger for python file (should not call structured extractor)
    mock_struct_instance.reset_mock()
    await analyzer._incremental_update('script.py')
    mock_struct_instance.extract_from_file.assert_not_called()
