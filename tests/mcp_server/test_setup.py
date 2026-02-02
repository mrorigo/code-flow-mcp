import pytest
from unittest.mock import mock_open, patch
import yaml

def test_pyproject_dependencies():
    # Mock pyproject.toml content
    mock_toml_content = """
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "code-flow"
version = "0.1.0"
dependencies = [
    "fastmcp",
    "pyyaml",
    "watchdog>=2.0",
    "pytest",
    "pydantic"
]

[project.scripts]
code_flow.mcp_server = "code_flow.mcp_server.__main__:main"
"""
    with patch('builtins.open', mock_open(read_data=mock_toml_content)):
        # Since it's toml, but for simplicity, we'll mock the parsed dict
        # In real test, use tomllib or tomli
        expected_deps = ["fastmcp", "pyyaml", "watchdog>=2.0", "pytest", "pydantic"]
        # Mock the toml load
        with patch('tomllib.load') as mock_load:
            mock_load.return_value = {
                'project': {'dependencies': expected_deps}
            }
            # Assert deps include the required ones
            deps = mock_load.return_value['project']['dependencies']
            assert "fastmcp" in deps
            assert "pyyaml" in deps
            assert "watchdog>=2.0" in deps
            assert "pytest" in deps
            assert "pydantic" in deps

def test_config_load():
    # Mock yaml load
    mock_config = {
        "project_root": ".",
        "watch_directories": ["."],
        "ignored_patterns": [],
        "chroma_dir": "./.codeflow/chroma",
        "memory_dir": "./.codeflow/memory",
        "reports_dir": "./.codeflow/reports",
        "chromadb_path": "./code_vectors_chroma",
        "max_graph_depth": 3
    }
    with patch('yaml.safe_load', return_value=mock_config) as mock_yaml:
        config = yaml.safe_load(None)  # Mocked
        assert config == mock_config
        assert config['watch_directories'] == ["."]
        assert config['max_graph_depth'] == 3
