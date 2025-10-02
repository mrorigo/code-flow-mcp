"""
Unit tests for TypeScriptASTVisitor functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from code_flow_graph.core.typescript_extractor import TypeScriptASTVisitor, TypeScriptASTExtractor


class TestTypeScriptASTVisitor:
    """Test cases for TypeScriptASTVisitor."""

    def test_visitor_initialization(self, typescript_visitor):
        """Test TypeScriptASTVisitor initializes correctly."""
        assert typescript_visitor.elements == []
        assert typescript_visitor.current_class is None
        assert typescript_visitor.current_file == ""
        assert typescript_visitor.source_lines == []
        # Note: Uses regex-based parsing only for optimal performance



    def test_visitor_regex_parsing_features(self, typescript_visitor):
        """Test that regex-based parsing features are available."""
        assert typescript_visitor.elements == []
        assert typescript_visitor.current_class is None
        assert typescript_visitor.current_file == ""
        assert typescript_visitor.source_lines == []
        # Test that performance metrics are available for tracking
        assert hasattr(typescript_visitor, '_update_tsx_metrics')
        assert hasattr(typescript_visitor, '_update_ts_metrics')



    def test_visit_file_basic_function(self, typescript_visitor, temp_dir):
        """Test visiting a file with a basic function."""
        ts_file = temp_dir / "test.ts"
        source = """
        function greetUser(name: string): string {
            return `Hello, ${name}!`;
        }
        """

        with open(ts_file, 'w') as f:
            f.write(source)

        elements = typescript_visitor.visit_file(str(ts_file), source)

        assert len(elements) >= 1
        func_element = next((e for e in elements if e.name == 'greetUser'), None)
        assert func_element is not None
        assert func_element.kind == 'function'
        assert 'name: string' in func_element.parameters
        assert func_element.return_type == 'string'

    def test_visit_file_with_class(self, typescript_visitor, temp_dir):
        """Test visiting a file with a class."""
        ts_file = temp_dir / "test.ts"
        source = """
        class UserService {
            private users: User[] = [];

            public getUser(id: number): User | null {
                return this.users.find(u => u.id === id) || null;
            }
        }
        """

        with open(ts_file, 'w') as f:
            f.write(source)

        elements = typescript_visitor.visit_file(str(ts_file), source)

        assert len(elements) >= 2  # Class + method
        class_element = next((e for e in elements if e.name == 'UserService'), None)
        assert class_element is not None
        assert class_element.kind == 'class'

        func_element = next((e for e in elements if e.name == 'getUser'), None)
        assert func_element is not None
        assert func_element.is_method is True
        assert func_element.class_name == 'UserService'

    def test_visit_file_with_interface(self, typescript_visitor, temp_dir):
        """Test visiting a file with an interface."""
        ts_file = temp_dir / "test.ts"
        source = """
        interface User {
            id: number;
            name: string;
            email: string;
        }
        """

        with open(ts_file, 'w') as f:
            f.write(source)

        elements = typescript_visitor.visit_file(str(ts_file), source)

        interface_element = next((e for e in elements if e.name == 'User'), None)
        assert interface_element is not None
        assert interface_element.kind == 'interface'
        assert interface_element.metadata.get('typescript_kind') == 'interface'


class TestTypeScriptASTExtractor:
    """Test cases for TypeScriptASTExtractor."""

    def test_extractor_initialization(self, typescript_extractor):
        """Test TypeScriptASTExtractor initializes correctly."""
        assert isinstance(typescript_extractor.visitor, TypeScriptASTVisitor)
        assert typescript_extractor.project_root is None
        assert typescript_extractor.tsconfig is None
        assert typescript_extractor.project_references == []
        assert typescript_extractor.path_mappings == {}

    def test_find_tsconfig_in_directory(self, typescript_extractor, temp_dir):
        """Test tsconfig.json file detection."""
        # Create tsconfig.json
        tsconfig = temp_dir / "tsconfig.json"
        tsconfig.write_text('{"compilerOptions": {"target": "ES2020"}}')

        result = typescript_extractor._find_tsconfig(temp_dir)
        assert result == tsconfig

    def test_find_tsconfig_in_parent_directory(self, typescript_extractor, temp_dir):
        """Test tsconfig.json detection in parent directories."""
        # Create nested directory structure
        subdir = temp_dir / "src" / "components"
        subdir.mkdir(parents=True)

        # Create tsconfig.json in root
        tsconfig = temp_dir / "tsconfig.json"
        tsconfig.write_text('{"compilerOptions": {"target": "ES2020"}}')

        result = typescript_extractor._find_tsconfig(subdir)
        assert result == tsconfig

    def test_find_tsconfig_not_found(self, typescript_extractor, temp_dir):
        """Test tsconfig.json detection when file doesn't exist."""
        result = typescript_extractor._find_tsconfig(temp_dir)
        assert result is None

    def test_parse_tsconfig_valid(self, typescript_extractor, temp_dir):
        """Test parsing of valid tsconfig.json."""
        tsconfig = temp_dir / "tsconfig.json"
        config_content = {
            "compilerOptions": {
                "target": "ES2020",
                "strict": True
            },
            "include": ["src/**/*"]
        }
        import json
        tsconfig.write_text(json.dumps(config_content))

        result = typescript_extractor._parse_tsconfig(tsconfig)
        assert result["compilerOptions"]["target"] == "ES2020"
        assert result["compilerOptions"]["strict"] is True
        assert "include" in result

    def test_parse_tsconfig_invalid_json(self, typescript_extractor, temp_dir):
        """Test parsing of invalid tsconfig.json."""
        tsconfig = temp_dir / "tsconfig.json"
        tsconfig.write_text('{"invalid": json}')

        result = typescript_extractor._parse_tsconfig(tsconfig)
        assert result == {}

    def test_setup_project_integration(self, typescript_extractor, temp_dir):
        """Test project integration setup."""
        # Create tsconfig.json
        tsconfig = temp_dir / "tsconfig.json"
        config_content = {
            "compilerOptions": {
                "target": "ES2020",
                "baseUrl": "./src",
                "paths": {
                    "@/*": ["*"],
                    "@/components/*": ["components/*"]
                }
            },
            "references": [
                {"path": "./packages/core"},
                {"path": "./packages/ui"}
            ]
        }
        tsconfig.write_text(str(config_content).replace("'", '"'))

        typescript_extractor._setup_project_integration(temp_dir)

        assert typescript_extractor.tsconfig is not None
        assert typescript_extractor.project_references == config_content["references"]
        assert typescript_extractor.path_mappings["baseUrl"] == "./src"
        assert typescript_extractor.path_mappings["paths"] == config_content["compilerOptions"]["paths"]

    def test_resolve_path_mapping(self, typescript_extractor, temp_dir):
        """Test path mapping resolution."""
        typescript_extractor.path_mappings = {
            "baseUrl": "./src",
            "paths": {
                "@/*": ["*"],
                "@/components/*": ["components/*"]
            }
        }

        resolved = typescript_extractor._resolve_path_mapping("@/components/Button")
        assert resolved == "components/Button"

    def test_extract_from_file_basic(self, typescript_extractor, temp_dir):
        """Test extraction from a single TypeScript file."""
        ts_file = temp_dir / "test.ts"
        source = """
        interface User {
            id: number;
            name: string;
        }

        function getUser(id: number): User | null {
            return null;
        }
        """

        ts_file.write_text(source)

        elements = typescript_extractor.extract_from_file(ts_file)

        assert len(elements) >= 2
        interface_element = next((e for e in elements if e.name == 'User'), None)
        func_element = next((e for e in elements if e.name == 'getUser'), None)

        assert interface_element is not None
        assert func_element is not None

    def test_extract_from_empty_file(self, typescript_extractor, temp_dir):
        """Test extraction from empty TypeScript file."""
        ts_file = temp_dir / "empty.ts"
        ts_file.write_text("")

        elements = typescript_extractor.extract_from_file(ts_file)
        assert elements == []

    def test_extract_from_nonexistent_file(self, typescript_extractor):
        """Test extraction from nonexistent file."""
        nonexistent = Path("nonexistent.ts")

        elements = typescript_extractor.extract_from_file(nonexistent)
        assert elements == []

    def test_extract_from_non_typescript_file(self, typescript_extractor, temp_dir):
        """Test extraction from non-TypeScript file."""
        js_file = temp_dir / "test.js"
        js_file.write_text("console.log('hello');")

        elements = typescript_extractor.extract_from_file(js_file)
        assert elements == []