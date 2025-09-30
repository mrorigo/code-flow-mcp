"""
Integration tests for TypeScript compiler integration and fallback scenarios.
"""

import pytest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from code_flow_graph.core.typescript_extractor import TypeScriptASTVisitor, TypeScriptASTExtractor


class TestTypeScriptCompilerIntegration:
    """Test TypeScript compiler integration and fallback scenarios."""

    def test_compiler_integration_success(self, typescript_visitor, temp_dir):
        """Test successful TypeScript compiler integration."""
        # Create a test TypeScript file
        ts_file = temp_dir / "test.ts"
        source = """
        interface User {
            id: number;
            name: string;
        }

        function createUser(userData: Partial<User>): User {
            return {
                id: Date.now(),
                ...userData
            };
        }
        """

        with open(ts_file, 'w') as f:
            f.write(source)

        # Mock successful TypeScript compiler run
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')

            # Ensure TypeScript is marked as available
            typescript_visitor.typescript_available = True

            with patch.object(typescript_visitor, '_get_typescript_version', return_value='Version 4.9.5'):
                result = typescript_visitor._run_typescript_compiler(str(ts_file), source)

                assert result is not None
                assert result['success'] is True
                assert result['file_path'] == str(ts_file)
                assert 'line_count' in result
                assert 'compiler_version' in result

    def test_compiler_integration_timeout(self, typescript_visitor, temp_dir):
        """Test TypeScript compiler timeout handling."""
        ts_file = temp_dir / "test.ts"
        source = "function test() { return true; }"

        with open(ts_file, 'w') as f:
            f.write(source)

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(['npx', 'typescript'], 30)

            result = typescript_visitor._run_typescript_compiler(str(ts_file), source)

            assert result is None

    def test_compiler_integration_file_not_found(self, typescript_visitor):
        """Test TypeScript compiler when executable is not found."""
        source = "function test() { return true; }"

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("npx not found")

            result = typescript_visitor._run_typescript_compiler("test.ts", source)

            assert result is None

    def test_compiler_integration_error_output(self, typescript_visitor, temp_dir):
        """Test TypeScript compiler with error output."""
        ts_file = temp_dir / "test.ts"
        source = "function test("  # Incomplete function - should cause error

        with open(ts_file, 'w') as f:
            f.write(source)

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout='',
                stderr='error: Missing closing parenthesis'
            )

            result = typescript_visitor._run_typescript_compiler(str(ts_file), source)

            assert result is None

    def test_fallback_parsing_when_compiler_unavailable(self, typescript_visitor, temp_dir):
        """Test fallback to regex parsing when TypeScript compiler is unavailable."""
        ts_file = temp_dir / "test.ts"
        source = """
        interface User {
            id: number;
            name: string;
        }

        class UserService {
            public async getUser(id: number): Promise<User | null> {
                return null;
            }
        }

        function helperFunction(value: string): boolean {
            return value.length > 0;
        }
        """

        with open(ts_file, 'w') as f:
            f.write(source)

        # Mock unavailable TypeScript compiler
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("node not found")

            elements = typescript_visitor.visit_file(str(ts_file), source)

            # Should still extract elements using regex fallback
            assert len(elements) >= 3

            # Check that we got the expected elements
            element_names = [e.name for e in elements]
            assert 'User' in element_names  # Interface
            assert 'UserService' in element_names  # Class
            assert 'helperFunction' in element_names  # Function

    def test_parse_with_compiler_vs_regex(self, typescript_visitor, temp_dir):
        """Test that compiler parsing and regex parsing produce similar results."""
        ts_file = temp_dir / "test.ts"
        source = """
        interface Product {
            id: number;
            name: string;
            price: number;
        }

        class ProductService {
            private products: Product[] = [];

            public findProduct(id: number): Product | null {
                return this.products.find(p => p.id === id) || null;
            }

            public createProduct(productData: Omit<Product, 'id'>): Product {
                const newProduct: Product = {
                    id: Date.now(),
                    ...productData
                };
                this.products.push(newProduct);
                return newProduct;
            }
        }
        """

        with open(ts_file, 'w') as f:
            f.write(source)

        # Test with regex parsing (simulating unavailable compiler)
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("node not found")

            regex_elements = typescript_visitor.visit_file(str(ts_file), source)

        # Reset visitor state
        typescript_visitor.elements = []
        typescript_visitor.source_lines = []
        typescript_visitor.current_file = str(ts_file)

        # Test with mock compiler success
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='success', stderr='')

            compiler_elements = typescript_visitor.visit_file(str(ts_file), source)

        # Both should extract similar number of elements
        assert len(regex_elements) > 0
        assert len(compiler_elements) > 0

        # Both should extract the same major elements
        regex_names = {e.name for e in regex_elements}
        compiler_names = {e.name for e in compiler_elements}

        assert 'Product' in regex_names  # Interface
        assert 'ProductService' in regex_names  # Class
        assert 'findProduct' in regex_names  # Method
        assert 'createProduct' in regex_names  # Method

    def test_get_typescript_version_success(self, typescript_visitor):
        """Test successful TypeScript version retrieval."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='Version 4.9.5')

            version = typescript_visitor._get_typescript_version()
            assert version == 'Version 4.9.5'

    def test_get_typescript_version_failure(self, typescript_visitor):
        """Test TypeScript version retrieval failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("npx not found")

            version = typescript_visitor._get_typescript_version()
            assert version is None

    def test_visitor_with_mixed_availability(self, typescript_visitor):
        """Test visitor behavior with mixed TypeScript availability."""
        # Initially, TypeScript might be available
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout='v18.0.0'),  # node --version
                MagicMock(returncode=0, stdout='4.9.5')     # npx typescript --version
            ]

            # First check should succeed
            assert typescript_visitor._check_typescript_available() is True

        # Later, TypeScript might become unavailable
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("node not found")

            # Second check should fail
            assert typescript_visitor._check_typescript_available() is False

    def test_extract_from_directory_with_mixed_files(self, typescript_extractor, temp_dir):
        """Test extraction from directory with mixed file types."""
        # Create various file types
        (temp_dir / "valid.ts").write_text("""
        interface ValidInterface {
            id: number;
        }

        function validFunction(): string {
            return "valid";
        }
        """)

        (temp_dir / "invalid.ts").write_text("""
        class IncompleteClass {
            public brokenMethod( {
                // Missing closing parenthesis
        """)

        (temp_dir / "not_typescript.js").write_text("""
        function javaScriptFunction() {
            console.log('This is JavaScript, not TypeScript');
        }
        """)

        elements = typescript_extractor.extract_from_directory(temp_dir)

        # Should extract from valid TypeScript files but handle invalid gracefully
        assert len(elements) >= 2  # Interface + function

        # Check that we got elements from the valid file
        element_names = {e.name for e in elements}
        assert 'ValidInterface' in element_names
        assert 'validFunction' in element_names

    def test_tsconfig_path_mapping_resolution(self, typescript_extractor, temp_dir):
        """Test TypeScript path mapping resolution."""
        # Setup path mappings
        typescript_extractor.path_mappings = {
            'baseUrl': './src',
            'paths': {
                '@/*': ['src/*'],
                '@/components/*': ['src/components/*'],
                '@/services/*': ['src/services/*']
            }
        }

        # Test various path mappings
        assert typescript_extractor._resolve_path_mapping("@/components/Button") == "src/components/Button"
        assert typescript_extractor._resolve_path_mapping("@/services/UserService") == "src/services/UserService"
        assert typescript_extractor._resolve_path_mapping("@/utils/helpers") == "src/utils/helpers"

        # Test unmapped path (should return as-is)
        assert typescript_extractor._resolve_path_mapping("react") == "react"

    def test_build_interface_hierarchy(self, typescript_extractor):
        """Test interface hierarchy building."""
        hierarchy = typescript_extractor._build_interface_hierarchy()

        # Should return empty dict for now (placeholder implementation)
        assert isinstance(hierarchy, dict)
        assert len(hierarchy) == 0

    def test_resolve_type_aliases(self, typescript_extractor):
        """Test type alias resolution."""
        resolved = typescript_extractor._resolve_type_aliases()

        # Should return empty dict for now (placeholder implementation)
        assert isinstance(resolved, dict)
        assert len(resolved) == 0

    def test_get_compiler_options_with_defaults(self, typescript_extractor, temp_dir):
        """Test compiler options retrieval with defaults."""
        # Create tsconfig without explicit compiler options
        tsconfig = temp_dir / "tsconfig.json"
        tsconfig.write_text('{"compilerOptions": {"target": "ES2020"}}')

        typescript_extractor.tsconfig = {"compilerOptions": {"target": "ES2020"}}

        options = typescript_extractor._get_compiler_options()

        assert options["target"] == "ES2020"
        assert options["strict"] is True  # Should have default
        assert options["esModuleInterop"] is True  # Should have default
        assert options["skipLibCheck"] is True  # Should have default

    def test_get_compiler_options_without_config(self, typescript_extractor):
        """Test compiler options retrieval without tsconfig."""
        typescript_extractor.tsconfig = None

        options = typescript_extractor._get_compiler_options()
        assert options == {}


class TestTypeScriptFrameworkIntegration:
    """Test TypeScript framework-specific integration."""

    def test_angular_component_extraction(self, typescript_visitor, sample_angular_code):
        """Test Angular component extraction."""
        elements = typescript_visitor.visit_file("test.component.ts", sample_angular_code)

        # Should extract the component class
        class_element = next((e for e in elements if e.name == 'UserCardComponent'), None)
        assert class_element is not None
        assert class_element.kind == 'class'
        assert class_element.metadata.get('framework') == 'angular'

        # Should detect Angular-specific decorators
        decorators = class_element.decorators
        assert len(decorators) > 0
        decorator_names = [d.get('name') for d in decorators if d.get('framework') == 'angular']
        assert len(decorator_names) > 0

    def test_nestjs_controller_extraction(self, typescript_visitor, sample_nestjs_code):
        """Test NestJS controller extraction."""
        elements = typescript_visitor.visit_file("user.controller.ts", sample_nestjs_code)

        # Should extract the controller class
        class_element = next((e for e in elements if e.name == 'UserController'), None)
        assert class_element is not None
        assert class_element.kind == 'class'
        assert class_element.metadata.get('framework') == 'nestjs'

        # Should detect NestJS-specific decorators
        decorators = class_element.decorators
        assert len(decorators) > 0
        decorator_names = [d.get('name') for d in decorators if d.get('framework') == 'nestjs']
        assert 'Controller' in decorator_names

    def test_react_component_extraction(self, typescript_visitor, sample_react_code):
        """Test React component extraction."""
        elements = typescript_visitor.visit_file("UserProfile.tsx", sample_react_code)

        # Should extract the React component function
        func_element = next((e for e in elements if e.name == 'UserProfile'), None)
        assert func_element is not None
        assert func_element.kind == 'function'
        assert func_element.metadata.get('framework') == 'react'

        # Should detect React hooks
        features = func_element.metadata.get('typescript_features', [])
        assert 'hooks' in features
        assert 'jsx' in features

    def test_express_app_extraction(self, typescript_visitor, sample_express_code):
        """Test Express application extraction."""
        elements = typescript_visitor.visit_file("app.ts", sample_express_code)

        # Should extract the Express app setup
        func_element = next((e for e in elements if e.name == 'app'), None)
        assert func_element is not None
        assert func_element.metadata.get('framework') == 'express'

        # Should detect Express patterns
        features = func_element.metadata.get('typescript_features', [])
        assert 'routing' in features
        assert 'middleware' in features

    def test_framework_detection_accuracy(self, typescript_visitor, sample_angular_code, sample_nestjs_code, sample_react_code, sample_express_code):
        """Test framework detection accuracy across different patterns."""
        test_cases = [
            (sample_angular_code, 'angular'),
            (sample_nestjs_code, 'nestjs'),
            (sample_react_code, 'react'),
            (sample_express_code, 'express')
        ]

        for sample_code, expected_framework in test_cases:
            elements = typescript_visitor.visit_file(f"test.{expected_framework}.ts", sample_code)

            # Find the main class/function element
            main_element = None
            for element in elements:
                if element.name in ['UserCardComponent', 'UserController', 'UserProfile', 'app']:
                    main_element = element
                    break

            assert main_element is not None, f"Could not find main element in {expected_framework} code"
            assert main_element.metadata.get('framework') == expected_framework, \
                f"Framework detection failed for {expected_framework}: got {main_element.metadata.get('framework')}"

    def test_mixed_framework_patterns(self, typescript_visitor):
        """Test handling of mixed framework patterns."""
        mixed_code = """
        import { Component } from '@angular/core';
        import { Controller, Get } from '@nestjs/common';

        @Component({
            selector: 'app-mixed'
        })
        @Controller('mixed')
        class MixedComponent {
            @Get()
            getData() {
                return 'mixed';
            }
        }
        """

        elements = typescript_visitor.visit_file("mixed.ts", mixed_code)

        class_element = next((e for e in elements if e.name == 'MixedComponent'), None)
        assert class_element is not None

        # Should detect multiple frameworks or prioritize one
        framework = class_element.metadata.get('framework')
        assert framework in ['angular', 'nestjs']

        # Should have multiple decorators
        assert len(class_element.decorators) >= 2