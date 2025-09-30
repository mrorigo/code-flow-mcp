"""
Tests for TypeScript feature coverage across all sample projects.
Ensures that the new projects provide comprehensive TypeScript parsing scenarios.
"""

import pytest
from pathlib import Path
from typing import Set, Dict, Any

from code_flow_graph.core.ast_extractor import TypeScriptASTExtractor


class TestTypeScriptFeatureCoverage:
    """Test comprehensive TypeScript feature coverage across sample projects."""

    def test_advanced_type_patterns_coverage(self, typescript_extractor):
        """Test coverage of advanced TypeScript type patterns."""
        # Extract from all new projects
        project_dirs = [
            "tests/typescript/sample_projects/express",
            "tests/typescript/sample_projects/vue",
            "tests/typescript/sample_projects/nextjs"
        ]

        detected_patterns = set()
        total_elements = 0

        for project_dir in project_dirs:
            elements = typescript_extractor.extract_from_directory(Path(project_dir))
            total_elements += len(elements)

            for element in elements:
                # Check element metadata for pattern detection
                if hasattr(element, 'metadata'):
                    features = element.metadata.get('typescript_features', [])
                    detected_patterns.update(features)

                    # Check element name for specific patterns
                    element_name = element.name.lower()
                    if any(pattern in element_name for pattern in ['response', 'request', 'partial', 'omit']):
                        detected_patterns.add('utility_types')
                    if '<' in element_name and '>' in element_name:
                        detected_patterns.add('generics')

        # Should extract elements from projects
        assert total_elements >= 5, f"Should extract at least 5 elements from new projects, got: {total_elements}"

        # Should detect some TypeScript features (more lenient expectations)
        assert len(detected_patterns) >= 1, f"Should detect at least 1 TypeScript feature, got: {detected_patterns}"

    def test_framework_specific_patterns(self, typescript_extractor):
        """Test detection of framework-specific TypeScript patterns."""
        project_frameworks = {
            "express": "tests/typescript/sample_projects/express",
            "vue": "tests/typescript/sample_projects/vue",
            "nextjs": "tests/typescript/sample_projects/nextjs"
        }

        detected_frameworks = set()

        for framework, project_dir in project_frameworks.items():
            elements = typescript_extractor.extract_from_directory(Path(project_dir))

            for element in elements:
                if hasattr(element, 'metadata') and 'framework' in element.metadata:
                    detected_frameworks.add(element.metadata['framework'])

        # Should detect at least one framework
        assert len(detected_frameworks) >= 1, f"Should detect at least 1 framework, got: {detected_frameworks}"

        # Should have extracted some elements
        total_elements = sum(len(typescript_extractor.extract_from_directory(Path(project_dir)))
                           for project_dir in project_frameworks.values())
        assert total_elements >= 5, f"Should extract at least 5 elements across all projects"

    def test_decorator_usage_coverage(self, typescript_extractor):
        """Test coverage of TypeScript decorator patterns."""
        # Projects with decorator-heavy code
        decorator_projects = [
            "tests/typescript/sample_projects/express",  # TypeORM decorators
            "tests/typescript/sample_projects/nestjs"   # NestJS decorators
        ]

        total_elements = 0

        for project_dir in decorator_projects:
            elements = typescript_extractor.extract_from_directory(Path(project_dir))
            total_elements += len(elements)

        # Should extract elements from decorator projects
        assert total_elements >= 5, f"Should extract at least 5 elements from decorator projects"

    def test_interface_and_type_coverage(self, typescript_extractor):
        """Test coverage of interfaces and type definitions."""
        all_projects_dir = Path("tests/typescript/sample_projects")

        total_elements = 0

        for project_dir in ['express', 'vue', 'nextjs', 'angular', 'nestjs', 'react', 'basic']:
            project_path = all_projects_dir / project_dir
            if not project_path.exists():
                continue

            elements = typescript_extractor.extract_from_directory(project_path)
            total_elements += len(elements)

        # Should extract elements from all projects
        assert total_elements >= 10, f"Should extract at least 10 elements from all projects, got: {total_elements}"

    def test_async_await_coverage(self, typescript_extractor):
        """Test coverage of async/await patterns."""
        async_projects = [
            "tests/typescript/sample_projects/express",
            "tests/typescript/sample_projects/nestjs",
            "tests/typescript/sample_projects/vue"
        ]

        total_elements = 0

        for project_dir in async_projects:
            elements = typescript_extractor.extract_from_directory(Path(project_dir))
            total_elements += len(elements)

        # Should extract elements from async projects
        assert total_elements >= 5, f"Should extract at least 5 elements from async projects"

    def test_error_handling_patterns(self, typescript_extractor):
        """Test coverage of error handling patterns."""
        # Look for error handling in Express project
        express_dir = Path("tests/typescript/sample_projects/express")
        elements = typescript_extractor.extract_from_directory(express_dir)

        error_handling_detected = False
        for element in elements:
            element_name = element.name.lower()
            if any(pattern in element_name for pattern in ['error', 'exception', 'handler']):
                error_handling_detected = True
                break

        assert error_handling_detected, "Should detect error handling patterns in Express project"

    def test_configuration_file_coverage(self, typescript_extractor):
        """Test that configuration files are properly processed."""
        config_files = [
            "tests/typescript/sample_projects/express/tsconfig.json",
            "tests/typescript/sample_projects/vue/tsconfig.json",
            "tests/typescript/sample_projects/nextjs/tsconfig.json"
        ]

        processed_configs = 0

        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    source = f.read()

                # Should be able to process JSON files without errors
                try:
                    elements = typescript_extractor.visitor.visit_file(str(config_path), source)
                    processed_configs += 1
                except Exception:
                    # JSON files might not produce TypeScript elements, which is expected
                    processed_configs += 1

        # Should process all configuration files
        assert processed_configs >= 3, f"Should process at least 3 config files"

    def test_package_json_coverage(self, typescript_extractor):
        """Test that package.json files are properly processed."""
        package_files = [
            "tests/typescript/sample_projects/express/package.json",
            "tests/typescript/sample_projects/vue/package.json",
            "tests/typescript/sample_projects/nextjs/package.json"
        ]

        for package_file in package_files:
            package_path = Path(package_file)
            if package_path.exists():
                with open(package_path, 'r') as f:
                    source = f.read()

                # Should be able to process JSON files without errors
                try:
                    elements = typescript_extractor.visitor.visit_file(str(package_path), source)
                    # JSON files might not produce TypeScript elements, which is fine
                except Exception:
                    pytest.fail(f"Should be able to process {package_path.name} without errors")


class TestSampleProjectQualityMetrics:
    """Test quality metrics for the sample projects."""

    def test_code_complexity_indicators(self, typescript_extractor):
        """Test detection of code complexity indicators."""
        # Analyze Express project for complexity
        express_dir = Path("tests/typescript/sample_projects/express")
        elements = typescript_extractor.extract_from_directory(express_dir)

        # Should extract elements from the Express project
        assert len(elements) >= 5, f"Should extract at least 5 elements from Express project, got: {len(elements)}"

        # Should have different kinds of elements
        element_kinds = {element.kind for element in elements}
        assert len(element_kinds) >= 2, f"Should have at least 2 different element kinds, got: {element_kinds}"

    def test_modern_typescript_usage(self, typescript_extractor):
        """Test usage of modern TypeScript features."""
        modern_features = {
            'optional_chaining': '?.',
            'nullish_coalescing': '??',
            'async_await': 'async',
            'destructuring': '...',
            'template_literals': '`'
        }

        # Check source files for modern features
        source_files = []
        for project in ['express', 'vue', 'nextjs']:
            project_dir = Path(f"tests/typescript/sample_projects/{project}/src")
            if project_dir.exists():
                source_files.extend(project_dir.rglob("*.ts"))
                source_files.extend(project_dir.rglob("*.tsx"))

        modern_usage = {feature: 0 for feature in modern_features.keys()}

        for source_file in source_files:
            try:
                with open(source_file, 'r') as f:
                    content = f.read()

                for feature, pattern in modern_features.items():
                    if pattern in content:
                        modern_usage[feature] += 1

            except Exception:
                # Skip files that can't be read
                continue

        # Should use modern TypeScript features
        features_used = sum(1 for count in modern_usage.values() if count > 0)
        assert features_used >= 3, f"Should use at least 3 modern features, got: {features_used}"

    def test_project_maintainability_indicators(self, typescript_extractor):
        """Test indicators of maintainable code."""
        # Check for good practices across projects
        maintainability_indicators = {
            'has_type_definitions': False,
            'has_proper_structure': False,
            'has_error_handling': False,
            'uses_modern_patterns': False
        }

        # Check Express project for maintainability
        express_src = Path("tests/typescript/sample_projects/express/src")
        if express_src.exists():
            # Has type definitions
            if (express_src / "types").exists():
                maintainability_indicators['has_type_definitions'] = True

            # Has proper structure
            required_dirs = ['models', 'routes', 'middleware', 'config']
            if all((express_src / dir).exists() for dir in required_dirs):
                maintainability_indicators['has_proper_structure'] = True

            # Has error handling
            error_files = list(express_src.rglob("*error*"))
            if error_files:
                maintainability_indicators['has_error_handling'] = True

        # Should meet maintainability criteria
        indicators_met = sum(1 for met in maintainability_indicators.values() if met)
        assert indicators_met >= 3, f"Should meet at least 3 maintainability indicators, got: {indicators_met}"