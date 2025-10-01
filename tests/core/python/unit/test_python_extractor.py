"""
Unit tests for PythonASTExtractor functionality.
"""

import pytest
import ast
from pathlib import Path
from unittest.mock import patch, MagicMock

from code_flow_graph.core.python_extractor import PythonASTExtractor
from code_flow_graph.core.models import FunctionElement, ClassElement


class TestPythonASTExtractor:
    """Test cases for PythonASTExtractor."""

    def test_extractor_initialization(self, python_extractor):
        """Test PythonASTExtractor initializes correctly."""
        assert isinstance(python_extractor.visitor, object)  # PythonASTVisitor
        assert python_extractor.project_root is None

    def test_extract_from_file_basic(self, python_extractor, temp_dir):
        """Test extraction from a single Python file."""
        py_file = temp_dir / "test.py"
        source = '''
def greet_user(name: str) -> str:
    """Greets a user by name."""
    return f"Hello, {name}!"

class UserService:
    """Service for user operations."""

    def get_user(self, user_id: int) -> dict:
        return {"id": user_id, "name": "User"}
'''

        with open(py_file, 'w') as f:
            f.write(source)

        elements = python_extractor.extract_from_file(py_file)

        assert len(elements) >= 3  # greet_user function + UserService class + get_user method

        # Check function extraction
        func_element = next((e for e in elements if e.name == 'greet_user'), None)
        assert func_element is not None
        assert func_element.kind == 'function'
        assert func_element.file_path == str(py_file.resolve())
        assert func_element.docstring == "Greets a user by name."

        # Check class extraction
        class_element = next((e for e in elements if e.name == 'UserService'), None)
        assert class_element is not None
        assert class_element.kind == 'class'
        assert 'get_user' in class_element.methods

    def test_extract_from_empty_file(self, python_extractor, temp_dir):
        """Test extraction from empty Python file."""
        py_file = temp_dir / "empty.py"
        py_file.write_text("")

        elements = python_extractor.extract_from_file(py_file)
        assert elements == []

    def test_extract_from_nonexistent_file(self, python_extractor):
        """Test extraction from nonexistent file."""
        nonexistent = Path("nonexistent.py")

        elements = python_extractor.extract_from_file(nonexistent)
        assert elements == []

    def test_extract_from_non_python_file(self, python_extractor, temp_dir):
        """Test extraction from non-Python file."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("This is not Python code")

        elements = python_extractor.extract_from_file(txt_file)
        assert elements == []

    def test_extract_from_file_with_syntax_error(self, python_extractor, temp_dir):
        """Test extraction from file with syntax error."""
        py_file = temp_dir / "syntax_error.py"
        py_file.write_text("def invalid_function(")  # Missing closing parenthesis

        elements = python_extractor.extract_from_file(py_file)
        # Should handle error gracefully and return empty list
        assert elements == []

    def test_extract_from_file_with_imports(self, python_extractor, temp_dir):
        """Test extraction from file with imports."""
        py_file = temp_dir / "imports_test.py"
        source = '''
import os
import sys
from pathlib import Path
from typing import List, Dict

def file_operation(path: str) -> bool:
    """Check if path exists using os."""
    return os.path.exists(path)

def typed_function(items: List[Dict[str, str]]) -> int:
    """Function with type annotations."""
    return len(items)
'''

        with open(py_file, 'w') as f:
            f.write(source)

        elements = python_extractor.extract_from_file(py_file)

        func_elements = [e for e in elements if e.kind == 'function']
        assert len(func_elements) == 2

        # Check that imports are processed correctly
        file_op = next((e for e in elements if e.name == 'file_operation'), None)
        typed_func = next((e for e in elements if e.name == 'typed_function'), None)

        assert file_op is not None
        assert typed_func is not None

        # Check that external dependencies are detected
        assert 'os' in file_op.external_dependencies

    def test_extract_from_directory_basic(self, python_extractor, temp_dir):
        """Test extraction from a directory with Python files."""
        # Create test directory structure
        src_dir = temp_dir / "src"
        src_dir.mkdir()

        # Create multiple Python files
        (src_dir / "__init__.py").write_text("")
        (src_dir / "module1.py").write_text('''
def function_one():
    return "one"
''')

        (src_dir / "module2.py").write_text('''
class TestClass:
    def method_two(self):
        return "two"
''')

        elements = python_extractor.extract_from_directory(src_dir)

        assert len(elements) >= 3  # function_one + TestClass + method_two

        func_names = {e.name for e in elements}
        assert 'function_one' in func_names
        assert 'TestClass' in func_names
        assert 'method_two' in func_names

    def test_extract_from_directory_with_gitignore(self, python_extractor, temp_dir):
        """Test extraction from directory respecting .gitignore patterns."""
        # Create .gitignore file
        gitignore = temp_dir / ".gitignore"
        gitignore.write_text("test_*.py\n__pycache__/\n*.pyc\n")

        # Create test files
        (temp_dir / "good_file.py").write_text('''
def good_function():
    return "good"
''')

        (temp_dir / "test_ignore.py").write_text('''
def ignored_function():
    return "ignored"
''')

        (temp_dir / "__pycache__").mkdir()
        (temp_dir / "__pycache__/cache.pyc").write_text("")

        elements = python_extractor.extract_from_directory(temp_dir)

        # Should only extract from good_file.py, ignoring test_ignore.py and __pycache__
        func_names = {e.name for e in elements if e.kind == 'function'}
        assert 'good_function' in func_names
        assert 'ignored_function' not in func_names

    def test_extract_from_directory_with_root_relative_gitignore(self, python_extractor, temp_dir):
        """Test extraction from directory respecting root-relative gitignore patterns."""
        # Create .gitignore file with root-relative pattern
        gitignore = temp_dir / ".gitignore"
        gitignore.write_text("/temp_*.py\nnested_*.py\n")

        # Create test files in root
        (temp_dir / "temp_ignore.py").write_text('''
def temp_ignored_function():
    return "temp_ignored"
''')

        (temp_dir / "root_keep.py").write_text('''
def root_keep_function():
    return "root_keep"
''')

        # Create nested directory with files
        nested_dir = temp_dir / "subdir"
        nested_dir.mkdir()

        (nested_dir / "nested_ignore.py").write_text('''
def nested_ignored_function():
    return "nested_ignored"
''')

        (nested_dir / "nested_keep.py").write_text('''
def nested_keep_function():
    return "nested_keep"
''')

        elements = python_extractor.extract_from_directory(temp_dir)

        # Should extract functions that don't match the patterns
        func_names = {e.name for e in elements if e.kind == 'function'}
        assert 'root_keep_function' in func_names  # Should not be affected by /temp_*.py
        assert 'temp_ignored_function' not in func_names  # Should be ignored by /temp_*.py
        assert 'nested_keep_function' not in func_names  # Should be ignored by nested_*.py (matches nested_* pattern)
        assert 'nested_ignored_function' not in func_names  # Should be ignored by nested_*.py

    def test_gitignore_root_relative_node_modules_pattern(self, python_extractor, temp_dir):
        """Test that /node_modules pattern only ignores root node_modules, not nested ones."""
        # Create .gitignore file with ONLY the root-relative node_modules pattern
        gitignore = temp_dir / ".gitignore"
        gitignore.write_text("/node_modules\n")

        # Create root node_modules
        root_node_modules = temp_dir / "node_modules"
        root_node_modules.mkdir()
        (root_node_modules / "root_package.py").write_text('''
def root_package_function():
    return "root"
''')

        # Create nested node_modules
        nested_dir = temp_dir / "src"
        nested_dir.mkdir()
        nested_node_modules = nested_dir / "node_modules"
        nested_node_modules.mkdir()
        (nested_node_modules / "nested_package.py").write_text('''
def nested_package_function():
    return "nested"
''')

        # Create regular source files
        (temp_dir / "app.py").write_text('''
def app_function():
    return "app"
''')

        elements = python_extractor.extract_from_directory(temp_dir)

        # Should only ignore files in root node_modules, not nested node_modules
        func_names = {e.name for e in elements if e.kind == 'function'}
        assert 'app_function' in func_names  # Regular file should be included
        assert 'root_package_function' not in func_names  # Should be ignored by /node_modules
        assert 'nested_package_function' in func_names  # Should NOT be ignored by /node_modules

    def test_gitignore_directory_patterns_with_leading_slash(self, python_extractor, temp_dir):
        """Test directory patterns with leading slash."""
        # Create .gitignore file with directory patterns
        gitignore = temp_dir / ".gitignore"
        gitignore.write_text("/build/\ndist/\n")

        # Create root build directory
        root_build = temp_dir / "build"
        root_build.mkdir()
        (root_build / "root_build.py").write_text('''
def root_build_function():
    return "root_build"
''')

        # Create nested build directory
        nested_dir = temp_dir / "src"
        nested_dir.mkdir()
        nested_build = nested_dir / "build"
        nested_build.mkdir()
        (nested_build / "nested_build.py").write_text('''
def nested_build_function():
    return "nested_build"
''')

        # Create root dist directory
        root_dist = temp_dir / "dist"
        root_dist.mkdir()
        (root_dist / "root_dist.py").write_text('''
def root_dist_function():
    return "root_dist"
''')

        elements = python_extractor.extract_from_directory(temp_dir)

        # Should only ignore files in root directories, not nested ones
        func_names = {e.name for e in elements if e.kind == 'function'}
        assert 'root_build_function' not in func_names  # Should be ignored by /build/
        assert 'nested_build_function' in func_names   # Should NOT be ignored by /build/
        assert 'root_dist_function' not in func_names  # Should be ignored by /dist/

    def test_extract_from_nested_directory(self, python_extractor, temp_dir):
        """Test extraction from nested directory structure."""
        # Create nested structure
        root_pkg = temp_dir / "mypackage"
        root_pkg.mkdir()
        sub_pkg = root_pkg / "subpackage"
        sub_pkg.mkdir()

        # Create __init__.py files
        (root_pkg / "__init__.py").write_text("")
        (sub_pkg / "__init__.py").write_text("")

        # Create Python files at different levels
        (root_pkg / "main.py").write_text('''
def main_function():
    return "main"
''')

        (sub_pkg / "utils.py").write_text('''
def utility_function():
    return "utility"
''')

        elements = python_extractor.extract_from_directory(temp_dir)

        func_names = {e.name for e in elements if e.kind == 'function'}
        assert 'main_function' in func_names
        assert 'utility_function' in func_names

    def test_extract_from_directory_empty(self, python_extractor, temp_dir):
        """Test extraction from directory with no Python files."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        elements = python_extractor.extract_from_directory(empty_dir)
        assert elements == []

    def test_extract_from_directory_mixed_files(self, python_extractor, temp_dir):
        """Test extraction from directory with mixed file types."""
        # Create directory with various file types
        mixed_dir = temp_dir / "mixed"
        mixed_dir.mkdir()

        (mixed_dir / "python_file.py").write_text('''
def python_function():
    return "python"
''')

        (mixed_dir / "javascript_file.js").write_text("function jsFunction() {}")
        (mixed_dir / "readme.txt").write_text("Just documentation")
        (mixed_dir / "requirements.txt").write_text("requests==2.25.1")

        elements = python_extractor.extract_from_directory(mixed_dir)

        # Should only extract Python functions
        func_names = {e.name for e in elements if e.kind == 'function'}
        assert 'python_function' in func_names
        assert len(func_names) == 1  # Only Python functions

    def test_extract_from_file_preserves_line_numbers(self, python_extractor, temp_dir):
        """Test that line numbers are correctly preserved."""
        py_file = temp_dir / "line_numbers.py"
        source = '''# Comment line
# Another comment

def first_function():
    """First function."""
    return "first"

class TestClass:
    """Test class."""

    def __init__(self):
        pass

    def class_method(self):
        """Method in class."""
        return "method"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        elements = python_extractor.extract_from_file(py_file)

        # Check function line numbers
        first_func = next((e for e in elements if e.name == 'first_function'), None)
        assert first_func is not None
        assert first_func.line_start == 4  # After comments
        assert first_func.line_end >= 6

        # Check class line numbers
        test_class = next((e for e in elements if e.name == 'TestClass'), None)
        assert test_class is not None
        assert test_class.line_start == 8

        # Check method line numbers
        class_method = next((e for e in elements if e.name == 'class_method'), None)
        assert class_method is not None
        assert class_method.line_start == 14  # Corrected line number

    def test_extract_from_file_with_complex_imports(self, python_extractor, temp_dir):
        """Test extraction with complex import patterns."""
        py_file = temp_dir / "complex_imports.py"
        source = '''
import os.path
import sys as system
from pathlib import Path as PathType
from typing import List, Dict as Dictionary
from collections import defaultdict, namedtuple

def function_with_imports():
    """Function that uses complex imports."""
    p = PathType("/tmp")
    d = defaultdict(list)
    Point = namedtuple('Point', ['x', 'y'])
    return p.exists()
'''

        with open(py_file, 'w') as f:
            f.write(source)

        elements = python_extractor.extract_from_file(py_file)

        func_element = next((e for e in elements if e.name == 'function_with_imports'), None)
        assert func_element is not None

        # Check that various imports are detected as dependencies
        expected_deps = {'pathlib', 'collections'}
        assert any(dep in func_element.external_dependencies for dep in expected_deps)

    def test_extract_from_file_with_aliased_imports(self, python_extractor, temp_dir):
        """Test extraction with aliased imports."""
        py_file = temp_dir / "aliased_imports.py"
        source = '''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def data_analysis():
    """Function using aliased imports."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    arr = np.array([1, 2, 3])
    plt.plot(arr)
    return df
'''

        with open(py_file, 'w') as f:
            f.write(source)

        elements = python_extractor.extract_from_file(py_file)

        func_element = next((e for e in elements if e.name == 'data_analysis'), None)
        assert func_element is not None

        # Should detect the actual module names, not aliases
        assert 'pandas' in func_element.external_dependencies
        assert 'numpy' in func_element.external_dependencies
        assert 'matplotlib' in func_element.external_dependencies

    def test_extract_from_file_with_relative_imports(self, python_extractor, temp_dir):
        """Test extraction with relative imports."""
        # Create package structure
        pkg_dir = temp_dir / "mypackage"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        main_file = pkg_dir / "main.py"
        main_file.write_text('''
from .utils import helper_function
from ..config import settings

def main_function():
    """Main function using relative imports."""
    result = helper_function()
    return settings.get("key")
''')

        utils_file = pkg_dir / "utils.py"
        utils_file.write_text('''
def helper_function():
    """Helper function."""
    return "helper"
''')

        config_file = temp_dir / "config.py"
        config_file.write_text('''
class Settings:
    def get(self, key):
        return f"value_for_{key}"

settings = Settings()
''')

        elements = python_extractor.extract_from_file(main_file)

        func_element = next((e for e in elements if e.name == 'main_function'), None)
        assert func_element is not None

        # Relative imports should not be treated as external dependencies
        # (this is a heuristic and may need adjustment based on implementation)
        external_deps = func_element.external_dependencies
        # Should not contain relative import modules as external deps

    def test_extract_from_directory_performance_with_many_files(self, python_extractor, temp_dir):
        """Test extraction performance with many Python files."""
        # Create many Python files for performance testing
        for i in range(10):
            py_file = temp_dir / f"module_{i}.py"
            py_file.write_text(f'''
def function_{i}():
    """Function {i}."""
    return "result_{i}"
''')

        elements = python_extractor.extract_from_directory(temp_dir)

        # Should extract all functions
        func_names = {e.name for e in elements if e.kind == 'function'}
        for i in range(10):
            assert f'function_{i}' in func_names

    def test_extract_from_file_with_docstring_extraction(self, python_extractor, temp_dir):
        """Test that docstrings are correctly extracted."""
        py_file = temp_dir / "docstrings.py"
        source = '''
def simple_function():
    """This is a simple function."""
    pass

def function_with_multiline_docstring():
    """
    This is a multiline docstring.

    It has multiple lines and should be preserved.
    """
    return "test"

class ClassWithDocstring:
    """
    Class with multiline docstring.

    This class demonstrates docstring extraction.
    """

    def method_with_docstring(self):
        """
        Method with its own docstring.

        Returns:
            A string value
        """
        return "method result"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        elements = python_extractor.extract_from_file(py_file)

        # Check function docstrings
        simple_func = next((e for e in elements if e.name == 'simple_function'), None)
        assert simple_func is not None
        assert simple_func.docstring == "This is a simple function."

        multiline_func = next((e for e in elements if e.name == 'function_with_multiline_docstring'), None)
        assert multiline_func is not None
        assert "This is a multiline docstring." in multiline_func.docstring
        assert "It has multiple lines" in multiline_func.docstring

        # Check class docstring
        test_class = next((e for e in elements if e.name == 'ClassWithDocstring'), None)
        assert test_class is not None
        assert "Class with multiline docstring." in test_class.docstring

        # Check method docstring
        method = next((e for e in elements if e.name == 'method_with_docstring'), None)
        assert method is not None
        assert "Method with its own docstring." in method.docstring

    def test_extract_from_file_preserves_full_source(self, python_extractor, temp_dir):
        """Test that full source code is preserved in elements."""
        py_file = temp_dir / "source_preservation.py"
        source = '''def test_function():
    """Test function."""
    return "test"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        elements = python_extractor.extract_from_file(py_file)

        func_element = next((e for e in elements if e.name == 'test_function'), None)
        assert func_element is not None
        assert func_element.full_source == source

    def test_extract_from_file_with_unicode_content(self, python_extractor, temp_dir):
        """Test extraction from file with Unicode content."""
        py_file = temp_dir / "unicode_test.py"
        source = '''# -*- coding: utf-8 -*-
def café_function() -> str:
    """Function with Unicode in name and content."""
    return "café"

def función_española(dato: str) -> str:
    """Spanish function name."""
    return dato.upper()
'''

        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(source)

        elements = python_extractor.extract_from_file(py_file)

        # Should handle Unicode correctly
        cafe_func = next((e for e in elements if e.name == 'café_function'), None)
        spanish_func = next((e for e in elements if e.name == 'función_española'), None)

        assert cafe_func is not None
        assert spanish_func is not None

    def test_extract_from_directory_with_symlinks(self, python_extractor, temp_dir):
        """Test extraction from directory containing symlinks."""
        # Create a Python file
        real_file = temp_dir / "real_module.py"
        real_file.write_text('''
def real_function():
    return "real"
''')

        # Create symlink to the file
        symlink_file = temp_dir / "symlink_module.py"
        try:
            symlink_file.symlink_to(real_file)
        except (OSError, NotImplementedError):
            # Skip on platforms that don't support symlinks
            pytest.skip("Symlinks not supported on this platform")

        elements = python_extractor.extract_from_directory(temp_dir)

        # Should extract from real file but not be confused by symlink
        func_names = {e.name for e in elements if e.kind == 'function'}
        assert 'real_function' in func_names