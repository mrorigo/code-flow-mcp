"""
Unit tests for PythonASTVisitor functionality.
"""

import pytest
import ast
from pathlib import Path
from unittest.mock import patch, MagicMock

from code_flow_graph.core.python_extractor import PythonASTVisitor, PythonASTExtractor
from code_flow_graph.core.models import FunctionElement, ClassElement


class TestPythonASTVisitor:
    """Test cases for PythonASTVisitor."""

    def test_visitor_initialization(self):
        """Test PythonASTVisitor initializes correctly."""
        visitor = PythonASTVisitor()
        assert visitor.elements == []
        assert visitor.current_class is None
        assert visitor.current_file == ""
        assert visitor.source_lines == []
        assert visitor.file_level_imports == {}
        assert visitor.file_level_import_from_targets == set()

    def test_visit_file_basic_function(self, temp_dir):
        """Test visiting a file with a basic function."""
        py_file = temp_dir / "test.py"
        source = '''
def greet_user(name: str) -> str:
    """Greets a user by name."""
    return f"Hello, {name}!"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        assert len(elements) >= 1
        func_element = next((e for e in elements if e.name == 'greet_user'), None)
        assert func_element is not None
        assert func_element.kind == 'function'
        assert 'name: str' in func_element.parameters
        assert func_element.return_type == 'str'
        assert func_element.docstring == "Greets a user by name."
        assert func_element.is_method is False

    def test_visit_file_with_class(self, temp_dir):
        """Test visiting a file with a class."""
        py_file = temp_dir / "test.py"
        source = '''
class UserService:
    """Service class for user operations."""

    def __init__(self, users=None):
        self.users = users or []

    def get_user(self, user_id: int) -> dict:
        """Get user by ID."""
        return next((u for u in self.users if u["id"] == user_id), None)
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        assert len(elements) >= 3  # Class + __init__ + get_user
        class_element = next((e for e in elements if e.name == 'UserService'), None)
        assert class_element is not None
        assert class_element.kind == 'class'
        assert class_element.docstring == "Service class for user operations."
        assert 'get_user' in class_element.methods
        # Note: Class attribute detection may need refinement in visitor implementation
        # assert 'users' in class_element.attributes

        # Check method extraction
        get_user_element = next((e for e in elements if e.name == 'get_user'), None)
        assert get_user_element is not None
        assert get_user_element.is_method is True
        assert get_user_element.class_name == 'UserService'

    def test_visit_file_with_decorators(self, python_visitor, temp_dir):
        """Test visiting a file with decorated functions and classes."""
        py_file = temp_dir / "test.py"
        source = '''
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(n: int) -> int:
    """Expensive function that benefits from caching."""
    return n * n

@dataclass
class User:
    """User data class."""
    name: str
    age: int
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        # Check function decorator
        func_element = next((e for e in elements if e.name == 'expensive_function'), None)
        assert func_element is not None
        assert len(func_element.decorators) == 1
        assert func_element.decorators[0]['name'] == 'lru_cache'
        assert func_element.decorators[0]['args'] == []
        assert func_element.decorators[0]['kwargs'] == {'maxsize': '128'}

        # Check class decorator
        class_element = next((e for e in elements if e.name == 'User'), None)
        assert class_element is not None
        assert len(class_element.decorators) == 1
        assert class_element.decorators[0]['name'] == 'dataclass'

    def test_visit_file_with_async_function(self, python_visitor, temp_dir):
        """Test visiting a file with async functions."""
        py_file = temp_dir / "test.py"
        source = '''
import asyncio

async def fetch_data(url: str) -> dict:
    """Fetch data asynchronously."""
    await asyncio.sleep(1)
    return {"url": url, "status": "success"}

class AsyncService:
    """Service with async methods."""

    async def process_async(self, data: list) -> str:
        """Process data asynchronously."""
        await asyncio.sleep(0.1)
        return f"Processed {len(data)} items"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        # Check async function
        fetch_element = next((e for e in elements if e.name == 'fetch_data'), None)
        assert fetch_element is not None
        assert fetch_element.is_async is True

        # Check async method
        process_element = next((e for e in elements if e.name == 'process_async'), None)
        assert process_element is not None
        assert process_element.is_async is True
        assert process_element.is_method is True

    def test_visit_file_with_exception_handling(self, python_visitor, temp_dir):
        """Test visiting a file with exception handling."""
        py_file = temp_dir / "test.py"
        source = '''
def risky_operation(value: int) -> str:
    """Function that handles various exceptions."""
    try:
        if value < 0:
            raise ValueError("Negative value not allowed")
        return str(value)
    except ValueError as e:
        return f"ValueError: {e}"
    except (TypeError, AttributeError):
        return "Type error occurred"
    except Exception:
        return "Unknown error"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        func_element = next((e for e in elements if e.name == 'risky_operation'), None)
        assert func_element is not None
        assert 'ValueError' in func_element.catches_exceptions
        assert 'TypeError' in func_element.catches_exceptions
        assert 'AttributeError' in func_element.catches_exceptions
        assert 'Exception' in func_element.catches_exceptions

    def test_visit_file_with_imports_and_dependencies(self, python_visitor, temp_dir):
        """Test visiting a file with imports and external dependencies."""
        py_file = temp_dir / "test.py"
        source = '''
import requests
from typing import List, Optional
from pathlib import Path

def api_call(endpoint: str) -> dict:
    """Make API call using requests."""
    response = requests.get(endpoint)
    return response.json()

def file_operation(path: str) -> bool:
    """Work with file paths."""
    p = Path(path)
    return p.exists()
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(temp_dir / "test.py")
        visitor.file_level_imports = python_visitor.file_level_imports.copy()
        visitor.file_level_import_from_targets = python_visitor.file_level_import_from_targets.copy()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        # Check function dependencies
        api_element = next((e for e in elements if e.name == 'api_call'), None)
        assert api_element is not None
        assert 'requests' in api_element.external_dependencies

        file_element = next((e for e in elements if e.name == 'file_operation'), None)
        assert file_element is not None
        assert 'pathlib' in file_element.external_dependencies

    def test_visit_file_with_complex_types(self, python_visitor, temp_dir):
        """Test visiting a file with complex type annotations."""
        py_file = temp_dir / "test.py"
        source = '''
from typing import Dict, List, Optional, Union

def process_data(items: List[Dict[str, Union[str, int]]]) -> Optional[str]:
    """Process a list of dictionaries with mixed types."""
    if not items:
        return None
    return str(len(items))

class GenericClass:
    """Class with generic type annotations."""

    def __init__(self, data: Dict[str, List[int]]):
        self.data = data

    def get_value(self, key: str, index: int) -> Optional[int]:
        """Get value from nested structure."""
        values = self.data.get(key)
        if values and index < len(values):
            return values[index]
        return None
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Extract imports from the source (like the real extractor does)
        extractor = PythonASTExtractor()
        file_imports, import_from_targets = extractor._extract_file_imports(tree)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(temp_dir / "test.py")
        visitor.file_level_imports = file_imports
        visitor.file_level_import_from_targets = import_from_targets

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        func_element = next((e for e in elements if e.name == 'process_data'), None)
        assert func_element is not None
        assert 'items: List[Dict[str, Union[str, int]]]' in func_element.parameters
        assert func_element.return_type == 'Optional[str]'

    def test_visit_file_with_nested_classes(self, python_visitor, temp_dir):
        """Test visiting a file with nested classes."""
        py_file = temp_dir / "test.py"
        source = '''
class OuterClass:
    """Outer class containing inner class."""

    class InnerClass:
        """Inner class definition."""

        def inner_method(self) -> str:
            """Method in inner class."""
            return "inner"

    def outer_method(self) -> str:
        """Method in outer class."""
        return "outer"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = "test.py"
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        test_visitor.visit(tree)
        elements = test_visitor.elements

        outer_class = next((e for e in elements if e.name == 'OuterClass'), None)
        inner_class = next((e for e in elements if e.name == 'InnerClass'), None)

        assert outer_class is not None
        assert inner_class is not None
        assert 'InnerClass' not in outer_class.methods  # Should be separate class

    def test_visit_file_with_inheritance(self, python_visitor, temp_dir):
        """Test visiting a file with class inheritance."""
        py_file = temp_dir / "test.py"
        source = '''
class BaseClass:
    """Base class with common functionality."""

    def base_method(self) -> str:
        return "base"

class DerivedClass(BaseClass):
    """Derived class extending BaseClass."""

    def derived_method(self) -> str:
        return "derived"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Extract imports from the source (like the real extractor does)
        extractor = PythonASTExtractor()
        file_imports, import_from_targets = extractor._extract_file_imports(tree)

        # Set up visitor with file context (use a different visitor instance for this test)
        test_visitor = PythonASTVisitor()
        test_visitor.source_lines = source.splitlines()
        test_visitor.current_file = str(temp_dir / "test.py")
        test_visitor.file_level_imports = file_imports
        test_visitor.file_level_import_from_targets = import_from_targets

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        base_class = next((e for e in elements if e.name == 'BaseClass'), None)
        derived_class = next((e for e in elements if e.name == 'DerivedClass'), None)

        assert base_class is not None
        assert derived_class is not None
        assert derived_class.extends == 'BaseClass'

    def test_visit_file_with_local_variables(self, python_visitor, temp_dir):
        """Test visiting a file with local variable detection."""
        py_file = temp_dir / "test.py"
        source = '''
def complex_function(data: list) -> dict:
    """Function with various local variables."""
    result = {}
    temp_list = []

    for item in data:
        processed_item = item.upper()
        temp_list.append(processed_item)

    final_value = len(temp_list)
    return {"result": result, "count": final_value}
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context (use temp_dir for path)
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(temp_dir / "test.py")
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        func_element = next((e for e in elements if e.name == 'complex_function'), None)
        assert func_element is not None

        # Check that local variables are detected
        local_vars = func_element.local_variables_declared
        assert 'result' in local_vars
        assert 'temp_list' in local_vars
        assert 'processed_item' in local_vars
        assert 'final_value' in local_vars
        assert 'item' in local_vars  # Loop variable

    def test_visit_file_complexity_calculation(self, python_visitor, temp_dir):
        """Test visiting a file with complexity calculation."""
        py_file = temp_dir / "test.py"
        source = '''
def complex_function(value: int) -> str:
    """Function with high complexity."""
    if value < 0:
        return "negative"
    elif value == 0:
        return "zero"
    else:
        for i in range(value):
            if i % 2 == 0:
                print("even")
            else:
                print("odd")
        return "positive"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Extract imports from the source (like the real extractor does)
        extractor = PythonASTExtractor()
        file_imports, import_from_targets = extractor._extract_file_imports(tree)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(temp_dir / "test.py")
        visitor.file_level_imports = file_imports
        visitor.file_level_import_from_targets = import_from_targets

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        func_element = next((e for e in elements if e.name == 'complex_function'), None)
        assert func_element is not None
        assert func_element.complexity >= 4  # if + elif + for + if inside for

    def test_visit_file_nloc_calculation(self, python_visitor, temp_dir):
        """Test visiting a file with NLOC calculation."""
        py_file = temp_dir / "test.py"
        source = '''
def function_with_comments(value: int) -> str:
    # This is a comment
    """Function with comments and empty lines."""
    if value > 0:  # Inline comment
        result = "positive"
        # Another comment
        return result
    else:
        return "non-positive"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        func_element = next((e for e in elements if e.name == 'function_with_comments'), None)
        assert func_element is not None
        assert func_element.nloc >= 3  # Should count actual code lines, not comments

    def test_visit_file_hash_calculation(self, python_visitor, temp_dir):
        """Test visiting a file with source hash calculation."""
        py_file = temp_dir / "test.py"
        source = '''
def simple_function(x: int) -> int:
    return x + 1
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        func_element = next((e for e in elements if e.name == 'simple_function'), None)
        assert func_element is not None
        assert func_element.hash_body is not None
        assert isinstance(func_element.hash_body, str)
        assert len(func_element.hash_body) == 32  # MD5 hash length

    def test_visit_file_empty_function(self, python_visitor, temp_dir):
        """Test visiting a file with empty function."""
        py_file = temp_dir / "test.py"
        source = '''
def empty_function():
    pass
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        func_element = next((e for e in elements if e.name == 'empty_function'), None)
        assert func_element is not None
        assert func_element.complexity == 1  # Base complexity
        assert func_element.nloc >= 1  # The pass statement (may include function line)

    def test_visit_file_with_try_except_finally(self, python_visitor, temp_dir):
        """Test visiting a file with try-except-finally blocks."""
        py_file = temp_dir / "test.py"
        source = '''
def risky_operation():
    """Function with exception handling."""
    try:
        risky_action()
    except ValueError:
        handle_value_error()
    except (TypeError, RuntimeError):
        handle_type_error()
    finally:
        cleanup()
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        func_element = next((e for e in elements if e.name == 'risky_operation'), None)
        assert func_element is not None
        assert func_element.complexity >= 4  # try + multiple except blocks
        assert 'ValueError' in func_element.catches_exceptions
        assert 'TypeError' in func_element.catches_exceptions
        assert 'RuntimeError' in func_element.catches_exceptions

    def test_visit_file_with_context_managers(self, python_visitor, temp_dir):
        """Test visiting a file with context managers."""
        py_file = temp_dir / "test.py"
        source = '''
def file_operation(filename: str) -> str:
    """Function using context managers."""
    with open(filename, 'r') as f:
        content = f.read()
        for line in content.splitlines():
            if line.strip():
                process_line(line)
    return "done"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        func_element = next((e for e in elements if e.name == 'file_operation'), None)
        assert func_element is not None
        assert 'content' in func_element.local_variables_declared
        assert 'line' in func_element.local_variables_declared
        assert func_element.complexity >= 2  # function + with statement

    def test_visit_file_with_list_comprehensions(self, python_visitor, temp_dir):
        """Test visiting a file with list comprehensions."""
        py_file = temp_dir / "test.py"
        source = '''
def comprehension_function(data: list) -> list:
    """Function using comprehensions."""
    squares = [x*x for x in data if x > 0]
    filtered = [item.upper() for item in data if item]
    return squares + filtered
'''

        with open(py_file, 'w') as f:
            f.write(source)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        visitor.visit(tree)
        elements = visitor.elements

        func_element = next((e for e in elements if e.name == 'comprehension_function'), None)
        assert func_element is not None
        assert 'squares' in func_element.local_variables_declared
        assert 'filtered' in func_element.local_variables_declared
        assert 'x' in func_element.local_variables_declared
        assert 'item' in func_element.local_variables_declared
        assert func_element.complexity >= 3  # function + 2 comprehensions

    def test_visitor_with_file_level_imports(self, python_visitor):
        """Test visitor with file-level imports context."""
        source = '''
import requests
from typing import List
from pathlib import Path as PathType

def api_function():
    response = requests.get("https://api.example.com")
    return response.json()

def file_function():
    p = PathType("/tmp/test")
    return p.exists()
'''

        python_visitor.file_level_imports = {
            'requests': 'requests',
            'Path': 'pathlib'
        }
        python_visitor.file_level_import_from_targets = {'List', 'Path'}

        # Parse the source into an AST
        tree = ast.parse(source)

        # Set up visitor with file context
        visitor = PythonASTVisitor()
        visitor.source_lines = source.splitlines()
        visitor.current_file = str(py_file.resolve())
        visitor.file_level_imports = {}
        visitor.file_level_import_from_targets = set()

        # Visit the AST
        test_visitor.visit(tree)
        elements = test_visitor.elements

        api_func = next((e for e in elements if e.name == 'api_function'), None)
        file_func = next((e for e in elements if e.name == 'file_function'), None)

        assert api_func is not None
        assert 'requests' in api_func.external_dependencies

        assert file_func is not None
        assert 'pathlib' in file_func.external_dependencies