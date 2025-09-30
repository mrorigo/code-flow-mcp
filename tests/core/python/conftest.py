"""
Pytest configuration and fixtures for Python AST tests.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from typing import Generator, List

from code_flow_graph.core.python_extractor import PythonASTExtractor, PythonASTVisitor


@pytest.fixture
def python_extractor() -> PythonASTExtractor:
    """Create a PythonASTExtractor instance for testing."""
    return PythonASTExtractor()


@pytest.fixture
def python_visitor() -> PythonASTVisitor:
    """Create a PythonASTVisitor instance for testing."""
    return PythonASTVisitor()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())

    yield temp_path

    # Cleanup after test
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_python_file(temp_dir: Path) -> Path:
    """Create a sample Python file for testing."""
    py_file = temp_dir / "sample.py"
    source = '''"""Sample Python file for testing."""

import os
from typing import List, Optional

def sample_function(name: str, items: List[str] = None) -> str:
    """A sample function for testing."""
    if items is None:
        items = []
    return f"Hello {name}, you have {len(items)} items"

class SampleClass:
    """A sample class for testing."""

    def __init__(self, value: int = 42):
        self.value = value

    def get_value(self) -> int:
        """Get the value."""
        return self.value

    def set_value(self, value: int) -> None:
        """Set the value."""
        self.value = value
'''

    with open(py_file, 'w') as f:
        f.write(source)

    return py_file


@pytest.fixture
def complex_python_project(temp_dir: Path) -> Path:
    """Create a complex Python project structure for testing."""
    project_root = temp_dir / "complex_project"
    project_root.mkdir()

    # Create main package
    main_pkg = project_root / "src" / "myapp"
    main_pkg.mkdir(parents=True)
    (main_pkg / "__init__.py").write_text('"""Main application package."""')

    # Create submodules
    core_py = main_pkg / "core.py"
    core_py.write_text('''"""Core functionality."""

import asyncio
from typing import Dict, List, Optional

class DataProcessor:
    """Core data processing class."""

    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.processed_count = 0

    async def process_batch(self, items: List[Dict]) -> List[Dict]:
        """Process a batch of items asynchronously."""
        results = []
        for item in items:
            try:
                processed = await self._process_item(item)
                results.append(processed)
                self.processed_count += 1
            except Exception as e:
                print(f"Error processing item: {e}")
        return results

    async def _process_item(self, item: Dict) -> Dict:
        """Process a single item."""
        await asyncio.sleep(0.01)  # Simulate async work
        return {"processed": True, **item}
''')

    # Create API module
    api_py = main_pkg / "api.py"
    api_py.write_text('''"""API endpoints."""

from fastapi import FastAPI, HTTPException
from .core import DataProcessor

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/process")
async def process_endpoint(data: dict):
    """Process data endpoint."""
    processor = DataProcessor({})
    result = await processor.process_batch([data])
    return result
''')

    # Create tests module
    tests_py = main_pkg / "tests.py"
    tests_py.write_text('''"""Unit tests."""

import pytest
from .core import DataProcessor

def test_data_processor():
    """Test data processor."""
    processor = DataProcessor({"test": "config"})

    # Test synchronous initialization
    assert processor.config == {"test": "config"}
    assert processor.processed_count == 0

@pytest.mark.asyncio
async def test_async_processing():
    """Test async processing."""
    processor = DataProcessor({})
    result = await processor.process_batch([{"test": "data"}])

    assert len(result) == 1
    assert result[0]["processed"] is True
''')

    # Create configuration files
    config_py = main_pkg / "config.py"
    config_py.write_text('''"""Configuration management."""

import os

class Config:
    """Configuration manager."""

    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///app.db")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.port = int(os.getenv("PORT", "8000"))

    def get(self, key: str, default=None):
        """Get configuration value."""
        return getattr(self, key, default)

config = Config()
''')

    # Create requirements.txt
    requirements_txt = project_root / "requirements.txt"
    requirements_txt.write_text('''fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
alembic==1.12.1
pytest==7.4.3
pytest-asyncio==0.21.1
''')

    # Create setup.py
    setup_py = project_root / "setup.py"
    setup_py.write_text('''"""Package setup."""

from setuptools import setup, find_packages

setup(
    name="myapp",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "pytest-asyncio==0.21.1",
        ],
    },
)
''')

    return project_root


@pytest.fixture
def python_files_with_imports(temp_dir: Path) -> dict:
    """Create Python files with various import patterns."""
    files = {}

    # File with standard imports
    standard_imports = temp_dir / "standard_imports.py"
    standard_imports.write_text('''import os
import sys
import json

def use_standard_imports():
    """Use standard library imports."""
    current_dir = os.getcwd()
    version = sys.version
    data = json.loads('{}')
    return current_dir, version
''')
    files['standard'] = standard_imports

    # File with from imports
    from_imports = temp_dir / "from_imports.py"
    from_imports.write_text('''from typing import List, Dict, Optional
from collections import defaultdict
from pathlib import Path

def use_from_imports():
    """Use from imports."""
    items: List[str] = ["a", "b", "c"]
    mapping: Dict[str, int] = defaultdict(int)
    p = Path("/tmp")
    return items, mapping, p
''')
    files['from_imports'] = from_imports

    # File with aliased imports
    aliased_imports = temp_dir / "aliased_imports.py"
    aliased_imports.write_text('''import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def use_aliased_imports():
    """Use aliased imports."""
    df = pd.DataFrame({"col": [1, 2, 3]})
    arr = np.array([1, 2, 3])
    plt.figure()
    return df, arr
''')
    files['aliased'] = aliased_imports

    return files


@pytest.fixture
def python_files_with_decorators(temp_dir: Path) -> dict:
    """Create Python files with various decorator patterns."""
    files = {}

    # File with function decorators
    func_decorators = temp_dir / "function_decorators.py"
    func_decorators.write_text('''from functools import lru_cache, wraps
import time

def timer(func):
    """Timer decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Executed in {time.time() - start:.2f}s")
        return result
    return wrapper

@timer
@lru_cache(maxsize=128)
def cached_function(n: int) -> int:
    """Function with multiple decorators."""
    time.sleep(0.1)
    return n * 2

def simple_decorator(func):
    """Simple decorator."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@simple_decorator
def decorated_function():
    """Function with decorator."""
    return "decorated"
''')
    files['function'] = func_decorators

    # File with class decorators
    class_decorators = temp_dir / "class_decorators.py"
    class_decorators.write_text('''from dataclasses import dataclass
from functools import total_ordering

@dataclass
class DataClass:
    """Data class with decorator."""
    name: str
    value: int

@total_ordering
class OrderedClass:
    """Class with ordering decorator."""

    def __init__(self, value: int):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

def class_decorator(cls):
    """Custom class decorator."""
    cls.decorated = True
    return cls

@class_decorator
class CustomDecoratedClass:
    """Class with custom decorator."""
    pass
''')
    files['class'] = class_decorators

    return files


@pytest.fixture
def python_files_with_exceptions(temp_dir: Path) -> dict:
    """Create Python files with various exception handling patterns."""
    files = {}

    # File with basic exception handling
    basic_exceptions = temp_dir / "basic_exceptions.py"
    basic_exceptions.write_text('''def basic_exception_handling():
    """Basic exception handling."""
    try:
        risky_operation()
        return "success"
    except ValueError as e:
        return f"value_error: {e}"
    except TypeError:
        return "type_error"
    except Exception as e:
        return f"other_error: {e}"

def multiple_exception_types():
    """Multiple exception types."""
    try:
        might_fail()
    except (ValueError, TypeError) as e:
        return f"multiple: {e}"
    except Exception:
        return "general"

def nested_exceptions():
    """Nested exception handling."""
    try:
        try:
            inner_risky()
        except ValueError:
            handle_inner()
            raise
    except RuntimeError:
        return "handled"
''')
    files['basic'] = basic_exceptions

    # File with custom exceptions
    custom_exceptions = temp_dir / "custom_exceptions.py"
    custom_exceptions.write_text('''class CustomError(Exception):
    """Custom exception class."""
    pass

class ValidationError(CustomError):
    """Validation exception."""
    pass

def function_with_custom_exceptions():
    """Function with custom exceptions."""
    try:
        validate_data({})
    except ValidationError as e:
        print(f"Validation failed: {e}")
        raise CustomError("Processing failed") from e
''')
    files['custom'] = custom_exceptions

    return files


@pytest.fixture
def python_files_with_complex_types(temp_dir: Path) -> dict:
    """Create Python files with complex type annotations."""
    files = {}

    # File with generic types
    generic_types = temp_dir / "generic_types.py"
    generic_types.write_text('''from typing import List, Dict, Optional, Union, Tuple, Generic, TypeVar

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

def generic_function(
    items: List[Dict[str, T]],
    mapping: Dict[K, V]
) -> Dict[str, List[T]]:
    """Function with generic types."""
    return {"items": [item for item in items]}

def union_types(value: Union[int, str, None]) -> str:
    """Function with union types."""
    if value is None:
        return "none"
    return str(value)

def complex_nested_types(
    data: Dict[str, List[Tuple[int, str]]]
) -> List[Dict[str, Union[int, str]]]:
    """Function with complex nested types."""
    return [{"result": item[0]} for item in data.get("items", [])]

class GenericClass(Generic[T]):
    """Generic class."""

    def __init__(self, value: T):
        self.value = value

    def get(self) -> T:
        return self.value

    def process(self, func: Callable[[T], T]) -> T:
        return func(self.value)
''')
    files['generic'] = generic_types

    return files


@pytest.fixture
def sample_python_source() -> str:
    """Provide sample Python source code for testing."""
    return '''"""Sample Python source for testing."""

import os
from typing import List, Dict, Optional

def sample_function(
    name: str,
    items: List[str] = None,
    count: Optional[int] = None
) -> Dict[str, str]:
    """Sample function with comprehensive features."""
    if items is None:
        items = []
    if count is None:
        count = len(items)

    result = {
        "name": name,
        "count": count,
        "status": "processed"
    }

    for item in items:
        if len(item) > 5:
            result["long_items"] = result.get("long_items", 0) + 1

    return result

class SampleClass:
    """Sample class for testing."""

    def __init__(self, value: int = 42):
        self.value = value
        self._internal = []

    def get_value(self) -> int:
        """Get the value."""
        return self.value

    def set_value(self, value: int) -> None:
        """Set the value."""
        self.value = value

    def add_item(self, item: str) -> None:
        """Add item to internal list."""
        self._internal.append(item)

    def get_items(self) -> List[str]:
        """Get internal items."""
        return self._internal.copy()
'''


@pytest.fixture
def python_visitor_with_imports(python_visitor) -> PythonASTVisitor:
    """Create a PythonASTVisitor with file-level imports set up."""
    python_visitor.file_level_imports = {
        'os': 'os',
        'List': 'typing',
        'Dict': 'typing',
        'Optional': 'typing'
    }
    python_visitor.file_level_import_from_targets = {'List', 'Dict', 'Optional'}
    return python_visitor


@pytest.fixture
def python_files_for_performance_test(temp_dir: Path) -> List[Path]:
    """Create multiple Python files for performance testing."""
    files = []

    for i in range(10):
        py_file = temp_dir / f"performance_test_{i}.py"

        # Create file with multiple functions and classes
        content = []

        # Add header comment
        content.append(f'"""Performance test module {i}."""\n')

        # Add imports
        content.append('import os\nfrom typing import List, Dict\n')

        # Add multiple functions
        for func_num in range(5):
            content.append(f'''
def function_{i}_{func_num}(data: List[Dict]) -> int:
    """Function {func_num} in module {i}."""
    count = 0
    for item in data:
        if "active" in item:
            count += 1
    return count
''')

        # Add a class
        content.append(f'''
class TestClass_{i}:
    """Test class {i}."""

    def __init__(self, value: int = {i}):
        self.value = value
        self.items = []

    def add_item(self, item: str) -> None:
        """Add item."""
        self.items.append(item)

    def get_value(self) -> int:
        """Get value."""
        return self.value
''')

        py_file.write_text(''.join(content))
        files.append(py_file)

    return files


@pytest.fixture
def python_project_with_gitignore(temp_dir: Path) -> Path:
    """Create a Python project with .gitignore file."""
    project_root = temp_dir / "project_with_gitignore"
    project_root.mkdir()

    # Create .gitignore
    gitignore = project_root / ".gitignore"
    gitignore.write_text('''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
*.log
.cache/
temp/
tmp/
''')

    # Create some files that should be ignored
    cache_dir = project_root / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "module.cpython-38.pyc").write_text("")

    temp_dir_proj = project_root / "temp"
    temp_dir_proj.mkdir()
    (temp_dir_proj / "temp_file.py").write_text('def temp_function(): pass')

    # Create files that should NOT be ignored
    (project_root / "main.py").write_text('def main(): pass')
    (project_root / "utils.py").write_text('def utility(): pass')

    return project_root


@pytest.fixture
def python_files_with_unicode(temp_dir: Path) -> dict:
    """Create Python files with Unicode content."""
    files = {}

    # File with Unicode in strings and comments
    unicode_strings = temp_dir / "unicode_strings.py"
    unicode_strings.write_text('''# -*- coding: utf-8 -*-
"""Module with Unicode strings."""

def greet_user(name: str) -> str:
    """Greet user with Unicode."""
    café = "café"  # Unicode in variable name would be invalid
    return f"Hello {name}! Would you like some café?"

def process_text(text: str) -> str:
    """Process text with emojis."""
    return f"Processed: {text} 🚀"

class Café:
    """Class with Unicode in name and docstring."""
    pass
''')
    files['strings'] = unicode_strings

    # File with various Unicode characters
    unicode_chars = temp_dir / "unicode_chars.py"
    unicode_chars.write_text('''"""Various Unicode characters."""

def math_symbols() -> str:
    """Function with math symbols."""
    result = "α + β = γ"
    return result

def international_text() -> str:
    """International text."""
    french = "Bonjour le monde"
    spanish = "Hola mundo"
    german = "Guten Tag Welt"
    japanese = "こんにちは世界"
    return f"{french}|{spanish}|{german}|{japanese}"
''')
    files['chars'] = unicode_chars

    return files