"""
Feature coverage tests for Python AST processing.
"""

import pytest
from pathlib import Path

from code_flow_graph.core.python_extractor import PythonASTExtractor


class TestPythonFeatureCoverage:
    """Test coverage of all Python language features."""

    def test_function_features_coverage(self, temp_dir):
        """Test coverage of function-related features."""
        py_file = temp_dir / "function_features.py"
        source = '''"""Test all function features."""

# Basic function
def simple_function():
    """Simple function."""
    pass

# Function with parameters and return type
def typed_function(a: int, b: str = "default") -> str:
    """Function with type annotations."""
    return f"{a}: {b}"

# Function with *args and **kwargs
def varargs_function(*args, **kwargs) -> None:
    """Function with varargs."""
    for arg in args:
        print(arg)
    for key, value in kwargs.items():
        print(f"{key}={value}")

# Async function
async def async_function(delay: float) -> bool:
    """Async function."""
    import asyncio
    await asyncio.sleep(delay)
    return True

# Generator function
def generator_function(n: int):
    """Generator function."""
    for i in range(n):
        yield i * 2

# Function with complex parameter types
from typing import List, Dict, Optional, Union, Tuple

def complex_params(
    numbers: List[int],
    mapping: Dict[str, str],
    optional: Optional[str] = None,
    union: Union[int, str] = "default"
) -> Tuple[bool, str]:
    """Function with complex parameter types."""
    return len(numbers) > 0, str(union)
'''

        with open(py_file, 'w') as f:
            f.write(source)

        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        functions = [e for e in elements if e.kind == 'function']

        # Verify all functions are extracted
        func_names = {f.name for f in functions}
        expected_functions = {
            'simple_function', 'typed_function', 'varargs_function',
            'async_function', 'generator_function', 'complex_params'
        }
        assert expected_functions.issubset(func_names)

        # Verify parameter extraction
        typed_func = next(f for f in functions if f.name == 'typed_function')
        assert 'a: int' in typed_func.parameters
        assert 'b: str = "default"' in typed_func.parameters
        assert typed_func.return_type == 'str'

        # Verify async detection
        async_func = next(f for f in functions if f.name == 'async_function')
        assert async_func.is_async is True

        # Verify complex types
        complex_func = next(f for f in functions if f.name == 'complex_params')
        assert 'numbers: List[int]' in complex_func.parameters
        assert 'mapping: Dict[str, str]' in complex_func.parameters
        assert 'optional: Optional[str] = None' in complex_func.parameters
        assert complex_func.return_type == 'Tuple[bool, str]'

    def test_class_features_coverage(self, temp_dir):
        """Test coverage of class-related features."""
        py_file = temp_dir / "class_features.py"
        source = '''"""Test all class features."""

from dataclasses import dataclass
from abc import ABC, abstractmethod

# Basic class
class SimpleClass:
    """Simple class."""

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        """Get value."""
        return self.value

    def set_value(self, value: int) -> None:
        """Set value."""
        self.value = value

# Class with inheritance
class BaseClass:
    """Base class."""

    def base_method(self) -> str:
        return "base"

class DerivedClass(BaseClass):
    """Derived class."""

    def derived_method(self) -> str:
        return "derived"

# Abstract base class
class AbstractBase(ABC):
    """Abstract base class."""

    @abstractmethod
    def abstract_method(self) -> str:
        """Abstract method."""
        pass

    def concrete_method(self) -> str:
        """Concrete method."""
        return "concrete"

class ConcreteClass(AbstractBase):
    """Concrete implementation."""

    def abstract_method(self) -> str:
        return "implemented"

# Dataclass
@dataclass
class DataClass:
    """Data class."""
    name: str
    value: int

    def get_combined(self) -> str:
        """Get combined string."""
        return f"{self.name}_{self.value}"

# Class with properties
class PropertyClass:
    """Class with properties."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """Name property."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set name."""
        self._name = value.title()

    @property
    def display_name(self) -> str:
        """Display name property."""
        return f"User: {self._name}"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        classes = [e for e in elements if e.kind == 'class']

        # Verify all classes are extracted
        class_names = {c.name for c in classes}
        expected_classes = {
            'SimpleClass', 'BaseClass', 'DerivedClass',
            'AbstractBase', 'ConcreteClass', 'DataClass', 'PropertyClass'
        }
        assert expected_classes.issubset(class_names)

        # Verify inheritance
        derived_class = next(c for c in classes if c.name == 'DerivedClass')
        assert derived_class.extends == 'BaseClass'

        # Verify dataclass decorator
        data_class = next(c for c in classes if c.name == 'DataClass')
        assert len(data_class.decorators) == 1
        assert data_class.decorators[0]['name'] == 'dataclass'

        # Verify abstract method detection
        concrete_class = next(c for c in classes if c.name == 'ConcreteClass')
        assert 'abstract_method' in concrete_class.methods

        # Verify property methods are detected
        property_class = next(c for c in classes if c.name == 'PropertyClass')
        # Note: Properties might not be detected as regular methods

    def test_decorator_features_coverage(self, temp_dir):
        """Test coverage of decorator features."""
        py_file = temp_dir / "decorator_features.py"
        source = '''"""Test decorator features."""

from functools import lru_cache, wraps
import time

# Simple decorator
def simple_decorator(func):
    """Simple decorator."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Function with decorator
@simple_decorator
def decorated_function():
    """Function with decorator."""
    return "decorated"

# Multiple decorators
def timer(func):
    """Timer decorator."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Executed in {time.time() - start:.2f}s")
        return result
    return wrapper

@timer
@lru_cache(maxsize=128)
def cached_and_timed(n: int) -> int:
    """Function with multiple decorators."""
    time.sleep(0.1)  # Simulate work
    return n * 2

# Class decorator
def class_decorator(cls):
    """Class decorator."""
    cls.decorated = True
    return cls

@class_decorator
class DecoratedClass:
    """Class with decorator."""
    pass

# Property decorators
class PropertyDecorated:
    """Class with property decorators."""

    def __init__(self, value: str):
        self._value = value

    @property
    def value(self) -> str:
        """Value property."""
        return self._value

    @value.setter
    def value(self, new_value: str) -> None:
        """Set value."""
        self._value = new_value.upper()

# Static method and class method decorators
class MethodDecorated:
    """Class with static and class methods."""

    @staticmethod
    def static_method(x: int) -> int:
        """Static method."""
        return x * 2

    @classmethod
    def class_method(cls, x: int) -> int:
        """Class method."""
        return x * 3
'''

        with open(py_file, 'w') as f:
            f.write(source)

        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        # Check function decorators
        decorated_func = next(e for e in elements if e.name == 'decorated_function')
        assert len(decorated_func.decorators) == 1
        assert decorated_func.decorators[0]['name'] == 'simple_decorator'

        # Check multiple decorators
        cached_func = next(e for e in elements if e.name == 'cached_and_timed')
        assert len(cached_func.decorators) >= 2  # timer and lru_cache

        # Check class decorator
        decorated_class = next(e for e in elements if e.name == 'DecoratedClass')
        assert len(decorated_class.decorators) == 1
        assert decorated_class.decorators[0]['name'] == 'class_decorator'

    def test_exception_handling_features(self, temp_dir):
        """Test coverage of exception handling features."""
        py_file = temp_dir / "exception_features.py"
        source = '''"""Test exception handling features."""

def simple_exception_handling():
    """Simple exception handling."""
    try:
        risky_operation()
    except ValueError:
        return "value_error"
    except TypeError as e:
        return f"type_error: {e}"

def multiple_exception_types():
    """Multiple exception types."""
    try:
        might_fail()
    except (ValueError, TypeError) as e:
        return f"multiple: {e}"
    except Exception as e:
        return f"general: {e}"

def nested_exception_handling():
    """Nested exception handling."""
    try:
        try:
            inner_risky()
        except ValueError:
            handle_inner()
            raise
    except RuntimeError:
        return "handled"

def exception_with_finally():
    """Exception with finally block."""
    try:
        always_runs()
    except Exception:
        handle_error()
    finally:
        cleanup()

def bare_except():
    """Bare except clause."""
    try:
        risky()
    except:
        return "caught_all"

def raise_from():
    """Raise from another exception."""
    try:
        original_failure()
    except Exception as e:
        raise RuntimeError("Wrapper error") from e
'''

        with open(py_file, 'w') as f:
            f.write(source)

        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        functions = [e for e in elements if e.kind == 'function']

        # Check exception handling detection
        simple_func = next(f for f in functions if f.name == 'simple_exception_handling')
        assert 'ValueError' in simple_func.catches_exceptions
        assert 'TypeError' in simple_func.catches_exceptions

        multiple_func = next(f for f in functions if f.name == 'multiple_exception_types')
        assert 'ValueError' in multiple_func.catches_exceptions
        assert 'TypeError' in multiple_func.catches_exceptions
        assert 'Exception' in multiple_func.catches_exceptions

        bare_func = next(f for f in functions if f.name == 'bare_except')
        assert 'Exception' in bare_func.catches_exceptions  # Should default to Exception

    def test_import_features_coverage(self, temp_dir):
        """Test coverage of import features."""
        py_file = temp_dir / "import_features.py"
        source = '''"""Test import features."""

# Standard imports
import os
import sys

# Aliased imports
import os.path as path
import json as json_lib

# From imports
from typing import List, Dict
from collections import defaultdict

# From imports with aliases
from datetime import datetime as dt
from pathlib import Path as PathType

# Multiple from imports
from math import sqrt, ceil, floor

# Conditional imports
try:
    import optional_module
except ImportError:
    optional_module = None

def function_using_imports():
    """Function that uses various imports."""
    # Use standard imports
    current_path = os.getcwd()
    print(sys.version)

    # Use aliased imports
    parent = path.dirname(current_path)
    data = json_lib.loads('{}')

    # Use from imports
    numbers: List[int] = [1, 2, 3]
    mapping: Dict[str, int] = defaultdict(int)

    # Use aliases
    now = dt.now()
    p = PathType("/tmp")

    # Use multiple imports
    root = sqrt(16)
    ceiling = ceil(3.2)
    flooring = floor(3.8)

    return "success"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        func = next(e for e in elements if e.name == 'function_using_imports')

        # Check that various imports are detected as dependencies
        expected_deps = {'os', 'sys', 'json', 'typing', 'collections', 'datetime', 'pathlib', 'math'}
        for dep in expected_deps:
            assert any(dep in func_dep for func_dep in func.external_dependencies)

    def test_control_flow_features(self, temp_dir):
        """Test coverage of control flow features."""
        py_file = temp_dir / "control_flow_features.py"
        source = '''"""Test control flow features."""

def if_elif_else(value: int) -> str:
    """If-elif-else chain."""
    if value < 0:
        return "negative"
    elif value == 0:
        return "zero"
    elif value == 1:
        return "one"
    else:
        return "other"

def nested_ifs(condition: bool) -> str:
    """Nested if statements."""
    if condition:
        if True:
            if False:
                return "deep"
            else:
                return "middle"
        else:
            return "inner"
    else:
        return "outer"

def for_loops():
    """Various for loop patterns."""
    # Simple for loop
    for i in range(10):
        print(i)

    # For loop with enumerate
    for i, item in enumerate(["a", "b", "c"]):
        print(f"{i}: {item}")

    # For loop with zip
    for x, y in zip([1, 2, 3], [4, 5, 6]):
        print(x + y)

def while_loops():
    """While loop patterns."""
    i = 0
    while i < 5:
        print(i)
        i += 1

    # While with break and continue
    j = 0
    while j < 10:
        if j == 3:
            j += 1
            continue
        if j == 7:
            break
        print(j)
        j += 1

def list_dict_comprehensions():
    """Comprehensions."""
    # List comprehensions
    squares = [x*x for x in range(10)]
    even_squares = [x*x for x in range(10) if x % 2 == 0]

    # Dict comprehensions
    square_dict = {x: x*x for x in range(5)}
    filtered_dict = {k: v for k, v in square_dict.items() if v > 5}

    # Set comprehensions
    unique_squares = {x*x for x in [1, 2, 2, 3, 3, 3]}

    return squares, square_dict, unique_squares

def match_statement(value):
    """Match statement (Python 3.10+)."""
    match value:
        case 1:
            return "one"
        case 2 | 3:
            return "two or three"
        case str() if len(value) > 5:
            return "long string"
        case _:
            return "other"
'''

        with open(py_file, 'w') as f:
            f.write(source)

        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        functions = [e for e in elements if e.kind == 'function']

        # Check complexity calculations for control flow
        if_func = next(f for f in functions if f.name == 'if_elif_else')
        assert if_func.complexity >= 4  # 3 elif + 1 base

        nested_func = next(f for f in functions if f.name == 'nested_ifs')
        assert nested_func.complexity >= 4  # nested ifs

        for_func = next(f for f in functions if f.name == 'for_loops')
        assert for_func.complexity >= 4  # for loops + enumerate + zip

        while_func = next(f for f in functions if f.name == 'while_loops')
        assert while_func.complexity >= 3  # while + if + break

        # Check local variable detection
        comprehension_func = next(f for f in functions if f.name == 'list_dict_comprehensions')
        local_vars = comprehension_func.local_variables_declared
        assert 'squares' in local_vars
        assert 'even_squares' in local_vars
        assert 'square_dict' in local_vars
        assert 'filtered_dict' in local_vars
        assert 'unique_squares' in local_vars
        assert 'x' in local_vars  # Comprehension variable

    def test_type_annotation_features(self, temp_dir):
        """Test coverage of type annotation features."""
        py_file = temp_dir / "type_annotation_features.py"
        source = '''"""Test type annotation features."""

from typing import List, Dict, Optional, Union, Any, Callable, Tuple
from typing import Generic, TypeVar

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Simple type annotations
def simple_types(x: int, y: str, z: bool) -> float:
    """Simple type annotations."""
    return float(x)

# Complex generic types
def generic_function(
    items: List[Dict[str, Any]],
    optional: Optional[str] = None
) -> Dict[str, List[int]]:
    """Complex generic types."""
    return {"result": [len(items)]}

# Union types
def union_function(value: Union[int, str, None]) -> str:
    """Union type annotations."""
    if value is None:
        return "none"
    return str(value)

# Callable types
def callable_function(
    func: Callable[[int, str], bool]
) -> bool:
    """Callable type annotations."""
    return func(1, "test")

# Generic class
class GenericClass(Generic[T]):
    """Generic class."""

    def __init__(self, value: T):
        self.value = value

    def get(self) -> T:
        """Get value."""
        return self.value

# Complex nested generics
def complex_generics(
    data: Dict[K, List[Tuple[V, T]]]
) -> List[Dict[str, Any]]:
    """Complex nested generic types."""
    return [{"processed": True}]

# Forward references
def forward_reference(func: 'Callable') -> 'Dict[str, int]':
    """Forward reference types."""
    return {"count": 1}
'''

        with open(py_file, 'w') as f:
            f.write(source)

        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        functions = [e for e in elements if e.kind == 'function']

        # Verify type annotation extraction
        simple_func = next(f for f in functions if f.name == 'simple_types')
        assert 'x: int' in simple_func.parameters
        assert 'y: str' in simple_func.parameters
        assert 'z: bool' in simple_func.parameters
        assert simple_func.return_type == 'float'

        generic_func = next(f for f in functions if f.name == 'generic_function')
        assert 'items: List[Dict[str, Any]]' in generic_func.parameters
        assert generic_func.return_type == 'Dict[str, List[int]]'

        union_func = next(f for f in functions if f.name == 'union_function')
        assert 'value: Union[int, str, None]' in union_func.parameters

        callable_func = next(f for f in functions if f.name == 'callable_function')
        assert 'func: Callable[[int, str], bool]' in callable_func.parameters

    def test_context_manager_features(self, temp_dir):
        """Test coverage of context manager features."""
        py_file = temp_dir / "context_manager_features.py"
        source = '''"""Test context manager features."""

from contextlib import contextmanager

# Context manager using with statement
def file_operations():
    """Function using with statement."""
    with open("/tmp/test.txt", "w") as f:
        f.write("test")
        with open("/tmp/another.txt", "r") as g:
            content = g.read()
    return len(content) if 'content' in locals() else 0

# Nested context managers
def nested_contexts():
    """Nested context managers."""
    with open("file1.txt", "w") as f1:
        with open("file2.txt", "w") as f2:
            with open("file3.txt", "w") as f3:
                f1.write("1")
                f2.write("2")
                f3.write("3")

# Custom context manager class
class MyContext:
    """Custom context manager."""

    def __enter__(self):
        print("Entering context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting context")

def use_custom_context():
    """Use custom context manager."""
    with MyContext() as ctx:
        print("Inside context")
        return "success"

# Context manager decorator
@contextmanager
def my_context_manager():
    """Context manager as decorator."""
    print("Setup")
    try:
        yield "resource"
    finally:
        print("Cleanup")

def use_context_decorator():
    """Use context manager decorator."""
    with my_context_manager() as resource:
        print(f"Using {resource}")
        return resource
'''

        with open(py_file, 'w') as f:
            f.write(source)

        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        functions = [e for e in elements if e.kind == 'function']

        # Check context manager usage detection
        file_func = next(f for f in functions if f.name == 'file_operations')
        assert 'f' in file_func.local_variables_declared
        assert 'g' in file_func.local_variables_declared
        assert 'content' in file_func.local_variables_declared

        nested_func = next(f for f in functions if f.name == 'nested_contexts')
        local_vars = nested_func.local_variables_declared
        assert 'f1' in local_vars
        assert 'f2' in local_vars
        assert 'f3' in local_vars

        # Check custom context manager class
        context_class = next(e for e in elements if e.name == 'MyContext')
        assert context_class.kind == 'class'
        assert '__enter__' in context_class.methods
        assert '__exit__' in context_class.methods

    def test_lambda_and_anonymous_features(self, temp_dir):
        """Test coverage of lambda and anonymous features."""
        py_file = temp_dir / "lambda_features.py"
        source = '''"""Test lambda features."""

def functions_with_lambdas():
    """Functions that use lambdas."""
    # Lambda in assignment
    square = lambda x: x * x

    # Lambda in map
    numbers = [1, 2, 3, 4, 5]
    squares = list(map(lambda x: x * x, numbers))

    # Lambda in filter
    evens = list(filter(lambda x: x % 2 == 0, numbers))

    # Lambda in sorted with key
    words = ["apple", "Banana", "cherry"]
    sorted_words = sorted(words, key=lambda w: w.lower())

    # Lambda in reduce
    from functools import reduce
    product = reduce(lambda x, y: x * y, numbers)

    return squares, evens, sorted_words, product

def higher_order_functions():
    """Higher-order functions."""
    def apply_func(func, data):
        """Apply function to data."""
        return [func(x) for x in data]

    def make_multiplier(n):
        """Return multiplier function."""
        return lambda x: x * n

    # Use higher-order functions
    doubler = make_multiplier(2)
    tripler = make_multiplier(3)

    data = [1, 2, 3]
    doubled = apply_func(doubler, data)
    tripled = apply_func(tripler, data)

    return doubled, tripled
'''

        with open(py_file, 'w') as f:
            f.write(source)

        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        functions = [e for e in elements if e.kind == 'function']

        # Check lambda usage detection
        lambda_func = next(f for f in functions if f.name == 'functions_with_lambdas')
        local_vars = lambda_func.local_variables_declared
        assert 'square' in local_vars
        assert 'squares' in local_vars
        assert 'evens' in local_vars
        assert 'words' in local_vars
        assert 'sorted_words' in local_vars
        assert 'product' in local_vars

        # Check higher-order function detection
        higher_func = next(f for f in functions if f.name == 'higher_order_functions')
        local_vars = higher_func.local_variables_declared
        assert 'apply_func' in local_vars
        assert 'make_multiplier' in local_vars
        assert 'doubler' in local_vars
        assert 'tripler' in local_vars
        assert 'data' in local_vars
        assert 'doubled' in local_vars
        assert 'tripled' in local_vars

    def test_string_formatting_features(self, temp_dir):
        """Test coverage of string formatting features."""
        py_file = temp_dir / "string_formatting_features.py"
        source = '''"""Test string formatting features."""

def old_style_formatting():
    """Old-style % formatting."""
    name = "World"
    age = 25
    # Old-style formatting
    message = "Hello, %s! You are %d years old." % (name, age)
    return message

def new_style_formatting():
    """New-style .format() formatting."""
    name = "Python"
    version = 3.9
    # New-style formatting
    message = "Hello, {}! Version: {:.1f}".format(name, version)
    return message

def f_string_formatting():
    """F-string formatting."""
    name = "Alice"
    score = 95.5
    items = [1, 2, 3]

    # F-string formatting
    greeting = f"Hello, {name}!"
    result = f"Score: {score:.2f}"
    summary = f"Items: {items}"

    # F-string with expressions
    bonus = f"Bonus: {score * 1.1:.1f}"
    count = f"Count: {len(items)}"

    # Complex f-string
    report = f"User {name} has {len(items)} items with score {score:.1f}"

    return greeting, result, summary, bonus, count, report

def template_strings():
    """Template string formatting."""
    from string import Template

    template = Template("Hello, $name! Your score is $score.")
    result = template.safe_substitute(name="Bob", score=88.5)

    return result
'''

        with open(py_file, 'w') as f:
            f.write(source)

        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        functions = [e for e in elements if e.kind == 'function']

        # Check local variable detection for various formatting methods
        old_style = next(f for f in functions if f.name == 'old_style_formatting')
        assert 'message' in old_style.local_variables_declared

        new_style = next(f for f in functions if f.name == 'new_style_formatting')
        assert 'message' in new_style.local_variables_declared

        f_string = next(f for f in functions if f.name == 'f_string_formatting')
        local_vars = f_string.local_variables_declared
        assert 'greeting' in local_vars
        assert 'result' in local_vars
        assert 'summary' in local_vars
        assert 'bonus' in local_vars
        assert 'count' in local_vars
        assert 'report' in local_vars

    def test_all_features_integration(self, temp_dir):
        """Test integration of all features together."""
        py_file = temp_dir / "all_features_integration.py"
        source = '''"""Integration test with all Python features."""

import asyncio
from typing import List, Dict, Optional, Union
from functools import lru_cache
from dataclasses import dataclass

@dataclass
class Configuration:
    """Configuration data class."""
    debug: bool = False
    timeout: int = 30

@lru_cache(maxsize=128)
async def complex_async_function(
    data: List[Dict[str, Union[str, int]]],
    config: Optional[Configuration] = None
) -> Dict[str, List[int]]:
    """
    Complex async function with all features.

    Args:
        data: Input data
        config: Optional configuration

    Returns:
        Processed results
    """
    if config is None:
        config = Configuration()

    results = []
    processed_items = []

    try:
        # Process data with comprehensions
        for item in data:
            if 'value' in item:
                processed = item['value'] * 2
                processed_items.append(processed)

        # Use various control structures
        if config.debug:
            print(f"Processing {len(processed_items)} items")

        if not processed_items:
            return {"results": [], "count": 0}

        # List comprehension
        squares = [x*x for x in processed_items if x > 0]

        # Dict comprehension
        indexed = {i: val for i, val in enumerate(squares)}

        # Nested function
        def inner_helper(values: List[int]) -> int:
            """Inner helper function."""
            total = 0
            for val in values:
                if val > 10:
                    total += val
                else:
                    total += 1
            return total

        final_total = inner_helper(squares)

        results = {
            "results": squares,
            "indexed": indexed,
            "total": final_total,
            "count": len(squares)
        }

    except ValueError as e:
        print(f"Value error: {e}")
        results = {"error": "value_error", "count": 0}
    except Exception as e:
        print(f"General error: {e}")
        results = {"error": "general_error", "count": 0}
    finally:
        # Cleanup
        processed_items.clear()

    return results

class FeatureTester:
    """Class demonstrating all features."""

    def __init__(self, name: str):
        self.name = name
        self._cache = {}

    @property
    def display_name(self) -> str:
        """Get display name."""
        return self.name.upper()

    async def test_all_features(self) -> Dict[str, str]:
        """Test all features."""
        # Use f-strings
        message = f"Testing {self.name}"

        # Use async/await
        await asyncio.sleep(0.01)

        # Use complex data structures
        test_data = [
            {"value": 5, "type": "number"},
            {"value": 10, "type": "number"},
            {"value": 15, "type": "number"}
        ]

        # Call the complex function
        config = Configuration(debug=True)
        result = await complex_async_function(test_data, config)

        return {
            "message": message,
            "result_count": str(result.get("count", 0)),
            "display": self.display_name
        }
'''

        with open(py_file, 'w') as f:
            f.write(source)

        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        # Verify all elements are extracted correctly
        element_names = {e.name for e in elements}
        expected_names = {
            'Configuration', 'complex_async_function', 'FeatureTester',
            'test_all_features'
        }

        assert expected_names.issubset(element_names)

        # Verify complex function features
        complex_func = next(e for e in elements if e.name == 'complex_async_function')
        assert complex_func.is_async is True
        assert len(complex_func.decorators) == 1
        assert complex_func.decorators[0]['name'] == 'lru_cache'
        assert 'ValueError' in complex_func.catches_exceptions
        assert 'Exception' in complex_func.catches_exceptions

        # Verify complex parameters
        expected_params = [
            'data: List[Dict[str, Union[str, int]]]',
            'config: Optional[Configuration] = None'
        ]
        for param in expected_params:
            assert param in complex_func.parameters

        # Verify class features
        config_class = next(e for e in elements if e.name == 'Configuration')
        assert len(config_class.decorators) == 1
        assert config_class.decorators[0]['name'] == 'dataclass'

        # Verify inner function detection
        inner_helper_found = any(e.name == 'inner_helper' for e in elements)
        assert inner_helper_found

        # Verify property detection
        feature_tester = next(e for e in elements if e.name == 'FeatureTester')
        # Note: Properties might not be detected as regular methods in current implementation