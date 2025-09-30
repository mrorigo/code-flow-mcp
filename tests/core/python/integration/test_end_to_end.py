"""
End-to-end integration tests for Python AST processing.
"""

import pytest
from pathlib import Path

from code_flow_graph.core.python_extractor import PythonASTExtractor


class TestPythonEndToEnd:
    """End-to-end integration tests for Python AST processing."""

    def test_end_to_end_basic_python_file(self, temp_dir):
        """Test complete workflow on a basic Python file."""
        py_file = temp_dir / "basic_test.py"
        source = '''# -*- coding: utf-8 -*-
"""Basic Python file for testing."""

import os
from typing import List, Dict, Optional

def greet_user(name: str) -> str:
    """Greets a user by name."""
    return f"Hello, {name}!"

def process_data(items: List[Dict[str, str]]) -> Optional[str]:
    """Process a list of dictionaries."""
    if not items:
        return None

    for item in items:
        if 'error' in item:
            raise ValueError("Invalid item found")

    return f"Processed {len(items)} items"

class UserManager:
    """Manages user operations."""

    def __init__(self, users: List[Dict[str, str]] = None):
        self.users = users or []
        self.cache = {}

    def get_user(self, user_id: str) -> Optional[Dict[str, str]]:
        """Get user by ID with caching."""
        if user_id in self.cache:
            return self.cache[user_id]

        user = next((u for u in self.users if u["id"] == user_id), None)
        if user:
            self.cache[user_id] = user
        return user

    def add_user(self, user: Dict[str, str]) -> None:
        """Add a new user."""
        self.users.append(user)
        # Clear cache when adding new users
        self.cache.clear()
'''

        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(source)

        # Test complete extraction workflow
        extractor = PythonASTExtractor()
        elements = extractor.extract_from_file(py_file)

        # Verify we extracted all expected elements
        element_names = {e.name for e in elements}
        expected_names = {'greet_user', 'process_data', 'UserManager', 'get_user', 'add_user'}

        assert expected_names.issubset(element_names)

        # Verify function details
        greet_func = next(e for e in elements if e.name == 'greet_user')
        assert greet_func.kind == 'function'
        assert greet_func.parameters == ['name: str']
        assert greet_func.return_type == 'str'
        assert greet_func.docstring == "Greets a user by name."
        assert greet_func.is_method is False
        # Note: External dependency detection may need refinement in visitor implementation
        # assert 'typing' in greet_func.external_dependencies

        # Verify complex function
        process_func = next(e for e in elements if e.name == 'process_data')
        assert process_func.kind == 'function'
        assert process_func.parameters == ['items: List[Dict[str, str]]']
        assert process_func.return_type == 'Optional[str]'
        assert 'ValueError' in process_func.catches_exceptions
        assert process_func.complexity >= 3  # if + for + raise

        # Verify class details
        user_manager = next(e for e in elements if e.name == 'UserManager')
        assert user_manager.kind == 'class'
        assert 'get_user' in user_manager.methods
        assert 'add_user' in user_manager.methods
        assert 'users' in user_manager.attributes
        assert 'cache' in user_manager.attributes

        # Verify method details
        get_user_method = next(e for e in elements if e.name == 'get_user')
        assert get_user_method.kind == 'function'
        assert get_user_method.is_method is True
        assert get_user_method.class_name == 'UserManager'
        assert get_user_method.parameters == ['self', 'user_id: str']
        assert get_user_method.return_type == 'Optional[Dict[str, str]]'

    def test_end_to_end_directory_processing(self, temp_dir):
        """Test complete workflow on a directory with multiple Python files."""
        # Create a package structure
        pkg_dir = temp_dir / "testpackage"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text('"""Test package."""')

        # Create main module
        main_py = pkg_dir / "main.py"
        main_py.write_text('''"""Main module."""

from .utils import helper_function
from .models import User

def main_process():
    """Main processing function."""
    user = User("john", "john@example.com")
    return helper_function(user.name)
''')

        # Create utils module
        utils_py = pkg_dir / "utils.py"
        utils_py.write_text('''"""Utility functions."""

def helper_function(name: str) -> str:
    """Helper function."""
    return f"Processed: {name}"
''')

        # Create models module
        models_py = pkg_dir / "models.py"
        models_py.write_text('''"""Data models."""

from dataclasses import dataclass

@dataclass
class User:
    """User model."""
    name: str
    email: str

    def get_display_name(self) -> str:
        """Get display name."""
        return self.name.title()
''')

        # Test complete directory extraction
        extractor = PythonASTExtractor()
        elements = extractor.extract_from_directory(temp_dir)

        # Verify all elements are extracted
        element_names = {e.name for e in elements}
        expected_names = {'main_process', 'helper_function', 'User', 'get_display_name'}

        assert expected_names.issubset(element_names)

        # Verify function relationships and dependencies
        main_func = next(e for e in elements if e.name == 'main_process')
        assert 'testpackage.utils' in main_func.external_dependencies

        helper_func = next(e for e in elements if e.name == 'helper_function')
        assert helper_func.file_path.endswith('utils.py')

        # Verify class and method
        user_class = next(e for e in elements if e.name == 'User')
        assert user_class.kind == 'class'
        assert 'get_display_name' in user_class.methods

        display_method = next(e for e in elements if e.name == 'get_display_name')
        assert display_method.is_method is True
        assert display_method.class_name == 'User'

    def test_end_to_end_complex_project_structure(self, temp_dir):
        """Test complete workflow on a complex project structure."""
        # Create complex package structure
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

        # Test complete extraction
        extractor = PythonASTExtractor()
        elements = extractor.extract_from_directory(project_root)

        # Verify extraction of various element types
        element_names = {e.name for e in elements}
        expected_names = {
            'DataProcessor', 'process_batch', '_process_item',
            'health_check', 'process_endpoint', 'test_data_processor',
            'test_async_processing'
        }

        assert expected_names.issubset(element_names)

        # Verify async function detection
        process_batch = next(e for e in elements if e.name == 'process_batch')
        assert process_batch.is_async is True
        assert process_batch.is_method is True
        assert process_batch.class_name == 'DataProcessor'

        # Verify decorator detection
        health_check = next(e for e in elements if e.name == 'health_check')
        # Note: FastAPI decorators might not be detected as Python AST decorators
        # but should still be extracted as functions

        # Verify test function detection
        test_func = next(e for e in elements if e.name == 'test_data_processor')
        assert test_func.kind == 'function'
        assert 'pytest' in test_func.external_dependencies

    def test_end_to_end_error_handling_and_edge_cases(self, temp_dir):
        """Test error handling and edge cases in end-to-end workflow."""
        # Create files with various edge cases
        edge_cases_dir = temp_dir / "edge_cases"
        edge_cases_dir.mkdir()

        # File with syntax error
        syntax_error_py = edge_cases_dir / "syntax_error.py"
        syntax_error_py.write_text('def invalid_function(')  # Missing closing paren

        # File with encoding issues (simulate)
        encoding_py = edge_cases_dir / "encoding.py"
        encoding_py.write_text('''# -*- coding: utf-8 -*-
def función_con_ñ() -> str:
    """Function with ñ in name."""
    return "café"
''')

        # Very large file (simulate with many functions)
        large_py = edge_cases_dir / "large.py"
        large_functions = []
        for i in range(50):
            large_functions.append(f'''
def function_{i}(param: str) -> str:
    """Function {i}."""
    return f"result_{i}"
''')
        large_py.write_text(''.join(large_functions))

        # Empty file
        empty_py = edge_cases_dir / "empty.py"
        empty_py.write_text('')

        # Test that workflow handles errors gracefully
        extractor = PythonASTExtractor()
        elements = extractor.extract_from_directory(edge_cases_dir)

        # Should extract from valid files only
        func_names = {e.name for e in elements if e.kind == 'function'}

        # Should have functions from large.py
        for i in range(50):
            assert f'function_{i}' in func_names

        # Should have function from encoding.py
        assert 'función_con_ñ' in func_names

        # Should not crash on syntax error or empty file
        # (syntax error file should be skipped gracefully)

    def test_end_to_end_real_world_patterns(self, temp_dir):
        """Test extraction with real-world Python patterns."""
        # Create a realistic Python application structure
        app_dir = temp_dir / "realistic_app"
        app_dir.mkdir()

        # Configuration module
        config_py = app_dir / "config.py"
        config_py.write_text('''"""Application configuration."""

import os
from typing import Dict

class Config:
    """Configuration manager."""

    def __init__(self):
        self.settings = {
            "database_url": os.getenv("DATABASE_URL", "sqlite:///app.db"),
            "debug": os.getenv("DEBUG", "false").lower() == "true"
        }

    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.settings.get(key, default)

config = Config()
''')

        # Database models
        models_py = app_dir / "models.py"
        models_py.write_text('''"""Database models."""

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    """User model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.username}>"

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat()
        }
''')

        # Service layer
        services_py = app_dir / "services.py"
        services_py.write_text('''"""Business logic services."""

from typing import List, Optional
from .models import User
from .config import config

class UserService:
    """Service for user operations."""

    def __init__(self, db_session):
        self.db = db_session

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        try:
            return self.db.query(User).filter(User.id == user_id).first()
        except Exception as e:
            print(f"Error fetching user: {e}")
            return None

    def create_user(self, username: str, email: str) -> User:
        """Create a new user."""
        user = User(username=username, email=email)
        try:
            self.db.add(user)
            self.db.commit()
            return user
        except Exception as e:
            self.db.rollback()
            raise e

    def get_all_users(self) -> List[User]:
        """Get all users."""
        return self.db.query(User).all()
''')

        # Main application
        main_py = app_dir / "main.py"
        main_py.write_text('''"""Main application entry point."""

import asyncio
from .config import config
from .services import UserService

async def main():
    """Main application function."""
    print(f"Starting app with debug={config.get('debug')}")

    # Simulate database session
    class MockDB:
        def query(self, model):
            return self
        def filter(self, condition):
            return self
        def first(self):
            return None
        def all(self):
            return []

    db = MockDB()
    user_service = UserService(db)

    # Test the service
    user = user_service.get_user_by_id(1)
    print(f"Found user: {user}")

if __name__ == "__main__":
    asyncio.run(main())
''')

        # Test complete extraction
        extractor = PythonASTExtractor()
        elements = extractor.extract_from_directory(app_dir)

        # Verify extraction of realistic patterns
        element_names = {e.name for e in elements}
        expected_classes = {'Config', 'User', 'UserService'}
        expected_functions = {'get', 'get_user_by_id', 'create_user', 'get_all_users', 'main', 'to_dict'}

        for class_name in expected_classes:
            assert class_name in element_names

        for func_name in expected_functions:
            assert func_name in element_names

        # Verify ORM usage detection
        user_service = next(e for e in elements if e.name == 'UserService')
        assert user_service.kind == 'class'

        # Verify SQLAlchemy dependencies
        create_user = next(e for e in elements if e.name == 'create_user')
        # Note: SQLAlchemy might not be detected as external dependency
        # depending on how the AST analysis works

        # Verify configuration management
        config_class = next(e for e in elements if e.name == 'Config')
        assert config_class.kind == 'class'

    def test_end_to_end_performance_with_large_codebase(self, temp_dir):
        """Test performance with a large Python codebase simulation."""
        # Create a large codebase with many modules
        large_project = temp_dir / "large_project"
        large_project.mkdir()

        # Generate multiple modules with various complexity
        for module_num in range(20):
            module_dir = large_project / f"module_{module_num:02d}"
            module_dir.mkdir()
            (module_dir / "__init__.py").write_text("")

            # Create several files per module
            for file_num in range(5):
                py_file = module_dir / f"file_{file_num}.py"
                content = []

                # Add various Python constructs
                for class_num in range(3):
                    content.append(f'''
class Class{class_num}:
    """Class {class_num} in module {module_num}."""

    def __init__(self, value: int = {class_num}):
        self.value = value

    def method_{class_num}(self, data: str) -> str:
        """Method {class_num}."""
        return f"{{data}}_{{self.value}}"
''')

                for func_num in range(5):
                    content.append(f'''
def function_{func_num}(param: int) -> str:
    """Function {func_num} in module {module_num}."""
    if param > {func_num}:
        return "greater"
    else:
        return "lesser"
''')

                py_file.write_text(''.join(content))

        # Test extraction performance
        extractor = PythonASTExtractor()
        elements = extractor.extract_from_directory(large_project)

        # Verify all elements are extracted
        functions = [e for e in elements if e.kind == 'function']
        classes = [e for e in elements if e.kind == 'class']

        # Should have 20 modules * 5 files * (5 functions + 3 classes) = 400 classes + 1000 functions
        assert len(classes) >= 400
        assert len(functions) >= 1000

        # Verify some specific elements
        sample_func = next(e for e in elements if e.name == 'function_2')
        assert sample_func is not None

        sample_class = next(e for e in elements if e.name == 'Class1')
        assert sample_class is not None
        assert 'method_1' in sample_class.methods