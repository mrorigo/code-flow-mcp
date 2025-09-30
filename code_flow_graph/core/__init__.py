"""
Unified interface for code analysis across multiple programming languages.

This module provides a simple, consistent API for extracting code elements
from Python and TypeScript source files, hiding the complexity of the
modular extractor architecture underneath.

Key Features:
- Factory functions for automatic language detection
- Unified API functions for file and directory extraction
- Clean imports for models and utilities
- Backward compatibility with existing code
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Import core data models
from .models import CodeElement, FunctionElement, ClassElement

# Import shared utilities
from .utils import (
    get_gitignore_patterns,
    match_file_against_pattern,
)

# Import extractor classes for backward compatibility and internal use
from .python_extractor import PythonASTExtractor, PythonASTVisitor
from .typescript_extractor import TypeScriptASTExtractor, TypeScriptASTVisitor

# Type alias for file paths
FilePath = Union[str, Path]

# Language detection and factory functions
def get_language_from_extension(file_path: FilePath) -> str:
    """
    Detect programming language from file extension.

    Args:
        file_path: Path to the file (string or Path object)

    Returns:
        Language name ('python' or 'typescript')

    Raises:
        ValueError: If file extension is not supported
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == '.py':
        return 'python'
    elif extension in ['.ts', '.tsx']:
        return 'typescript'
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

def create_extractor(file_path: FilePath) -> Any:
    """
    Create appropriate extractor instance based on file extension.

    Args:
        file_path: Path to the file to analyze

    Returns:
        Extractor instance (PythonASTExtractor or TypeScriptASTExtractor)

    Raises:
        ValueError: If file extension is not supported
    """
    language = get_language_from_extension(file_path)

    if language == 'python':
        return PythonASTExtractor()
    elif language == 'typescript':
        return TypeScriptASTExtractor()
    else:
        raise ValueError(f"Unsupported language: {language}")

# Unified API functions
def extract_from_file(file_path: FilePath, **kwargs) -> List[CodeElement]:
    """
    Extract code elements from a single file (works for both Python and TypeScript).

    Args:
        file_path: Path to the file to analyze
        **kwargs: Additional arguments passed to the specific extractor

    Returns:
        List of extracted code elements

    Example:
        >>> elements = extract_from_file('src/app.py')
        >>> elements = extract_from_file('src/component.tsx')
    """
    extractor = create_extractor(file_path)
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return extractor.extract_from_file(path)

def extract_from_directory(directory_path: FilePath, **kwargs) -> List[CodeElement]:
    """
    Extract code elements from all supported files in a directory.

    Args:
        directory_path: Directory to analyze
        **kwargs: Additional arguments passed to the specific extractor

    Returns:
        List of extracted code elements from all files

    Example:
        >>> elements = extract_from_directory('src/')
        >>> elements = extract_from_directory('./')
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Find all supported files
    python_files = list(directory.rglob('*.py'))
    typescript_files = list(directory.rglob('*.ts')) + list(directory.rglob('*.tsx'))
    all_files = python_files + typescript_files

    if not all_files:
        return []

    # Apply gitignore filtering
    ignored_patterns_with_dirs = get_gitignore_patterns(directory)

    filtered_files = [
        file_path
        for file_path in all_files
        if not any(
            match_file_against_pattern(file_path, pattern, gitignore_dir, directory)
            for pattern, gitignore_dir in ignored_patterns_with_dirs
        )
    ]

    if not filtered_files:
        return []

    print(f"Found {len(filtered_files)} files to analyze (after filtering .gitignore).")

    all_elements = []

    # Process Python files
    python_extractor = PythonASTExtractor()
    for py_file in python_files:
        if py_file in filtered_files:
            elements = python_extractor.extract_from_file(py_file)
            all_elements.extend(elements)

    # Process TypeScript files
    typescript_extractor = TypeScriptASTExtractor()
    for ts_file in typescript_files:
        if ts_file in filtered_files:
            elements = typescript_extractor.extract_from_file(ts_file)
            all_elements.extend(elements)

    return all_elements

# Convenience functions for direct language-specific extraction
def extract_python_file(file_path: FilePath) -> List[CodeElement]:
    """
    Extract code elements from a Python file.

    Args:
        file_path: Path to the Python file

    Returns:
        List of extracted code elements
    """
    extractor = PythonASTExtractor()
    return extractor.extract_from_file(Path(file_path))

def extract_python_directory(directory_path: FilePath) -> List[CodeElement]:
    """
    Extract code elements from all Python files in a directory.

    Args:
        directory_path: Directory containing Python files

    Returns:
        List of extracted code elements
    """
    extractor = PythonASTExtractor()
    return extractor.extract_from_directory(Path(directory_path))

def extract_typescript_file(file_path: FilePath) -> List[CodeElement]:
    """
    Extract code elements from a TypeScript file.

    Args:
        file_path: Path to the TypeScript file

    Returns:
        List of extracted code elements
    """
    extractor = TypeScriptASTExtractor()
    return extractor.extract_from_file(Path(file_path))

def extract_typescript_directory(directory_path: FilePath) -> List[CodeElement]:
    """
    Extract code elements from all TypeScript files in a directory.

    Args:
        directory_path: Directory containing TypeScript files

    Returns:
        List of extracted code elements
    """
    extractor = TypeScriptASTExtractor()
    return extractor.extract_from_directory(Path(directory_path))

# Export all important classes and functions for backward compatibility
__all__ = [
    # Core models
    'CodeElement',
    'FunctionElement',
    'ClassElement',

    # Factory and unified API functions
    'get_language_from_extension',
    'create_extractor',
    'extract_from_file',
    'extract_from_directory',

    # Language-specific convenience functions
    'extract_python_file',
    'extract_python_directory',
    'extract_typescript_file',
    'extract_typescript_directory',

    # Extractor classes (for advanced usage)
    'PythonASTExtractor',
    'PythonASTVisitor',
    'TypeScriptASTExtractor',
    'TypeScriptASTVisitor',

    # Type aliases
    'FilePath',
]

# Optional: Set up some commonly used aliases for convenience
# These can be imported directly for quick access
python = extract_python_file
typescript = extract_typescript_file
analyze = extract_from_file
analyze_directory = extract_from_directory