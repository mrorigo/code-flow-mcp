"""
Core data models for code elements extracted from AST parsing.

This module contains pure data structures for representing code elements
without any logic or behavior methods.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class CodeElement:
    """Base class for all code elements extracted from source code."""
    name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    full_source: str # This is the full file source where the element is found
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class FunctionElement(CodeElement):
    """Represents a function or method extracted from source code."""
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None # Now correctly parsed
    is_async: bool = False
    is_static: bool = False
    access_modifier: Optional[str] = None # e.g., 'public', 'private' (inferred by convention)
    docstring: Optional[str] = None
    is_method: bool = False
    class_name: Optional[str] = None
    # --- NEW ATTRIBUTES ---
    complexity: Optional[int] = None # Cyclomatic complexity
    nloc: Optional[int] = None      # Non-comment lines of code
    external_dependencies: List[str] = field(default_factory=list) # e.g., ['requests', 'numpy']
    decorators: List[Dict[str, Any]] = field(default_factory=list) # e.g., [{'name': 'app.route', 'args': ['/'], 'kwargs': {'methods': ['GET']}}]
    catches_exceptions: List[str] = field(default_factory=list) # e.g., ['ValueError', 'IOError']
    local_variables_declared: List[str] = field(default_factory=list) # Variables declared within the function
    hash_body: Optional[str] = None # Hash of the function's source body for change detection
    summary: Optional[str] = None # Natural language summary of the function

@dataclass
class ClassElement(CodeElement):
    """Represents a class, interface, or enum extracted from source code."""
    methods: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    extends: Optional[str] = None
    implements: List[str] = field(default_factory=list) # For interfaces in TS, or inferred in Py
    docstring: Optional[str] = None
    # --- NEW ATTRIBUTES ---
    decorators: List[Dict[str, Any]] = field(default_factory=list) # Class decorators
    hash_body: Optional[str] = None # Hash of the class's source body
    summary: Optional[str] = None # Natural language summary of the class

@dataclass
class StructuredDataElement(CodeElement):
    """Represents a chunk of structured data (JSON/YAML)."""
    json_path: str = ""
    value_type: str = ""
    key_name: str = ""
    content: str = "" # The string representation of the chunk