"""
Python AST processor for extracting code elements from Python source files.

This module contains the Python-specific AST visitor and extractor classes
for parsing Python codebases and extracting structured information about
functions, classes, and their metadata.
"""

import ast
import os
import re
import sys
import hashlib
import fnmatch
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

# Import models and utilities from the modular structure
from .models import CodeElement, FunctionElement, ClassElement
from .utils import (
    hash_source_snippet,
    calculate_nloc_python,
    calculate_complexity_python,
    extract_decorators_python,
    extract_file_imports_python,
    get_gitignore_patterns,
    match_file_against_pattern,
    _unparse_annotation_py
)

class PythonASTVisitor(ast.NodeVisitor):
    """
    AST visitor that extracts code elements with enhanced metadata.
    It expects full file source and relevant file-level imports to be set prior to visiting.
    """

    def __init__(self):
        self.elements: List[CodeElement] = []
        self.current_class: Optional[str] = None
        self.current_file: str = ""
        self.source_lines: List[str] = []
        self.file_level_imports: Dict[str, str] = {} # {local_name: original_module_name} e.g. {'requests': 'requests', 'Path': 'pathlib'}
        self.file_level_import_from_targets: Set[str] = set() # e.g. {'get', 'post'} for `from requests import get, post`

    def _unparse_annotation(self, node: Optional[ast.AST]) -> Optional[str]:
        """Recursively unparses an AST annotation node back to a string."""
        return _unparse_annotation_py(node)

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculates a simple approximation of Cyclomatic Complexity for a function body."""
        return calculate_complexity_python(node)

    def _calculate_nloc(self, start_line: int, end_line: int) -> int:
        """Calculates Non-Comment Lines of Code for a given line range."""
        return calculate_nloc_python(self.source_lines, start_line, end_line)

    def _extract_decorators(self, node: ast.AST) -> List[Dict[str, Any]]:
        """Extracts decorator names and arguments."""
        return extract_decorators_python(node)

    def _extract_exception_handlers(self, node: ast.AST) -> List[str]:
        """Extracts the types of exceptions handled within a function."""
        caught_exceptions = set()
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.ExceptHandler):
                if sub_node.type:
                    caught_exceptions.add(self._unparse_annotation(sub_node.type))
                else: # bare except
                    caught_exceptions.add('Exception') # Default catch-all
        return sorted(list(caught_exceptions))

    def _extract_local_variables(self, node: ast.AST) -> List[str]:
        """Extracts names of local variables declared via assignment within a function."""
        local_vars = set()
        for sub_node in ast.walk(node):
            # Only consider assignments *within* the function scope, not outer scopes or class attributes
            if isinstance(sub_node, ast.Assign):
                for target in sub_node.targets:
                    if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                        local_vars.add(target.id)
            # Correctly handle variables from loops, context managers, and comprehensions
            elif isinstance(sub_node, (ast.For, ast.AsyncFor)):
                 if isinstance(sub_node.target, ast.Name):
                     local_vars.add(sub_node.target.id)
            elif isinstance(sub_node, (ast.With, ast.AsyncWith)):
                 for item in sub_node.items:
                     if isinstance(item.optional_vars, ast.Name): # 'as var_name'
                         local_vars.add(item.optional_vars.id)
            elif isinstance(sub_node, ast.comprehension):
                 if isinstance(sub_node.target, ast.Name):
                     local_vars.add(sub_node.target.id)

        # Filter common keywords or self/cls
        return sorted([v for v in local_vars if v not in {'self', 'cls', '_', '__init__', '__post_init__'}])


    def _extract_external_dependencies(self, node: ast.AST) -> List[str]:
        """
        Identifies calls to external modules/functions based on file-level imports.
        This is a heuristic and may not be exhaustive for dynamic imports or complex aliasing.
        """
        external_deps = set()
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call):
                func = sub_node.func
                if isinstance(func, ast.Name):
                    # Check if it's a direct import (e.g., `requests.get`) or `from module import func`
                    if func.id in self.file_level_imports:
                        external_deps.add(self.file_level_imports[func.id])
                    elif func.id in self.file_level_import_from_targets:
                        external_deps.add(func.id)
                elif isinstance(func, ast.Attribute):
                    # e.g., `requests.get` where `requests` is an imported module
                    if isinstance(func.value, ast.Name) and func.value.id in self.file_level_imports:
                        external_deps.add(self.file_level_imports[func.value.id])
        return sorted(list(external_deps))

    def _hash_source_snippet(self, start_line: int, end_line: int) -> str:
        """Generates an MD5 hash of the stripped source code snippet."""
        return hash_source_snippet(self.source_lines, start_line, end_line)

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extracts function definitions with enhanced metadata."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)

        # Calculate function-specific source snippet for hash
        func_source_snippet_hash = self._hash_source_snippet(start_line, end_line)

        params = []
        for arg in node.args.args:
            param_name = arg.arg
            param_type = self._unparse_annotation(arg.annotation)
            if param_type:
                params.append(f"{param_name}: {param_type}")
            else:
                params.append(param_name)

        is_async = isinstance(node, ast.AsyncFunctionDef)
        docstring = ast.get_docstring(node)
        is_method = self.current_class is not None
        return_type_str = self._unparse_annotation(node.returns)

        # --- NEW METADATA EXTRACTION ---
        complexity = self._calculate_complexity(node)
        nloc = self._calculate_nloc(start_line, end_line)
        decorators = self._extract_decorators(node)
        catches_exceptions = self._extract_exception_handlers(node)
        local_variables_declared = self._extract_local_variables(node)
        external_dependencies = self._extract_external_dependencies(node)

        func_element = FunctionElement(
            name=node.name,
            kind='function',
            file_path=self.current_file,
            line_start=start_line,
            line_end=end_line,
            full_source='\n'.join(self.source_lines), # This is the full file content
            parameters=params,
            return_type=return_type_str,
            is_async=is_async,
            docstring=docstring,
            is_method=is_method,
            class_name=self.current_class if is_method else None,
            complexity=complexity, # NEW
            nloc=nloc,             # NEW
            external_dependencies=external_dependencies, # NEW
            decorators=decorators, # NEW
            catches_exceptions=catches_exceptions, # NEW
            local_variables_declared=local_variables_declared, # NEW
            hash_body=func_source_snippet_hash, # NEW
            metadata={}
        )
        self.elements.append(func_element)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)

        class_source_snippet_hash = self._hash_source_snippet(start_line, end_line)

        extends = self._unparse_annotation(node.bases[0]) if node.bases else None
        methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        attributes = [t.id for n in node.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)]
        docstring = ast.get_docstring(node)

        # --- NEW METADATA EXTRACTION FOR CLASSES ---
        decorators = self._extract_decorators(node)

        class_element = ClassElement(
            name=node.name,
            kind='class',
            file_path=self.current_file,
            line_start=start_line,
            line_end=end_line,
            full_source='\n'.join(self.source_lines),
            methods=methods,
            attributes=attributes,
            extends=extends,
            docstring=docstring,
            decorators=decorators, # NEW
            hash_body=class_source_snippet_hash, # NEW
            metadata={}
        )
        self.elements.append(class_element)

        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)


class PythonASTExtractor:
    """
    Main interface for extracting Python code elements from files and directories.
    """

    def __init__(self):
        self.visitor = PythonASTVisitor()
        self.project_root: Optional[Path] = None # Store the project root for consistent FQNs

    def _extract_file_imports(self, tree: ast.AST) -> Tuple[Dict[str, str], Set[str]]:
        """
        Extracts top-level imports from the file's AST.
        Returns a dictionary mapping local names to original module names
        and a set of names imported directly (e.g. `get` from `from requests import get`).
        """
        return extract_file_imports_python(tree)

    def extract_from_file(self, file_path: Path) -> List[CodeElement]:
        """
        Extract code elements from a single Python file.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            List of extracted code elements
        """
        if not file_path.exists() or file_path.suffix != '.py':
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))
            file_imports, import_from_targets = self._extract_file_imports(tree)

            self.visitor.source_lines = source.splitlines()
            self.visitor.current_file = str(file_path.resolve())
            self.visitor.elements = [] # Reset elements for each file
            self.visitor.file_level_imports = file_imports
            self.visitor.file_level_import_from_targets = import_from_targets

            self.visitor.visit(tree)
            return self.visitor.elements.copy()

        except Exception as e:
            # Re-raise to show the error if it's critical, or return empty if tolerable
            # For now, print a warning with the specific file if errors occur
            print(f"   Warning: Error processing {file_path}: {e}", file=sys.stderr)
            return []

    def extract_from_directory(self, directory: Path) -> List[CodeElement]:
        """
        Extract code elements from all Python files in a directory.

        Args:
            directory: Directory to analyze

        Returns:
            List of extracted code elements
        """
        self.project_root = directory.resolve() # Set project root for consistent FQNs later, resolved to absolute
        elements = []
        python_files = list(directory.rglob('*.py'))

        ignored_patterns_with_dirs = get_gitignore_patterns(directory)

        # Filter files based on gitignore patterns
        filtered_files = [
            file_path
            for file_path in python_files
            if not any(
                match_file_against_pattern(file_path, pattern, gitignore_dir, directory)
                for pattern, gitignore_dir in ignored_patterns_with_dirs
            )
        ]

        print(f"Found {len(filtered_files)} Python files to analyze (after filtering .gitignore).")

        # Use tqdm for progress indication
        for file_path in filtered_files:
            elements.extend(self.extract_from_file(file_path))

        return elements