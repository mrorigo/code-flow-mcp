"""
Extracts and represents AST nodes for Python codebases.
Includes logic for parsing function parameter and return type annotations.
"""

import ast
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import os
import fnmatch

# --- Data Structures (Unchanged, but provided for completeness) ---

@dataclass
class CodeElement:
    name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    full_source: str
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class FunctionElement(CodeElement):
    parameters: List[str] # Now formatted as "name: type"
    return_type: Optional[str] # Now correctly parsed
    is_async: bool
    is_static: bool = False
    access_modifier: Optional[str] = None
    docstring: Optional[str] = None
    is_method: bool = False
    class_name: Optional[str] = None

@dataclass
class ClassElement(CodeElement):
    methods: List[str]
    attributes: List[str]
    extends: Optional[str] = None
    implements: List[str] = None
    docstring: Optional[str] = None

# --- Core Logic with Fixes ---

class PythonASTVisitor(ast.NodeVisitor):
    """AST visitor that extracts code elements, including detailed type annotations."""

    def __init__(self):
        self.elements: List[CodeElement] = []
        self.current_class: Optional[str] = None
        self.current_file: str = ""
        self.source_lines: List[str] = []

    def _unparse_annotation(self, node: Optional[ast.AST]) -> Optional[str]:
        """Recursively unparses an AST annotation node back to a string."""
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant): # Handles string forward references like "MyClass"
            return str(node.value)
        if isinstance(node, ast.Attribute):
            # Handles namespaced types like os.PathLike
            value = self._unparse_annotation(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        if isinstance(node, ast.Subscript):
            # Handles generics like List[str] or Dict[int, str]
            value = self._unparse_annotation(node.value)
            slice_val = self._unparse_annotation(node.slice)
            return f"{value}[{slice_val}]"
        if isinstance(node, ast.Tuple): # For slices like Dict[str, int]
            return ", ".join([self._unparse_annotation(e) for e in node.elts])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr): # For Python 3.10+ unions like str | None
            left = self._unparse_annotation(node.left)
            right = self._unparse_annotation(node.right)
            return f"{left} | {right}"

        # Fallback for complex or unhandled types
        return "any"

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extracts function definitions with full type annotation parsing."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)

        # Parse parameters with their types
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

        # *** THE FIX: Parse the return type annotation ***
        return_type_str = self._unparse_annotation(node.returns)

        func_element = FunctionElement(
            name=node.name,
            kind='function',
            file_path=self.current_file,
            line_start=start_line,
            line_end=end_line,
            full_source='\n'.join(self.source_lines),
            parameters=params,
            return_type=return_type_str,
            is_async=is_async,
            docstring=docstring,
            is_method=is_method,
            class_name=self.current_class if is_method else None,
            metadata={}
        )
        self.elements.append(func_element)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        extends = self._unparse_annotation(node.bases[0]) if node.bases else None
        methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        attributes = [t.id for n in node.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)]
        docstring = ast.get_docstring(node)

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
    def __init__(self):
        self.visitor = PythonASTVisitor()

    def extract_from_file(self, file_path: Path) -> List[CodeElement]:
        if not file_path.exists() or file_path.suffix != '.py': return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            self.visitor.source_lines = source.splitlines()
            self.visitor.current_file = str(file_path)
            self.visitor.elements = []
            tree = ast.parse(source, filename=str(file_path))
            self.visitor.visit(tree)
            return self.visitor.elements.copy()
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return []

    def extract_from_directory(self, directory: Path) -> List[CodeElement]:
        elements = []
        python_files = list(directory.rglob('*.py'))

        # Filter out files ignored by .gitignore
        ignored_patterns_with_dirs = self._get_gitignore_patterns(directory)

        filtered_files = [
            file_path
            for file_path in python_files
            if not any(
                self._match_file_against_pattern(file_path, pattern, gitignore_dir)
                for pattern, gitignore_dir in ignored_patterns_with_dirs
            )
        ]

        print(f"Found {len(filtered_files)} Python files to analyze (after filtering .gitignore).")

        for file_path in filtered_files:
            print(f"Processing: {file_path}")
            elements.extend(self.extract_from_file(file_path))
        return elements

    def _get_gitignore_patterns(self, directory: Path) -> List[tuple[str, Path]]:
        """
        Collect .gitignore patterns from the directory and its parents, also returning the directory
        where the .gitignore file was found.
        """
        patterns_with_dirs: List[tuple[str, Path]] = []
        current_dir = directory
        while current_dir != current_dir.parent:  # Stop at the root directory
            gitignore_path = current_dir / ".gitignore"
            if gitignore_path.exists() and gitignore_path.is_file():
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):  # Ignore comments and empty lines
                            patterns_with_dirs.append((line, current_dir))
            current_dir = current_dir.parent
        return patterns_with_dirs

    def _match_file_against_pattern(self, file_path: Path, pattern: str, gitignore_dir: Path) -> bool:
        """
        Match a file path against a gitignore pattern, considering the pattern's origin directory.
        """
        # Create a path relative to the directory where the pattern was defined
        try:
            relative_path = file_path.relative_to(gitignore_dir)
        except ValueError:
            return False  # File is not within the directory containing the .gitignore

        rel_str = str(relative_path)

        # Check if the pattern is a directory (ends with /)
        if pattern.endswith('/'):
            # Check if the relative path starts with the pattern directory
            return rel_str.startswith(pattern[:-1] + os.sep)
        else:
            # Check if the relative path matches the pattern exactly
            # Or if it starts with pattern followed by a path separator (directory)
            return fnmatch.fnmatch(rel_str, pattern) or rel_str.startswith(pattern + os.sep)

class TypeScriptASTExtractor:
    def extract_from_directory(self, directory: Path) -> List[CodeElement]:
        print("⏳ TypeScript parsing not implemented.")
        return []
