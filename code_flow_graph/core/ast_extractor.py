"""
Extracts and represents AST nodes for Python codebases with enhanced metadata.
Includes logic for parsing function parameter and return type annotations,
calculating complexity, detecting decorators, and more.
"""

import ast
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import os
import fnmatch
import hashlib
import sys

# --- Data Structures with Enhancements (unchanged from previous version) ---

@dataclass
class CodeElement:
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


@dataclass
class ClassElement(CodeElement):
    methods: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    extends: Optional[str] = None
    implements: List[str] = field(default_factory=list) # For interfaces in TS, or inferred in Py
    docstring: Optional[str] = None
    # --- NEW ATTRIBUTES ---
    decorators: List[Dict[str, Any]] = field(default_factory=list) # Class decorators
    hash_body: Optional[str] = None # Hash of the class's source body

# --- Core Logic with Fixes and Enhancements ---

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

    def _calculate_complexity(self, node: ast.AST) -> int:
        """
        Calculates a simple approximation of Cyclomatic Complexity for a function body.
        Counts decision points (if, for, while, except, etc.).
        """
        complexity = 1 # Start with 1 for the function itself
        for sub_node in ast.walk(node):
            if isinstance(sub_node, (ast.If, ast.For, ast.While, ast.AsyncFor,
                                     ast.Try, ast.ExceptHandler, ast.With, ast.AsyncWith,
                                     ast.comprehension)): # Add list/dict/set comprehensions
                complexity += 1
            # Add 'and' and 'or' operators in boolean expressions
            elif isinstance(sub_node, (ast.BoolOp)) and isinstance(sub_node.op, (ast.And, ast.Or)):
                complexity += len(sub_node.values) - 1 # Each 'and' or 'or' adds one decision point
        return complexity

    def _calculate_nloc(self, start_line: int, end_line: int) -> int:
        """Calculates Non-Comment Lines of Code for a given line range."""
        nloc = 0
        if not self.source_lines:
            return 0

        # Adjust for 0-based indexing if necessary
        start_idx = max(0, start_line - 1)
        end_idx = min(len(self.source_lines), end_line)

        for i in range(start_idx, end_idx):
            line = self.source_lines[i].strip()
            if line and not line.startswith('#'):
                nloc += 1
        return nloc

    def _extract_decorators(self, node: ast.AST) -> List[Dict[str, Any]]:
        """Extracts decorator names and arguments."""
        decorators_list = []
        if hasattr(node, 'decorator_list'):
            for decorator in node.decorator_list:
                dec_info = {}
                if isinstance(decorator, ast.Name):
                    dec_info['name'] = decorator.id
                elif isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name):
                        dec_info['name'] = decorator.func.id
                    elif isinstance(decorator.func, ast.Attribute):
                        # e.g., @app.route
                        if hasattr(decorator.func.value, 'id'):
                            dec_info['name'] = f"{decorator.func.value.id}.{decorator.func.attr}"
                        else:
                            dec_info['name'] = decorator.func.attr # Fallback for more complex expressions
                    dec_info['args'] = [self._unparse_annotation(arg) for arg in decorator.args]
                    dec_info['kwargs'] = {kw.arg: self._unparse_annotation(kw.value) for kw in decorator.keywords if kw.arg}

                if 'name' in dec_info: # Only add if a name was successfully extracted
                    decorators_list.append(dec_info)
        return decorators_list

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
        if not self.source_lines or start_line > end_line:
            return ""

        # Adjust for 0-based indexing
        start_idx = max(0, start_line - 1)
        end_idx = min(len(self.source_lines), end_line)

        snippet_lines = self.source_lines[start_idx:end_idx]
        stripped_source = "\n".join(line.strip() for line in snippet_lines if line.strip())
        return hashlib.md5(stripped_source.encode('utf-8')).hexdigest()

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
    def __init__(self):
        self.visitor = PythonASTVisitor()
        self.project_root: Optional[Path] = None # Store the project root for consistent FQNs

    def _extract_file_imports(self, tree: ast.AST) -> Tuple[Dict[str, str], Set[str]]:
        """
        Extracts top-level imports from the file's AST.
        Returns a dictionary mapping local names to original module names
        and a set of names imported directly (e.g. `get` from `from requests import get`).
        """
        file_imports = {}
        import_from_targets = set()
        for node in ast.walk(tree):
            # Only consider top-level imports (not nested within functions/classes)
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        local_name = alias.asname if alias.asname else alias.name
                        file_imports[local_name] = alias.name # e.g. {'requests': 'requests'} or {'np': 'numpy'}
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            local_name = alias.asname if alias.asname else alias.name
                            file_imports[local_name] = f"{node.module}.{alias.name}" if node.module else alias.name
                            import_from_targets.add(local_name) # Track direct imports like 'get'
            else: # Stop traversing deeper once we hit a function or class definition
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Don't recurse into function/class bodies for file-level imports
                    continue
        return file_imports, import_from_targets

    def extract_from_file(self, file_path: Path) -> List[CodeElement]:
        if not file_path.exists() or file_path.suffix != '.py': return []
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
            # print(f"❌ Error processing {file_path}: {e}") # Suppress for cleaner tqdm output
            # Re-raise to show the error if it's critical, or return empty if tolerable
            # For now, print a warning with the specific file if errors occur
            print(f"   Warning: Error processing {file_path}: {e}", file=sys.stderr)
            return []

    def extract_from_directory(self, directory: Path) -> List[CodeElement]:
        self.project_root = directory.resolve() # Set project root for consistent FQNs later, resolved to absolute
        elements = []
        python_files = list(directory.rglob('*.py'))

        ignored_patterns_with_dirs = self._get_gitignore_patterns(directory)

        # Pass the main `directory` to the matching function
        filtered_files = [
            file_path
            for file_path in python_files
            if not any(
                self._match_file_against_pattern(file_path, pattern, gitignore_dir, directory)
                for pattern, gitignore_dir in ignored_patterns_with_dirs
            )
        ]

        print(f"Found {len(filtered_files)} Python files to analyze (after filtering .gitignore).")

        # Use tqdm for progress indication
        for file_path in filtered_files:
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

    def _match_file_against_pattern(self, file_path: Path, pattern: str, gitignore_dir: Path, root_directory: Path) -> bool:
        """
        Match a file path against a gitignore pattern, considering the pattern's origin directory
        and the overall root directory.
        """
        try:
            # Path relative to where the .gitignore file lives
            relative_to_gitignore_dir = file_path.relative_to(gitignore_dir)
            rel_str_gitignore = str(relative_to_gitignore_dir)
        except ValueError:
            # File is not within the directory containing the .gitignore, so it can't be ignored by it.
            return False

        # Path relative to the main analysis root directory (for patterns like 'docs/conf.py')
        try:
            relative_to_root_dir = file_path.relative_to(root_directory)
            rel_str_root = str(relative_to_root_dir)
        except ValueError:
            # This should generally not happen if file_path is within root_directory
            rel_str_root = rel_str_gitignore # Fallback, though less precise

        # Check if the pattern is a directory (ends with /)
        if pattern.endswith('/'):
            # Match against path relative to gitignore_dir
            return rel_str_gitignore.startswith(pattern[:-1] + os.sep) or rel_str_gitignore == pattern[:-1]
        else:
            # Match directly against path relative to gitignore_dir
            if fnmatch.fnmatch(rel_str_gitignore, pattern):
                return True

            # Additional check: If pattern is a simple name (no /), it should match
            # if any part of the path (relative to gitignore_dir) matches it.
            # E.g., pattern 'build' should match 'path/to/build/file.py' or 'path/to/build.py'
            if '/' not in pattern and relative_to_gitignore_dir.parts:
                for part in relative_to_gitignore_dir.parts:
                    if fnmatch.fnmatch(part, pattern):
                        return True

            # Finally, check against path relative to the *root analysis directory* for patterns
            # that specify a full path like 'docs/conf.py' regardless of where the .gitignore is.
            if fnmatch.fnmatch(rel_str_root, pattern):
                return True
        return False


class TypeScriptASTExtractor:
    def extract_from_directory(self, directory: Path) -> List[CodeElement]:
        print("⏳ TypeScript parsing not implemented.")
        return []
