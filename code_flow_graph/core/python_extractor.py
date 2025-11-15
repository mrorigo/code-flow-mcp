"""
Python AST processor for extracting code elements from Python source files.

This module contains the Python-specific AST visitor and extractor classes
for parsing Python codebases and extracting structured information about
functions, classes, and their metadata.
"""

import ast
import logging
import re
import sys
import time
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
        self.full_source: str = ""
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
                    # Handle both single exception types and tuples of exception types
                    exception_types = self._extract_exception_types(sub_node.type)
                    caught_exceptions.update(exception_types)
                else: # bare except
                    caught_exceptions.add('Exception') # Default catch-all
        return sorted(list(caught_exceptions))

    def _extract_exception_types(self, node: ast.AST) -> List[str]:
        """Recursively extract exception type names from AST node."""
        if isinstance(node, ast.Name):
            return [node.id]
        elif isinstance(node, ast.Tuple):
            # Handle tuple of exception types like (ValueError, TypeError)
            result = []
            for elt in node.elts:
                result.extend(self._extract_exception_types(elt))
            return result
        elif isinstance(node, ast.Attribute):
            # Handle module.exception like ValueError
            if isinstance(node.value, ast.Name):
                return [f"{node.value.id}.{node.attr}"]
            return [node.attr]
        else:
            # Fallback for complex expressions
            return [self._unparse_annotation(node)]

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
            # Also detect function definitions as local variables
            elif isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local_vars.add(sub_node.name)

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
                        # Extract just the module part (e.g., '.utils.helper_function' -> '.utils')
                        full_module = self.file_level_imports[func.id]
                        if full_module.startswith('.'):
                            # For relative imports, get the module part
                            module_part = '.'.join(full_module.split('.')[:-1])
                            if module_part:  # Avoid empty string
                                external_deps.add(module_part)
                        else:
                            # For absolute imports, extract base module name
                            base_module = full_module.split('.')[0]
                            external_deps.add(base_module)
                    elif func.id in self.file_level_import_from_targets:
                        external_deps.add(func.id)
                elif isinstance(func, ast.Attribute):
                    # e.g., `requests.get` where `requests` is an imported module
                    if isinstance(func.value, ast.Name) and func.value.id in self.file_level_imports:
                        full_module = self.file_level_imports[func.value.id]
                        if full_module.startswith('.'):
                            module_part = '.'.join(full_module.split('.')[:-1])
                            if module_part:
                                external_deps.add(module_part)
                        else:
                            base_module = full_module.split('.')[0]
                            external_deps.add(base_module)
            elif isinstance(sub_node, ast.Attribute):
                # Also check for direct attribute access like `os.path`
                if isinstance(sub_node.value, ast.Name) and sub_node.value.id in self.file_level_imports:
                    full_module = self.file_level_imports[sub_node.value.id]
                    if full_module.startswith('.'):
                        module_part = '.'.join(full_module.split('.')[:-1])
                        if module_part:
                            external_deps.add(module_part)
                    else:
                        base_module = full_module.split('.')[0]
                        external_deps.add(base_module)
            elif isinstance(sub_node, ast.Name):
                # Check if it's a reference to an imported name in type annotations or other contexts
                if sub_node.id in self.file_level_imports:
                    full_module = self.file_level_imports[sub_node.id]
                    if full_module.startswith('.'):
                        module_part = '.'.join(full_module.split('.')[:-1])
                        if module_part:
                            external_deps.add(module_part)
                    else:
                        base_module = full_module.split('.')[0]
                        external_deps.add(base_module)
                elif sub_node.id in self.file_level_import_from_targets:
                    # For 'from module import name' patterns, we need to find the original module
                    # This is a heuristic - check if any import ends with the target name
                    for import_name, full_import in self.file_level_imports.items():
                        if import_name == sub_node.id:
                            if full_import.startswith('.'):
                                module_part = '.'.join(full_import.split('.')[:-1])
                                if module_part:
                                    external_deps.add(module_part)
                            else:
                                base_module = full_import.split('.')[0]
                                external_deps.add(base_module)
                            break
        return sorted(list(external_deps))

    def _hash_source_snippet(self, start_line: int, end_line: int) -> str:
        """Generates an MD5 hash of the stripped source code snippet."""
        return hash_source_snippet(self.source_lines, start_line, end_line)

    def _compile_regex_patterns(self) -> Dict[str, Any]:
        """Pre-compile regex patterns used in utility functions for maximum performance."""
        import time
        start_time = time.time()

        patterns = {}

        # Import patterns for TypeScript files (used in utils.py)
        import_patterns = [
            r'import\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',  # import name from 'module'
            r'import\s+\*\s+as\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',  # import * as name from 'module'
            r'import\s*{\s*([^}]+)\s*}\s+from\s+[\'"]([^\'"]+)[\'"]',  # import { ... } from 'module'
        ]

        patterns['imports'] = [re.compile(pattern) for pattern in import_patterns]

        # Framework detection patterns
        framework_patterns = [
            r'@Entity\s*\(', r'@Column\s*\(', r'@PrimaryGeneratedColumn',
            r'@OneToMany', r'@ManyToOne', r'useState\s*\(', r'useEffect\s*\(',
            r'React\.FC', r'express\s*\(\s*\)', r'createApp\s*\(',
            r'@Controller\s*\(', r'@Component\s*\(', r'fastify\s*\(\s*\)'
        ]

        patterns['frameworks'] = [re.compile(pattern, re.IGNORECASE) for pattern in framework_patterns]

        # Generic type patterns
        patterns['generics'] = re.compile(r'<([^>]+)>')

        # Decorator patterns
        patterns['decorators'] = re.compile(r'@(\w+)')

        compilation_time = time.time() - start_time
        pattern_count = sum(len(p) if isinstance(p, list) else 1 for p in patterns.values())
        logging.info(f"   Info: Pre-compiled {pattern_count} regex patterns for Python analysis in {compilation_time:.4f}s")

        return patterns

    def visit(self, node: ast.AST) -> None:
        super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extracts function definitions with enhanced metadata."""
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)

        # Calculate function-specific source snippet for hash
        func_source_snippet_hash = self._hash_source_snippet(start_line, end_line)

        params = []
        # Calculate which arguments have defaults
        # defaults are stored in reverse order from the end of the args list
        num_args = len(node.args.args)
        num_defaults = len(node.args.defaults)
        args_with_defaults = set(num_args - num_defaults + i for i in range(num_defaults))

        for i, arg in enumerate(node.args.args):
            param_name = arg.arg
            param_type = self._unparse_annotation(arg.annotation)

            # Check if this argument has a default value
            default_value = None
            if i in args_with_defaults:
                default_idx = i - (num_args - num_defaults)
                if 0 <= default_idx < len(node.args.defaults):
                    default_value = self._unparse_annotation(node.args.defaults[default_idx])

            if param_type:
                if default_value:
                    params.append(f"{param_name}: {param_type} = {default_value}")
                else:
                    params.append(f"{param_name}: {param_type}")
            else:
                if default_value:
                    params.append(f"{param_name} = {default_value}")
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
            full_source=self.full_source, # Preserve exact original source
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
        attributes = []
        for n in node.body:
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
                    elif isinstance(target, ast.Tuple):
                        # Handle tuple unpacking like a, b = 1, 2
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                attributes.append(elt.id)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                # Handle annotated assignments like self.value: int
                attributes.append(n.target.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == '__init__':
                # Look for instance variable assignments in __init__ method
                for sub_node in ast.walk(n):
                    if isinstance(sub_node, ast.Assign):
                        for target in sub_node.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                                # Found self.attribute = value
                                attributes.append(target.attr)
        docstring = ast.get_docstring(node)

        # --- NEW METADATA EXTRACTION FOR CLASSES ---
        decorators = self._extract_decorators(node)

        class_element = ClassElement(
            name=node.name,
            kind='class',
            file_path=self.current_file,
            line_start=start_line,
            line_end=end_line,
            full_source=self.full_source, # Preserve exact original source
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

    def __init__(self, enable_performance_monitoring: bool = True):
        self.visitor = PythonASTVisitor()
        self.project_root: Optional[Path] = None # Store the project root for consistent FQNs

        # Performance monitoring
        self.enable_performance_monitoring = enable_performance_monitoring
        self.performance_metrics = {
            'total_files': 0,
            'total_elements': 0,
            'processing_time': 0.0,
            'ast_parsing_time': 0.0,
            'regex_compilation_time': 0.0,
            'io_time': 0.0
        }

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

        start_time = time.time()

        try:
            io_start = time.time()
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            io_time = time.time() - io_start

            ast_start = time.time()
            tree = ast.parse(source, filename=str(file_path))
            file_imports, import_from_targets = self._extract_file_imports(tree)
            ast_time = time.time() - ast_start

            visitor_start = time.time()
            self.visitor.source_lines = source.splitlines()
            self.visitor.current_file = str(file_path.resolve())
            self.visitor.full_source = source  # Preserve exact original source
            self.visitor.elements = [] # Reset elements for each file
            self.visitor.file_level_imports = file_imports
            self.visitor.file_level_import_from_targets = import_from_targets

            self.visitor.visit(tree)
            visitor_time = time.time() - visitor_start

            total_time = time.time() - start_time
            elements = self.visitor.elements.copy()

            # Update performance metrics
            if self.enable_performance_monitoring:
                self.performance_metrics['total_files'] += 1
                self.performance_metrics['total_elements'] += len(elements)
                self.performance_metrics['processing_time'] += total_time
                self.performance_metrics['ast_parsing_time'] += ast_time
                self.performance_metrics['io_time'] += io_time

            return elements

        except Exception as e:
            total_time = time.time() - start_time
            # Update performance metrics even for failed files
            if self.enable_performance_monitoring:
                self.performance_metrics['total_files'] += 1
                self.performance_metrics['processing_time'] += total_time
                self.performance_metrics['io_time'] += io_time

            # Re-raise to show the error if it's critical, or return empty if tolerable
            # For now, print a warning with the specific file if errors occur
            logging.info(f"   Warning: Error processing {file_path}: {e}", file=sys.stderr)
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

        logging.info(f"Found {len(filtered_files)} Python files to analyze (after filtering .gitignore).")

        start_time = time.time()

        # Use tqdm for progress indication
        for file_path in filtered_files:
            elements.extend(self.extract_from_file(file_path))

        total_time = time.time() - start_time

        # Print performance summary if enabled
        if self.enable_performance_monitoring:
            self._print_performance_summary(total_time)

        return elements

    def _print_performance_summary(self, total_time: float):
        """Print detailed performance metrics for Python analysis."""
        metrics = self.performance_metrics
        logging.info(f"   📊 Python Analysis Performance Summary:")
        logging.info(f"      • Files processed: {metrics['total_files']}")
        logging.info(f"      • Elements found: {metrics['total_elements']}")
        logging.info(f"      • Total time: {total_time:.3f}s")
        logging.info(f"      • AST parsing time: {metrics['ast_parsing_time']:.3f}s")
        logging.info(f"      • I/O time: {metrics['io_time']:.3f}s")

        if metrics['total_files'] > 0:
            avg_time_per_file = total_time / metrics['total_files']
            logging.info(f"      • Avg time per file: {avg_time_per_file:.3f}s")

        if metrics['total_elements'] > 0:
            avg_time_per_element = total_time / metrics['total_elements']
            logging.info(f"      • Avg time per element: {avg_time_per_element:.4f}s")

        # Calculate efficiency metrics
        if total_time > 0:
            ast_efficiency = (metrics['ast_parsing_time'] / total_time) * 100
            io_efficiency = (metrics['io_time'] / total_time) * 100
            logging.info(f"      • AST parsing efficiency: {ast_efficiency:.1f}%")
            logging.info(f"      • I/O efficiency: {io_efficiency:.1f}%")

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def reset_performance_metrics(self):
        """Reset performance metrics for new analysis."""
        self.performance_metrics = {
            'total_files': 0,
            'total_elements': 0,
            'processing_time': 0.0,
            'ast_parsing_time': 0.0,
            'regex_compilation_time': 0.0,
            'io_time': 0.0
        }