"""
Builds a call graph from extracted code elements.
Uses explicit data structures to minimize cognitive load.
"""

import os
from typing import Dict, List, Set, Optional, Any, Tuple, cast
from dataclasses import dataclass, field
from pathlib import Path
import ast
import sys
import logging
import itertools

from code_flow.core.models import CodeElement, FunctionElement

@dataclass
class CallEdge:
    """Represents a single function call from caller to callee."""
    caller: str  # Fully qualified name
    callee: str  # Fully qualified name
    file_path: str
    line_number: int
    call_type: str  # 'direct', 'indirect', 'dynamic'
    parameters: List[str]  # Parameter names/types
    is_static_call: bool
    confidence: float  # How certain we are about this call

@dataclass
class FunctionNode:
    """Represents a function in the call graph with complete metadata."""
    name: str
    fully_qualified_name: str
    file_path: str
    line_start: int
    line_end: int # <--- ADDED THIS!
    parameters: List[str]
    incoming_edges: List[CallEdge] = field(default_factory=list) # Default to empty list
    outgoing_edges: List[CallEdge] = field(default_factory=list) # Default to empty list
    return_type: Optional[str] = None
    is_entry_point: bool = False
    is_exported: bool = False
    is_async: bool = False
    # Additional attributes from FunctionElement for complete analysis
    is_static: bool = False
    access_modifier: Optional[str] = None
    docstring: Optional[str] = None
    is_method: bool = False
    class_name: Optional[str] = None
    # --- NEW ATTRIBUTES (Copied from FunctionElement) ---
    complexity: Optional[int] = None
    nloc: Optional[int] = None
    external_dependencies: List[str] = field(default_factory=list)
    decorators: List[Dict[str, Any]] = field(default_factory=list)
    catches_exceptions: List[str] = field(default_factory=list)
    local_variables_declared: List[str] = field(default_factory=list)
    hash_body: Optional[str] = None
    summary: Optional[str] = None  # Natural language summary for Meta-RAG


@dataclass
class ModuleNode:
    """Represents a module in the call graph."""
    name: str
    file_path: str
    functions: List[FunctionNode] = field(default_factory=list) # Ensure default_factory for mutable type
    imports: Dict[str, str] = field(default_factory=dict)  # module -> imported items (simplified)
    exports: List[str] = field(default_factory=list)
    is_entry_point: bool = False

@dataclass
class PotentialCall:
    """Represents a potential function call found in source code."""
    callee: str
    file_path: str
    line_number: int
    parameters: List[str]
    context: str  # Surrounding line for context

class CallGraphBuilder:
    """Builds call graphs from code elements with explicit processing steps."""

    def __init__(self, confidence_threshold: float = 0.8):
        self.functions: Dict[str, FunctionNode] = {}
        self.modules: Dict[str, ModuleNode] = {}
        self.edges: List[CallEdge] = []
        self.functions_by_name: Dict[str, FunctionNode] = {}  # Local name to node mapping
        self.known_imports: Dict[str, Dict[str, str]] = {}  # module -> {local_name: fqn} (Not fully implemented/used here)
        self.project_root: Optional[Path] = None
        self.confidence_threshold = confidence_threshold
        self.module_symbol_tables: Dict[str, Dict[str, str]] = {}
        self.module_imports: Dict[str, Dict[str, str]] = {}

    def add_function(self, func_element: FunctionElement) -> FunctionNode:
        """Add a function to the graph with fully qualified naming and complete metadata."""
        module_name = self._get_module_name(Path(func_element.file_path))
        fqn = f"{module_name}.{func_element.name}"

        node = FunctionNode(
            name=func_element.name,
            fully_qualified_name=fqn,
            file_path=str(func_element.file_path),
            line_start=func_element.line_start,
            line_end=func_element.line_end, # <--- POPULATED HERE!
            parameters=func_element.parameters,
            return_type=func_element.return_type,
            is_async=func_element.is_async,
            is_static=func_element.is_static,
            access_modifier=func_element.access_modifier,
            docstring=func_element.docstring,
            is_method=func_element.is_method,
            class_name=func_element.class_name,
            # Initialize new attributes from func_element
            complexity=func_element.complexity,
            nloc=func_element.nloc,
            external_dependencies=func_element.external_dependencies,
            decorators=func_element.decorators,
            catches_exceptions=func_element.catches_exceptions,
            local_variables_declared=func_element.local_variables_declared,
            hash_body=func_element.hash_body,
            summary=func_element.summary,  # Copy summary from FunctionElement
            incoming_edges=[], # These are populated later
            outgoing_edges=[] # These are populated later
        )

        self.functions[fqn] = node
        # IMPORTANT: functions_by_name is a simplified map. If multiple functions
        # have the same simple name (e.g., in different modules), this will
        # overwrite. For robust resolution, would need FQN.
        self.functions_by_name[func_element.name] = node

        if module_name not in self.modules:
            self.modules[module_name] = ModuleNode(
                name=module_name,
                file_path=str(func_element.file_path),
                imports={},
                exports=[],
                functions=[]
            )
        self.modules[module_name].functions.append(node)

        # logging.info(f"   Added function: {fqn} {'(method)' if func_element.is_method else ''}{' (async)' if func_element.is_async else ''}")
        return node

    def build_from_elements(self, elements: List[CodeElement]) -> None:
        """Build the complete call graph from code elements."""
        # Note: project_root should be explicitly set by the caller (CLI or MCP analyzer)
        # before calling this method. If not set, _get_module_name will use appropriate fallbacks.

        logging.info(f"Building call graph from {len(elements)} code elements...")

        # for e in elements:
        #     logging.info(f" - Element: {e.name} ({type(e).__name__}) in {e.file_path}")
    
        function_elements = [e for e in elements if hasattr(e, 'kind') and e.kind == 'function']
        logging.info(f"Step 1: Creating function nodes... {len(function_elements)} functions found")
        for element in function_elements:
            self.add_function(cast(FunctionElement, element))

        logging.info(f"   Created {len(self.functions)} function nodes")
        logging.info(f"   Methods: {sum(1 for f in self.functions.values() if f.is_method)}")
        logging.info(f"   Async functions: {sum(1 for f in self.functions.values() if f.is_async)}")
        # logging.info(f"   Available functions: {list(self.functions_by_name.keys())[:10]}...")

        logging.info("Step 2: Building module symbol tables...")
        self._build_module_symbol_tables(elements)

        logging.info("Step 3: Extracting call edges...")
        for element in function_elements:
            self._extract_calls_from_function(cast(FunctionElement, element))

        logging.info(f"   Found {len(self.edges)} call edges")

        logging.info("Step 4: Identifying entry points...")
        self._identify_entry_points()

    def _build_module_symbol_tables(self, elements: List[CodeElement]) -> None:
        """Build symbol tables per module for local defs and imports."""
        self.module_symbol_tables = {}
        self.module_imports = {}

        for element in elements:
            if not hasattr(element, "file_path"):
                continue
            module_name = self._get_module_name(Path(element.file_path))
            self.module_symbol_tables.setdefault(module_name, {})
            if hasattr(element, "kind") and element.kind == "function":
                self.module_symbol_tables[module_name][element.name] = f"{module_name}.{element.name}"

        for module_name, module in self.modules.items():
            self.module_imports[module_name] = self._extract_imports_for_module(Path(module.file_path))

    def _extract_calls_from_function(self, func_element: FunctionElement) -> None:
        """Extract all function calls from a single function's source using AST for Python, regex for TypeScript."""
        try:
            with open(func_element.file_path, 'r', encoding='utf-8') as f:
                full_source = f.read()

            # Handle TypeScript files differently - use regex-based extraction
            if func_element.file_path.endswith(('.ts', '.tsx')):
                calls = self._extract_calls_from_typescript_function(func_element, full_source)
            else:
                # Use AST parsing for Python files
                tree = ast.parse(full_source, filename=func_element.file_path)

                # Find the specific function node using its name and exact start line for precision
                func_node = self._find_function_node(tree, func_element.name, func_element.line_start)
                if not func_node:
                    # logging.info(f"   Warning: Could not find AST node for function {func_element.name} at line {func_element.line_start} in {func_element.file_path}")
                    return

                calls = self._extract_calls_from_ast_node(func_node, func_element)

            module_name = self._get_module_name(Path(func_element.file_path))
            caller_fqn = f"{module_name}.{func_element.name}"

            for call in calls:
                self._create_edge_if_valid(caller_fqn, call, func_element)

        except Exception as e:
            logging.warning(f"   Warning: Error extracting calls from {func_element.name} in {func_element.file_path}: {e}")

    def _extract_calls_from_typescript_function(self, func_element: FunctionElement, full_source: str) -> List[PotentialCall]:
        """Extract function calls from TypeScript function using comprehensive regex patterns."""
        calls = []

        try:
            lines = full_source.splitlines()
            func_start = func_element.line_start - 1  # Convert to 0-based indexing
            func_end = func_element.line_end

            # Extract the function body lines
            func_body_lines = lines[func_start:func_end] if func_end > func_start else lines[func_start:func_start+1]
            func_body = '\n'.join(func_body_lines)

            # Debug: print function body for problematic cases
            if func_element.name in ['getUser', 'processUser', 'main']:
                logging.info(f"     Function {func_element.name} body (lines {func_start+1}-{func_end}):")
                for i, line in enumerate(func_body_lines):
                    logging.info(f"       {func_start + i + 1}: {line}")

            # Enhanced patterns to match various TypeScript function call patterns
            import re

            # Comprehensive call patterns for TypeScript - more specific and ordered by priority
            call_patterns = [
                # Method calls with 'this': this.methodName() or this.database.findUser()
                r'\bthis\.([a-zA-Z_$][a-zA-Z0-9_$]*(?:\.[a-zA-Z_$][a-zA-Z0-9_$]*)*)\s*\(',
                # Method calls with 'super': super.methodName()
                r'\bsuper\.([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(',
                # Calls with await: await functionName() or await this.getUser()
                r'\bawait\s+([a-zA-Z_$][a-zA-Z0-9_$]*(?:\.[a-zA-Z_$][a-zA-Z0-9_$]*)*)\s*\(',
                # Constructor calls: new ClassName()
                r'\bnew\s+([A-Z][a-zA-Z0-9_$]*(?:\.[a-zA-Z_$][a-zA-Z0-9_$]*)*)\s*\(',
                # Static method calls: ClassName.methodName() or DatabaseImpl.findUser()
                r'\b([A-Z][a-zA-Z0-9_$]*(?:\.[a-zA-Z_$][a-zA-Z0-9_$]*)*)\s*\(',
                # Regular function calls: functionName()
                r'\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(',
                # Generic function calls: functionName<T>()
                r'\b([a-zA-Z_$][a-zA-Z0-9_$]*(?:<[^>]*>)*)\s*\(',
            ]

            for pattern in call_patterns:
                for match in re.finditer(pattern, func_body):
                    call_text = match.group(1)
                    if call_text and self._is_valid_typescript_call(call_text):
                        line_number = func_start + match.start() + 1

                        # Extract parameters
                        params = self._extract_typescript_call_parameters(match.group(0))

                        if not self._is_allowed_typescript_call(call_text, func_element):
                            continue
                        call = PotentialCall(
                            callee=call_text,
                            file_path=func_element.file_path,
                            line_number=line_number,
                            parameters=params,
                            context=lines[line_number-1].strip() if line_number <= len(lines) else ""
                        )
                        calls.append(call)

        except Exception as e:
            logging.warning(f"   Warning: Error in TypeScript call extraction for {func_element.name}: {e}")

        return calls

    def _is_valid_typescript_call(self, call_text: str) -> bool:
        """Check if a potential call is valid (not a keyword or built-in)."""
        # TypeScript keywords and built-ins to avoid
        invalid_calls = {
            'if', 'for', 'while', 'switch', 'case', 'try', 'catch', 'finally',
            'function', 'class', 'interface', 'type', 'enum', 'namespace',
            'return', 'break', 'continue', 'throw', 'yield', 'await', 'async',
            'import', 'export', 'from', 'as', 'typeof', 'instanceof', 'in', 'of',
            'true', 'false', 'null', 'undefined', 'this', 'super', 'new',
            'console.log', 'console.error', 'console.warn', 'console.info',
            'Math.', 'JSON.', 'Object.', 'Array.', 'String.', 'Number.', 'Boolean.',
            'Date.', 'RegExp.', 'Promise.', 'Symbol.',
            # Add common TypeScript/JavaScript built-ins that might be called
            'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'encodeURI', 'decodeURI',
            'encodeURIComponent', 'decodeURIComponent', 'eval', 'setTimeout', 'setInterval',
            'clearTimeout', 'clearInterval', 'require', 'module', 'exports', 'global',
            'process', 'Buffer', 'setImmediate', 'clearImmediate'
        }

        # Check if it's a direct invalid call
        if call_text in invalid_calls:
            return False

        # Check if it starts with any invalid pattern
        for invalid in invalid_calls:
            if call_text.startswith(invalid):
                return False

        # Allow 'this' and 'super' method calls (these are valid)
        if '.' in call_text:
            parts = call_text.split('.')
            if parts[0] in ['this', 'super']:
                return True

        return True

    def _is_allowed_typescript_call(self, call_text: str, func_element: FunctionElement) -> bool:
        module_name = self._get_module_name(Path(func_element.file_path))
        module_symbols = self.module_symbol_tables.get(module_name, {})
        module_imports = self.module_imports.get(module_name, {})
        base = call_text.split("<")[0]
        if base in module_symbols or base in module_imports:
            return True
        if "." in base:
            prefix = base.split(".", 1)[0]
            if prefix in module_symbols or prefix in module_imports:
                return True
        return False

    def _extract_typescript_call_parameters(self, call_text: str) -> List[str]:
        """Extract parameters from a TypeScript function call."""
        try:
            import re

            # Extract parameter list from function call
            param_match = re.search(r'\(([^)]*)\)', call_text)
            if param_match:
                param_list = param_match.group(1).strip()
                if param_list:
                    # Simple parameter extraction - split by comma, respecting basic nesting
                    params = []
                    current = ""
                    depth = 0

                    for char in param_list:
                        if char in '([{' :
                            depth += 1
                            current += char
                        elif char in ')]}':
                            depth -= 1
                            current += char
                        elif char == ',' and depth == 0:
                            if current.strip():
                                params.append(current.strip())
                            current = ""
                        else:
                            current += char

                    if current.strip():
                        params.append(current.strip())

                    return params[:5]  # Limit to first 5 parameters
            return []
        except Exception:
            return []

    def _find_function_node(self, tree: ast.AST, func_name: str, line_start: int) -> Optional[ast.FunctionDef|ast.AsyncFunctionDef]:
        """Find the specific function node in the AST tree by name and approximate line number."""
        class FunctionFinder(ast.NodeVisitor):
            def __init__(self, target_name: str, target_line: int):
                self.target_name = target_name
                self.target_line = target_line
                self.found_node = None

            def visit_FunctionDef(self, node):
                # Check for exact name and line number match
                if node.name == self.target_name and getattr(node, 'lineno', 0) == self.target_line:
                    self.found_node = node
                    return # Stop visiting children once found
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                if node.name == self.target_name and getattr(node, 'lineno', 0) == self.target_line:
                    self.found_node = node
                    return
                self.generic_visit(node)

        finder = FunctionFinder(func_name, line_start)
        finder.visit(tree)
        return finder.found_node

    def _extract_calls_from_ast_node(self, node: ast.AST, func_element: FunctionElement) -> List[PotentialCall]:
        """Extract function calls from an AST node using proper AST traversal."""
        class CallExtractor(ast.NodeVisitor):
            def __init__(self, parent_func_element: FunctionElement, source_lines: List[str]):
                self.calls = []
                self.parent_file = parent_func_element.file_path
                self.start_line = parent_func_element.line_start
                self.end_line = parent_func_element.line_end
                self.source_lines = source_lines

            def visit_Call(self, node: ast.Call):
                # Only consider calls within the function body's reported lines
                node_lineno = getattr(node, 'lineno', 0)
                if self.start_line <= node_lineno <= self.end_line:
                    callee_name = self._get_callee_name(node.func)
                    if callee_name and self._is_valid_call_target(callee_name):
                        line_number = node_lineno
                        context = ""
                        if line_number > 0 and line_number <= len(self.source_lines):
                            context = self.source_lines[line_number-1].strip()

                        params = self._extract_call_parameters(node)

                        call = PotentialCall(
                            callee=callee_name,
                            file_path=self.parent_file,
                            line_number=line_number,
                            parameters=params,
                            context=context
                        )
                        self.calls.append(call)
                self.generic_visit(node)

            def _get_callee_name(self, func: ast.AST) -> Optional[str]:
                if isinstance(func, ast.Name):
                    return func.id
                elif isinstance(func, ast.Attribute):
                    # For obj.method(), we want "method" but also context if possible.
                    # For now, we'll try to get "obj.method" if obj is a Name.
                    if isinstance(func.value, ast.Name):
                        return f"{func.value.id}.{func.attr}"
                    return func.attr # Fallback for complex func.value
                elif isinstance(func, ast.Subscript): # e.g. some_dict['func_name']()
                    return "<dynamic_subscript_call>"
                return None

            def _is_valid_call_target(self, name: str) -> bool:
                """Filter out non-function call identifiers and common built-ins."""
                invalid_keywords = {
                    'if', 'for', 'while', 'with', 'try', 'except', 'class', 'def', 'import', 'from',
                    'return', 'yield', 'in', 'is', 'not', 'and', 'or', 'True', 'False', 'None',
                    'self', 'cls', # 'super', # Super can be called, so don't exclude it here.
                    'int', 'str', 'list', 'dict', 'set', 'tuple', 'bool', 'float' # Built-in types that can be called for casting
                }
                # Also exclude methods of common built-in types e.g., list.append, str.lower
                # This is a heuristic and not exhaustive.
                if '.' in name:
                    obj_name, method_name = name.split('.', 1)
                    if obj_name in {'str', 'list', 'dict', 'set', 'tuple'}:
                        return False

                return name not in invalid_keywords and not name.startswith('__')

            def _extract_call_parameters(self, node: ast.Call) -> List[str]:
                params = []
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        params.append(arg.id)
                    elif isinstance(arg, ast.Constant):
                        params.append(repr(arg.value)) # Use repr for better representation of literals
                    else:
                        params.append(str(type(arg).__name__)) # Generic type name for complex args
                return params[:5] # Limit to first 5 for brevity

        # Pass source_lines for context extraction in CallExtractor
        extractor = CallExtractor(func_element, func_element.full_source.splitlines())
        extractor.visit(node)

        # logging.info(f"     Found {len(extractor.calls)} potential calls in {func_element.name}")
        return extractor.calls

    def _create_edge_if_valid(self, caller_fqn: str, call: PotentialCall,
                             func_element: FunctionElement) -> None:
        """Create a call edge if the target function exists with improved resolution."""
        target_fqn = None
        target_node = None
        confidence = 0.0

        module_name = self._get_module_name(Path(func_element.file_path))
        module_symbols = self.module_symbol_tables.get(module_name, {})
        module_imports = self.module_imports.get(module_name, {})

        def _resolve_imported_target(callee: str, imports: Dict[str, str]) -> Optional[str]:
            if callee in imports:
                return imports[callee]
            if "." in callee:
                base, attr = callee.split(".", 1)
                base_target = imports.get(base)
                if base_target:
                    return f"{base_target}.{attr}"
            return None

        # Strategy 1: Check if it's a local function call by its simple name
        if call.callee in module_symbols:
            target_fqn = module_symbols[call.callee]
            target_node = self.functions.get(target_fqn)
            confidence = 0.95

        # Strategy 2: Handle method calls (e.g., "obj.method", "this.method", "super.method")
        elif '.' in call.callee:
            parts = call.callee.split('.')
            method_name = parts[-1]

            imported_target = _resolve_imported_target(call.callee, module_imports)
            if imported_target:
                target_fqn = imported_target
                if target_fqn in self.functions:
                    target_node = self.functions.get(target_fqn)
                confidence = 0.9

            # First try to find the exact method name
            if not target_node and method_name in module_symbols:
                candidate_node = self.functions_by_name[method_name]
                # Check if this could be a method (has class_name or is_method=True)
                if candidate_node.is_method or candidate_node.class_name:
                    target_node = candidate_node
                    target_fqn = candidate_node.fully_qualified_name
                    confidence = 0.95

            # If not found, try to find any method with this name
            if not target_node and method_name in self.functions_by_name:
                for fqn, node in self.functions.items():
                    if (node.name == method_name and
                        (node.is_method or node.class_name) and
                        node.fully_qualified_name != caller_fqn):  # Avoid self-calls
                        target_node = node
                        target_fqn = node.fully_qualified_name
                        confidence = 0.6
                        break

        # Strategy 3: Handle constructor calls and static method calls
        elif call.callee in module_symbols:
            # Look for methods or class functions with this name
            candidate_fqn = module_symbols.get(call.callee)
            if candidate_fqn in self.functions:
                candidate_node = self.functions[candidate_fqn]
                if candidate_node.is_method or candidate_node.class_name:
                    target_node = candidate_node
                    target_fqn = candidate_node.fully_qualified_name
                    confidence = 0.95

        # Strategy 4: Handle generic function calls (functionName<T>())
        else:
            # Remove generic type parameters for matching
            simple_name = call.callee.split('<')[0] if '<' in call.callee else call.callee
            if simple_name in module_symbols:
                target_fqn = module_symbols[simple_name]
                target_node = self.functions.get(target_fqn)
                confidence = 0.95

            imported_target = _resolve_imported_target(simple_name, module_imports)
            if not target_node and imported_target:
                target_fqn = imported_target
                if target_fqn in self.functions:
                    target_node = self.functions.get(target_fqn)
                confidence = 0.9

        # Final check if the resolved FQN actually exists in our graph
        if target_fqn and target_fqn in self.functions and confidence > 0:
            # Determine call type based on the target
            call_type = 'direct'
            if target_node and target_node.is_method:
                call_type = 'method'
            elif target_node and target_node.class_name:
                call_type = 'static_method'

            # Determine if it's a static call
            is_static_call = (call_type == 'static_method' or
                             (target_node and target_node.is_static) or
                             '.' not in call.callee or
                             call.callee.startswith(('this.', 'super.')))

            edge = CallEdge(
                caller=caller_fqn,
                callee=target_fqn,
                file_path=call.file_path,
                line_number=call.line_number,
                call_type=call_type,
                parameters=call.parameters,
                is_static_call=is_static_call,
                confidence=confidence
            )

            self.edges.append(edge)
            self.functions[caller_fqn].outgoing_edges.append(edge)
            self.functions[target_fqn].incoming_edges.append(edge)
            # logging.info(f"     ✓ Created edge: {caller_fqn} -> {target_fqn} ({call_type}, {confidence:.2f})")
        # else:
            # logging.info(f"     Info: Could not resolve call target for '{call.callee}' from '{caller_fqn}'")

    def get_drift_edges(self) -> List[CallEdge]:
        """Return edges above the configured confidence threshold for drift analysis."""
        return [edge for edge in self.edges if edge.confidence >= self.confidence_threshold]


    def _get_module_name(self, file_path: Path) -> str:
        """
        Convert file path to module name relative to the project root.
        Attempts to find the package root (directory containing code_flow)
        to make FQNs consistent regardless of CWD.
        """
        if self.project_root:
            try:
                relative_path = file_path.relative_to(self.project_root)
                module_path = str(relative_path).replace('.py', '').replace(os.sep, '.')
                if module_path.endswith('.__init__'):
                    return module_path[:-9]
                return module_path
            except ValueError:
                # file_path is not relative to self.project_root, use a fallback
                pass

        # Fallback if project_root is not set or file is outside it
        # Try to find the root of the 'codeflow' package
        try:
            package_base = Path(__file__).parent.parent.parent.resolve()
            relative_path = file_path.relative_to(package_base)
            module_path = str(relative_path).replace('.py', '').replace(os.sep, '.')
            if module_path.endswith('.__init__'):
                return module_path[:-9]
            return module_path
        except ValueError:
            # If still not relative to package base, fall back to current working directory
            try:
                relative_path = file_path.relative_to(Path.cwd())
                module_path = str(relative_path).replace('.py', '').replace(os.sep, '.')
                if module_path.endswith('.__init__'):
                    return module_path[:-9]
                return module_path
            except ValueError:
                # Last resort: use full path, but this often results in very long FQNs
                logging.warning(f"   Warning: Could not determine relative module name for {file_path}. Using absolute path segment.")
                return str(file_path.stem) # Use just the file name as module name, without path

    def _extract_imports_for_module(self, file_path: Path) -> Dict[str, str]:
        """Extract Python/TypeScript imports for a module."""
        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception:
            return {}

        imports: Dict[str, str] = {}
        if file_path.suffix == ".py":
            try:
                tree = ast.parse(source, filename=str(file_path))
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            local_name = alias.asname or alias.name.split(".")[-1]
                            imports[local_name] = alias.name
                    elif isinstance(node, ast.ImportFrom):
                        if not node.module:
                            continue
                        for alias in node.names:
                            local_name = alias.asname or alias.name
                            imports[local_name] = f"{node.module}.{alias.name}"
            except Exception:
                return {}
        else:
            import re
            for match in re.finditer(r"import\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]", source):
                names = match.group(1)
                module = match.group(2)
                for name in names.split(","):
                    local = name.strip()
                    if not local:
                        continue
                    if " as " in local:
                        original, alias = [part.strip() for part in local.split(" as ", 1)]
                        imports[alias] = module
                    else:
                        imports[local] = module

            for match in re.finditer(r"import\s+\*\s+as\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]", source):
                alias = match.group(1)
                module = match.group(2)
                imports[alias] = module

            for match in re.finditer(r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]", source):
                alias = match.group(1)
                module = match.group(2)
                imports[alias] = module

        return imports

    def compute_metrics(self) -> Dict[str, Any]:
        edges_per_module: Dict[str, int] = {}
        for edge in self.edges:
            source_module = self._get_module_name(Path(self.functions[edge.caller].file_path))
            edges_per_module[source_module] = edges_per_module.get(source_module, 0) + 1

        low_confidence = sum(1 for edge in self.edges if edge.confidence < self.confidence_threshold)

        return {
            "total_edges": len(self.edges),
            "edges_per_module": edges_per_module,
            "low_confidence_edges": low_confidence,
        }

    def _identify_entry_points(self) -> None:
        """Identify entry points using multiple strategies."""
        entry_point_patterns = {
            'python': ['main', 'run', '__main__', 'app', 'server', 'start', 'init', 'create', 'cli', 'execute'],
            'typescript': ['main', 'run', 'app', 'server', 'start', 'init', 'bootstrap', 'index', 'entry']
        }

        # Use appropriate patterns based on file extensions in the project
        has_ts_files = any(f.file_path.endswith(('.ts', '.tsx')) for f in self.functions.values())
        patterns = entry_point_patterns['typescript'] if has_ts_files else entry_point_patterns['python']

        logging.info("   Identifying entry points using multiple strategies...")

        # Determine all callable FQNs in the graph for quicker lookup
        all_fqns = set(self.functions.keys())

        # First pass: identify entry points using multiple strategies
        potential_entry_points = []
        for fqn, func in self.functions.items():
            # Skip functions with None names (shouldn't happen but safety check)
            if not func.name:
                continue

            # Strategy 1: Common entry point names
            if any(pattern in func.name.lower() for pattern in patterns):
                func.is_entry_point = True
                potential_entry_points.append(func)
                logging.info(f"     ✓ Entry point (name): {func.name} ({fqn})")

            # Strategy 2: Functions with no incoming edges (likely entry points), but not methods
            # and not common dunder methods that might not have explicit calls but aren't entry points
            if (len(func.incoming_edges) == 0 and
                not func.is_method and
                not func.name.startswith('__') and
                func.name not in ['_get_extractor', '_get_gitignore_patterns', '_match_file_against_pattern'] and # Specific internal helpers
                not func.is_entry_point):
                func.is_entry_point = True
                potential_entry_points.append(func)
                logging.info(f"     ✓ Entry point (no incoming): {func.name} ({fqn})")

            # Strategy 3: Functions in '__main__' modules or files containing 'main' or 'app'
            if ('__main__' in fqn or
                'main' in func.file_path.lower() or
                'app' in func.file_path.lower() or
                'index' in func.file_path.lower() or
                (has_ts_files and ('main.ts' in func.file_path or 'index.ts' in func.file_path))) and not func.is_entry_point:
                func.is_entry_point = True
                potential_entry_points.append(func)
                logging.info(f"     ✓ Entry point (location): {func.name} ({fqn})")

            # Strategy 4: Functions with common 'cli' patterns or called by Typer/Argparse
            if 'cli' in func.name.lower() or 'command' in func.name.lower():
                # Check for decorators like @app.command() if we extract them
                if any('command' in d.get('name', '') for d in func.decorators): # check for 'command' in decorator name
                    func.is_entry_point = True
                    potential_entry_points.append(func)
                elif not func.is_method and not func.is_entry_point:
                    func.is_entry_point = True
                    potential_entry_points.append(func)

            # Strategy 5: Simple top-level functions in small modules without being called
            # (Similar to strategy 2, but emphasizes module context)
            module_fqn = '.'.join(fqn.split('.')[:-1])
            if module_fqn in self.modules:
                module_functions = [f for f in self.modules[module_fqn].functions if not f.is_method and f.name]
                if len(module_functions) <= 3 and len(func.incoming_edges) == 0 and not func.is_entry_point:
                    func.is_entry_point = True
                    potential_entry_points.append(func)

        # Fallback Strategy: If no entry points found, mark the first few functions as entry points
        if not potential_entry_points and self.functions:
            logging.info("   No entry points found with standard strategies, applying fallback...")
            # Mark all functions with no incoming edges as entry points (up to 5)
            no_incoming_functions = [f for f in self.functions.values() if len(f.incoming_edges) == 0]
            for func in no_incoming_functions[:5]:  # Limit to first 5
                if not func.is_entry_point:
                    func.is_entry_point = True
                    potential_entry_points.append(func)
                    logging.info(f"     ✓ Entry point (fallback): {func.name} ({func.fully_qualified_name})")

        # Final fallback: if still no entry points, mark the first function as entry point
        if not potential_entry_points and self.functions:
            first_func = next(iter(self.functions.values()))
            if not first_func.is_entry_point:
                first_func.is_entry_point = True
                logging.info(f"     ✓ Entry point (final fallback): {first_func.name} ({first_func.fully_qualified_name})")

        entry_points = self.get_entry_points()
        logging.info(f"   Total entry points identified: {len(entry_points)}")

    def get_entry_points(self) -> List[FunctionNode]:
        """Get all identified entry points."""
        return [f for f in self.functions.values() if f.is_entry_point]

    def _score_entry_point(self, func: FunctionNode) -> Dict[str, Any]:
        """Score and categorize an entry point using lightweight heuristics."""
        name_lower = func.name.lower() if func.name else ""
        file_lower = func.file_path.lower() if func.file_path else ""
        fqn_lower = func.fully_qualified_name.lower() if func.fully_qualified_name else ""
        decorators = [d.get("name", "").lower() for d in func.decorators]

        signals: List[str] = []
        score = 0

        # Strong negatives: tests and dunder/private helpers
        if any(token in file_lower for token in ["/tests/", "/test/", "_test.py", "test_", "/tests_"]):
            signals.append("test_path")
            score -= 5
        if func.name.startswith("__") and func.name.endswith("__"):
            signals.append("dunder")
            score -= 4
        if func.name.startswith("_") and not func.name.startswith("__"):
            signals.append("private_name")
            score -= 2

        # Runtime entry hints
        runtime_name_hits = [
            "main", "run", "app", "server", "start", "bootstrap", "cli", "execute"
        ]
        if any(token in name_lower for token in runtime_name_hits):
            signals.append("runtime_name")
            score += 3
        if any(token in file_lower for token in ["__main__", "main.py", "app.py", "server.py", "cli", "index.py"]):
            signals.append("runtime_file")
            score += 3
        if "__main__" in fqn_lower:
            signals.append("runtime_module")
            score += 4

        # Framework/CLI decorators
        if any("command" in d for d in decorators):
            signals.append("cli_decorator")
            score += 3
        if any(d in ["route", "get", "post", "put", "delete", "patch"] for d in decorators):
            signals.append("http_decorator")
            score += 2

        # Graph structure signals
        if len(func.incoming_edges) == 0:
            signals.append("no_incoming")
            score += 1
        if not func.is_method:
            signals.append("top_level")
            score += 1
        if func.docstring:
            signals.append("docstring")
            score += 1
        outgoing = len(func.outgoing_edges)
        if outgoing >= 8:
            signals.append("high_fanout")
            score += 2
        elif outgoing >= 3:
            signals.append("moderate_fanout")
            score += 1
        if func.complexity and func.complexity >= 10:
            signals.append("high_complexity")
            score += 1

        # Method penalty (entry points are more likely module-level)
        if func.is_method and func.class_name:
            signals.append("method_penalty")
            score -= 1

        # Category assignment
        category = "low-signal"
        if "test_path" in signals:
            category = "test"
        elif any(s in signals for s in ["runtime_module", "runtime_file", "runtime_name"]):
            category = "runtime"
        elif any(s in signals for s in ["cli_decorator", "http_decorator"]):
            category = "framework"
        elif func.is_exported or (not func.name.startswith("_") and func.incoming_edges and outgoing > 0):
            category = "library"

        priority = "low"
        if score >= 5:
            priority = "likely"
        elif score >= 2:
            priority = "candidate"

        return {
            "score": score,
            "category": category,
            "priority": priority,
            "signals": signals,
        }

    def get_entry_points_scored(self) -> List[Dict[str, Any]]:
        """Return entry points with score and category metadata."""
        scored = []
        for func in self.get_entry_points():
            meta = self._score_entry_point(func)
            scored.append({"function": func, "meta": meta})
        scored.sort(key=lambda item: item["meta"]["score"], reverse=True)
        return scored

    def _build_subgraph(self, seed_fqns: Optional[List[str]], depth: int = 1) -> tuple[Dict[str, FunctionNode], List[CallEdge]]:
        """Build a subgraph from seed FQNs within the given depth (undirected traversal)."""
        if not seed_fqns:
            return self.functions, self.edges

        normalized_depth = max(0, depth)
        seed_set = {fqn for fqn in seed_fqns if fqn in self.functions}
        missing = [fqn for fqn in seed_fqns if fqn not in self.functions]
        if missing:
            logging.warning(f"Warning: {len(missing)} seed FQNs not found in call graph: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        if not seed_set:
            return {}, []

        neighbors: Dict[str, set[str]] = {fqn: set() for fqn in self.functions}
        for edge in self.edges:
            if edge.caller in neighbors and edge.callee in neighbors:
                neighbors[edge.caller].add(edge.callee)
                neighbors[edge.callee].add(edge.caller)

        visited: set[str] = set(seed_set)
        queue: list[tuple[str, int]] = [(fqn, 0) for fqn in seed_set]
        while queue:
            current, distance = queue.pop(0)
            if distance >= normalized_depth:
                continue
            for neighbor in neighbors.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        sub_functions = {fqn: self.functions[fqn] for fqn in visited}
        sub_edges = [edge for edge in self.edges if edge.caller in visited and edge.callee in visited]
        return sub_functions, sub_edges

    def export_graph(self, format: str = 'json', fqns: Optional[List[str]] = None, depth: int = 1, llm_optimized: bool = False) -> Optional[Dict | str]:
        """Export the call graph in various formats."""
        if not self.functions:
            logging.warning("Warning: No functions in graph to export")
            return None

        functions, edges = self._build_subgraph(fqns, depth) if fqns else (self.functions, self.edges)
        if not functions:
            logging.warning("Warning: No functions in subgraph to export")
            return None

        if format == 'json':
            graph_data = {
                'functions': {
                    f.fully_qualified_name: {
                        'name': f.name,
                        'file_path': f.file_path,
                        'line_start': f.line_start,
                        'line_end': f.line_end, # Added line_end to export
                        'is_entry_point': f.is_entry_point,
                        'is_exported': f.is_exported,
                        'is_async': f.is_async,
                        'is_method': f.is_method,
                        'class_name': f.class_name,
                        'parameters': f.parameters,
                        'docstring': f.docstring[:100] + '...' if f.docstring and len(f.docstring) > 100 else f.docstring,
                        'incoming_count': len(f.incoming_edges),
                        'outgoing_count': len(f.outgoing_edges),
                        'return_type': f.return_type or 'unknown',
                        'access_modifier': f.access_modifier or 'public',
                        # --- NEW ATTRIBUTES ---
                        'complexity': f.complexity,
                        'nloc': f.nloc,
                        'external_dependencies': f.external_dependencies,
                        'decorators': f.decorators,
                        'catches_exceptions': f.catches_exceptions,
                        'local_variables_declared': f.local_variables_declared,
                        'hash_body': f.hash_body,
                    }
                    for f in functions.values()
                },
                'edges': [
                    {
                        'caller': edge.caller,
                        'callee': edge.callee,
                        'file_path': edge.file_path,
                        'line_number': edge.line_number,
                        'call_type': edge.call_type,
                        'parameters': edge.parameters,
                        'confidence': edge.confidence
                    }
                    for edge in edges
                ],
                'entry_points': [f.fully_qualified_name for f in functions.values() if f.is_entry_point],
                'summary': {
                    'total_functions': len(functions),
                    'total_methods': sum(1 for f in functions.values() if f.is_method),
                    'total_async': sum(1 for f in functions.values() if f.is_async),
                    'total_edges': len(edges),
                    'total_modules': len({self._get_module_name(Path(f.file_path)) for f in functions.values()}),
                    'entry_points_count': sum(1 for f in functions.values() if f.is_entry_point),
                    'avg_degree': round(len(edges) / len(functions), 2) if functions else 0
                }
            }
            logging.info(
                "Exporting call graph (json) with %s functions, %s edges, depth=%s, seeds=%s",
                len(functions),
                len(edges),
                depth,
                len(fqns or []),
            )
            return graph_data
        elif format == 'mermaid':
            logging.info(
                "Exporting call graph (mermaid) with %s functions, %s edges, depth=%s, seeds=%s",
                len(functions),
                len(edges),
                depth,
                len(fqns or []),
            )
            return self.export_mermaid_graph(
                highlight_fqns=fqns,
                llm_optimized=llm_optimized,
                functions=functions,
                edges=edges,
            )
        return None

    def _sanitize_mermaid_id(self, identifier: str) -> str:
        """
        Sanitizes an identifier to be valid for Mermaid node IDs.
        Mermaid node IDs must be alphanumeric, start with a letter, and cannot contain reserved keywords.
        """
        # List of Mermaid reserved keywords that cannot be used as node IDs
        mermaid_keywords = {
            'graph', 'flowchart', 'subgraph', 'end', 'direction', 'class', 'classDef', 'style',
            'linkStyle', 'click', 'call', 'callback', 'function', 'subgraph', 'accdesc', 'accdescr',
            'accdesc:multiline', 'accdescr:multiline', 'title', 'acc', 'accdescr', 'acc:title',
            'acc:descr', 'acc:descr:multiline', 'accdescr:multiline', 'LR', 'RL', 'TB', 'BT',
            'TD', 'BR', 'BL', 'TR', 'TL', 'end', '=>', '->', '-->', '==>', '-.->', '==', 'linkStyle',
            'style', 'click', 'href', 'target', 'default', 'stroke', 'stroke-width', 'fill',
            'classDef', 'class', 'node', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        }
        
        # Start with a clean identifier
        sanitized = identifier
        
        # Replace problematic characters with underscores
        sanitized = sanitized.replace('.', '_')
        sanitized = sanitized.replace('-', '_')
        sanitized = sanitized.replace('/', '_')
        sanitized = sanitized.replace(':', '_')
        sanitized = sanitized.replace(' ', '_')
        sanitized = sanitized.replace('(', '_')
        sanitized = sanitized.replace(')', '_')
        sanitized = sanitized.replace('[', '_')
        sanitized = sanitized.replace(']', '_')
        sanitized = sanitized.replace('{', '_')
        sanitized = sanitized.replace('}', '_')
        sanitized = sanitized.replace('<', '_')
        sanitized = sanitized.replace('>', '_')
        sanitized = sanitized.replace('"', '_')
        sanitized = sanitized.replace("'", '_')
        sanitized = sanitized.replace('|', '_')
        sanitized = sanitized.replace('!', '_')
        sanitized = sanitized.replace('@', '_')
        sanitized = sanitized.replace('#', '_')
        sanitized = sanitized.replace('$', '_')
        sanitized = sanitized.replace('%', '_')
        sanitized = sanitized.replace('^', '_')
        sanitized = sanitized.replace('&', '_')
        sanitized = sanitized.replace('*', '_')
        sanitized = sanitized.replace('+', '_')
        sanitized = sanitized.replace('=', '_')
        sanitized = sanitized.replace('?', '_')
        sanitized = sanitized.replace('~', '_')
        sanitized = sanitized.replace('`', '_')
        
        # Remove any remaining non-alphanumeric characters
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', sanitized)
        
        # Ensure it starts with a letter or underscore
        if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = 'node_' + sanitized
            
        # Convert to lowercase to avoid case-sensitivity issues
        sanitized = sanitized.lower()
        
        # If it's a reserved keyword, add a prefix
        if sanitized in mermaid_keywords:
            sanitized = 'func_' + sanitized
            
        # Ensure it's not empty after sanitization
        if not sanitized.strip():
            sanitized = 'node_' + str(abs(hash(identifier)) % 1000)
            
        return sanitized

    def _generate_short_alias(self, fqn: str, existing_aliases: Set[str]) -> str:
        """
        Generates a short, unique alias for an FQN.
        Prioritizes the function's simple name, then adds suffixes if needed.
        """
        # Try simple name first, but sanitize it
        simple_name = self._sanitize_mermaid_id(fqn.split('.')[-1])
        if simple_name not in existing_aliases:
            return simple_name

        # If simple name conflicts, try with a short prefix from the module
        parts = fqn.split('.')
        if len(parts) > 1:
            module_prefix = self._sanitize_mermaid_id(parts[-2]) # e.g., 'core' from 'code_flow.core.func'
            alias_candidate = f"{module_prefix}_{simple_name}"
            if alias_candidate not in existing_aliases:
                return alias_candidate

        # Fallback: append a short, unique suffix (e.g., 'A', 'B', 'AA', 'AB'...)
        # This guarantees uniqueness, essential for Mermaid IDs.
        for suffix_len in range(1, 4): # Try suffixes of length 1, 2, 3
            for suffix_chars in itertools.product('ABCDEFGHIJKLMNOPQRSTUVWXYZ', repeat=suffix_len):
                suffix = "".join(suffix_chars)
                alias_candidate = f"{simple_name}_{suffix}"
                if alias_candidate not in existing_aliases:
                    return alias_candidate

        # If all else fails (highly unlikely), use a hash-based short ID
        return f"node_{abs(hash(fqn)) % 10000}"


    def export_mermaid_graph(
        self,
        highlight_fqns: Optional[List[str]] = None,
        llm_optimized: bool = False,
        functions: Optional[Dict[str, FunctionNode]] = None,
        edges: Optional[List[CallEdge]] = None,
    ) -> str:
        """
        Exports the call graph in Mermaid format (Flowchart or Graph).
        Args:
            highlight_fqns: Optional list of FQNs to highlight in the graph.
            llm_optimized: If True, optimizes output for LLM ingestion (removes styling, uses aliases).
        Returns:
            A string containing the Mermaid graph definition.
        """
        functions = functions or self.functions
        edges = edges or self.edges

        if not functions and not edges:
            return "graph TD\n    No graph data available."

        mermaid_lines = []
        seen_nodes_in_graph = set()

        relevant_fqns = set()
        if highlight_fqns:
            relevant_fqns.update(highlight_fqns)
        for edge in edges:
            if edge.caller in functions and edge.callee in functions:
                relevant_fqns.add(edge.caller)
                relevant_fqns.add(edge.callee)

        if not relevant_fqns and functions:
            for func_fqn, _ in list(functions.items())[:5]:
                relevant_fqns.add(func_fqn)

        # Map FQN to alias (node ID) and actual label
        fqn_to_alias: Dict[str, str] = {}
        alias_to_fqn: Dict[str, str] = {} # For LLM to map back
        used_aliases: Set[str] = set()

        # We'll collect all node definitions, edge definitions, and (if not LLM optimized) style commands.
        node_definitions = []
        edge_definitions = []
        style_commands = []
        alias_mapping_comments = [] # For LLM to understand aliases

        # 1. Generate Aliases and Node Definitions
        for fqn in sorted(list(relevant_fqns)):
            func = functions.get(fqn)
            if not func:
                continue

            # Use proper sanitization for node IDs
            node_id = self._sanitize_mermaid_id(fqn)

            if llm_optimized:
                # Use a shorter, human-readable alias if optimized for LLM
                alias = self._generate_short_alias(fqn, used_aliases)
                fqn_to_alias[fqn] = alias
                alias_to_fqn[alias] = fqn # Store reverse mapping for LLM guidance
                used_aliases.add(alias)
                node_id_for_mermaid = alias # Use the alias as Mermaid's internal ID
                node_label = func.name # Keep the simple function name as the label
                alias_mapping_comments.append(f'%% Alias: {alias} = {fqn} (Function: {func.name})')
            else:
                # Use the sanitized FQN as node ID, no separate alias
                fqn_to_alias[fqn] = node_id
                node_id_for_mermaid = node_id
                node_label = func.name

            if not node_id_for_mermaid.strip(): # Fallback for empty ID
                 node_id_for_mermaid = f"node_{abs(hash(fqn))}"

            node_definitions.append(f'    {node_id_for_mermaid}("{node_label}")')
            seen_nodes_in_graph.add(fqn)

            if not llm_optimized:
                styles = []
                if func.is_entry_point:
                    styles.append("fill:#e0ffc7")
                    styles.append("stroke:#3c3")
                if highlight_fqns and fqn in highlight_fqns:
                    styles.append("stroke:red")
                    styles.append("stroke-width:2px")

                if styles:
                    style_commands.append(f'    style {node_id_for_mermaid} {",".join(styles)}')


        # 2. Define Edges
        for edge in edges:
            if edge.caller in seen_nodes_in_graph and edge.callee in seen_nodes_in_graph:
                caller_id_for_mermaid = fqn_to_alias.get(edge.caller, self._sanitize_mermaid_id(edge.caller)) # Fallback if alias not found
                callee_id_for_mermaid = fqn_to_alias.get(edge.callee, self._sanitize_mermaid_id(edge.callee)) # Fallback if alias not found

                edge_label = f'Line {edge.line_number}'

                edge_definitions.append(f'    {caller_id_for_mermaid} --> |{edge_label}| {callee_id_for_mermaid}')

        # Assemble the final Mermaid graph string
        result_lines = ["graph TD"]
        
        if llm_optimized and alias_mapping_comments:
            result_lines.extend(alias_mapping_comments) # Add alias definitions as comments
            result_lines.append("") # Add a blank line for readability before the graph

        result_lines.extend(node_definitions)
        result_lines.extend(edge_definitions)
        if style_commands:
            result_lines.extend(style_commands)

        return "\n".join(result_lines)

    def print_summary(self) -> None:
        """Print a human-readable summary of the call graph."""
        logging.info("\n📊 Call Graph Summary:")
        logging.info(f"   Functions: {len(self.functions)}")
        logging.info(f"   Methods: {sum(1 for f in self.functions.values() if f.is_method)}")
        logging.info(f"   Async functions: {sum(1 for f in self.functions.values() if f.is_async)}")
        logging.info(f"   Modules: {len(self.modules)}")
        logging.info(f"   Edges: {len(self.edges)}")
        logging.info(f"   Entry points: {len(self.get_entry_points())}")

        if self.edges and self.functions:
            avg_degree = len(self.edges) / len(self.functions)
            logging.info(f"   Average degree: {avg_degree:.1f}")

        logging.info("\n🏁 Top Entry Points:")
        entry_points = self.get_entry_points()
        for ep in entry_points[:5]:  # Show top 5
            logging.info(f"   • {ep.name} ({ep.fully_qualified_name})")
            logging.info(f"     Location: {ep.file_path}:{ep.line_start}")
            logging.info(f"     Type: {'Method' if ep.is_method else 'Function'}{' (async)' if ep.is_async else ''}")
            logging.info(f"     Class: {ep.class_name if ep.class_name else 'N/A'}")
            logging.info(f"     Connections: {len(ep.incoming_edges)} in, {len(ep.outgoing_edges)} out")
            if ep.docstring:
                logging.info(f"     Docstring: {ep.docstring[:80]}{'...' if len(ep.docstring) > 80 else ''}")
            logging.info("")
