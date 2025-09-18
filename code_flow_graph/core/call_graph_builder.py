"""
Builds a call graph from extracted code elements.
Uses explicit data structures to minimize cognitive load.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
import re
from pathlib import Path
import ast
from collections import defaultdict

from core.ast_extractor import CodeElement, FunctionElement

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
    parameters: List[str]
    incoming_edges: List[CallEdge]
    outgoing_edges: List[CallEdge]
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

    def __post_init__(self):
        if self.incoming_edges is None:
            self.incoming_edges = []
        if self.outgoing_edges is None:
            self.outgoing_edges = []

@dataclass
class ModuleNode:
    """Represents a module in the call graph."""
    name: str
    file_path: str
    functions: List[FunctionNode]
    imports: Dict[str, str]  # module -> imported items
    exports: List[str]
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

    def __init__(self):
        self.functions: Dict[str, FunctionNode] = {}
        self.modules: Dict[str, ModuleNode] = {}
        self.edges: List[CallEdge] = []
        self.functions_by_name: Dict[str, FunctionNode] = {}  # Local name to node mapping
        self.known_imports: Dict[str, Dict[str, str]] = {}  # module -> {local_name: fqn}

    def add_function(self, func_element: FunctionElement) -> FunctionNode:
        """Add a function to the graph with fully qualified naming and complete metadata."""
        # Create fully qualified name: module.function_name
        module_name = self._get_module_name(Path(func_element.file_path))
        fqn = f"{module_name}.{func_element.name}"

        node = FunctionNode(
            name=func_element.name,
            fully_qualified_name=fqn,
            file_path=str(func_element.file_path),
            line_start=func_element.line_start,
            parameters=func_element.parameters,
            return_type=func_element.return_type,
            is_async=func_element.is_async,
            is_static=func_element.is_static,
            access_modifier=func_element.access_modifier,
            docstring=func_element.docstring,
            is_method=func_element.is_method,
            class_name=func_element.class_name,
            incoming_edges=[],
            outgoing_edges=[]
        )

        self.functions[fqn] = node
        self.functions_by_name[func_element.name] = node  # Track by local name too

        # Add to module
        if module_name not in self.modules:
            self.modules[module_name] = ModuleNode(
                name=module_name,
                file_path=str(func_element.file_path),
                imports={},
                exports=[],
                functions=[]
            )
        self.modules[module_name].functions.append(node)

        print(f"   Added function: {fqn} {'(method)' if func_element.is_method else ''}{' (async)' if func_element.is_async else ''}")
        return node

    def build_from_elements(self, elements: List[CodeElement]) -> None:
        """Build the complete call graph from code elements."""
        print("Step 1: Creating function nodes...")

        # First pass: create all function nodes
        function_elements = [e for e in elements if isinstance(e, FunctionElement)]
        for element in function_elements:
            self.add_function(element)

        print(f"   Created {len(self.functions)} function nodes")
        print(f"   Methods: {sum(1 for f in self.functions.values() if f.is_method)}")
        print(f"   Async functions: {sum(1 for f in self.functions.values() if f.is_async)}")
        print(f"   Available functions: {list(self.functions_by_name.keys())[:10]}...")  # Show first 10

        print("Step 2: Extracting call edges...")
        # Second pass: find calls
        for element in function_elements:
            self._extract_calls_from_function(element)

        print(f"   Found {len(self.edges)} call edges")

        print("Step 3: Identifying entry points...")
        self._identify_entry_points()

    def _extract_calls_from_function(self, func_element: FunctionElement) -> None:
        """Extract all function calls from a single function's source using AST."""
        try:
            # Read the full source file
            with open(func_element.file_path, 'r', encoding='utf-8') as f:
                full_source = f.read()

            # Parse the full AST for this file
            tree = ast.parse(full_source, filename=func_element.file_path)

            # Find the specific function node
            func_node = self._find_function_node(tree, func_element.name)
            if not func_node:
                print(f"   Warning: Could not find AST node for function {func_element.name}")
                return

            # Extract calls from this function's body
            calls = self._extract_calls_from_ast_node(func_node, func_element)

            # Create edges for valid calls
            module_name = self._get_module_name(Path(func_element.file_path))
            caller_fqn = f"{module_name}.{func_element.name}"

            for call in calls:
                self._create_edge_if_valid(caller_fqn, call, func_element)

        except Exception as e:
            print(f"   Warning: Error extracting calls from {func_element.name}: {e}")
            # Fallback to regex-based extraction
            self._extract_calls_regex_fallback(func_element)

    def _find_function_node(self, tree: ast.AST, func_name: str) -> Optional[ast.FunctionDef]:
        """Find the specific function node in the AST tree."""
        class FunctionFinder(ast.NodeVisitor):
            def __init__(self, target_name: str):
                self.target_name = target_name
                self.found_node = None

            def visit_FunctionDef(self, node):
                if node.name == self.target_name:
                    self.found_node = node
                    # Don't continue visiting children to avoid finding nested functions
                    return
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                if node.name == self.target_name:
                    self.found_node = node
                    return
                self.generic_visit(node)

        finder = FunctionFinder(func_name)
        finder.visit(tree)
        return finder.found_node

    def _extract_calls_from_ast_node(self, node: ast.AST, func_element: FunctionElement) -> List[PotentialCall]:
        """Extract function calls from an AST node using proper AST traversal."""
        class CallExtractor(ast.NodeVisitor):
            def __init__(self, parent_func_element: FunctionElement):
                self.calls = []
                self.parent_file = parent_func_element.file_path
                self.start_line = parent_func_element.line_start
                self.end_line = parent_func_element.line_end

            def visit_Call(self, node: ast.Call):
                # Only consider calls within the function body
                if (hasattr(node, 'lineno') and
                    self.start_line <= node.lineno <= self.end_line):

                    callee_name = self._get_callee_name(node.func)
                    line_number = getattr(node, 'lineno', 1)
                    if callee_name and self._is_valid_call(callee_name):
                        # Get line context
                        try:
                            with open(self.parent_file, 'r') as f:
                                lines = f.readlines()
                            line_number = getattr(node, 'lineno', 1)
                            context = lines[line_number-1].strip() if line_number <= len(lines) else ""
                        except Exception as e:
                            context = f"Error: {e}"

                        # Extract parameter names (simplified)
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
                """Extract the function name from a Call's func attribute."""
                if isinstance(func, ast.Name):
                    return func.id
                elif isinstance(func, ast.Attribute):
                    # Handle method calls like obj.method()
                    if isinstance(func.value, ast.Name):
                        return f"{func.value.id}.{func.attr}"
                    return func.attr
                elif isinstance(func, ast.NameConstant) or isinstance(func, ast.Str):
                    # Ignore string literals, constants, etc.
                    return None
                return None

            def _is_valid_call(self, name: str) -> bool:
                """Filter out non-function call identifiers."""
                invalid_keywords = {
                    'if', 'for', 'while', 'with', 'try', 'except', 'class',
                    'def', 'import', 'from', 'return', 'yield', 'in', 'is', 'not',
                    'and', 'or', 'True', 'False', 'None', 'self', 'cls'
                }
                return name not in invalid_keywords and not name.startswith('_')

            def _extract_call_parameters(self, node: ast.Call) -> List[str]:
                """Extract parameter names from a call (simplified)."""
                params = []
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        params.append(arg.id)
                    elif isinstance(arg, ast.Constant):
                        params.append(repr(arg.value))
                    else:
                        params.append(str(type(arg).__name__))
                return params[:3]  # Limit to first 3 for brevity

        extractor = CallExtractor(func_element)
        extractor.visit(node)

        print(f"     Found {len(extractor.calls)} potential calls in {func_element.name}")
        return extractor.calls

    def _create_edge_if_valid(self, caller_fqn: str, call: PotentialCall,
                            func_element: FunctionElement) -> None:
        """Create a call edge if the target function exists."""
        # First try to match by fully qualified name
        target_fqn = None
        target_node = None

        # Strategy 1: Check if it's a local function call
        if call.callee in self.functions_by_name:
            target_node = self.functions_by_name[call.callee]
            target_fqn = target_node.fully_qualified_name
            print(f"     Local call: {caller_fqn} -> {target_fqn}")

        # Strategy 2: If it's a method call (e.g., "obj.method"), try to resolve
        elif '.' in call.callee:
            parts = call.callee.split('.')
            if len(parts) == 2 and parts[1] in self.functions_by_name:
                # This is a simplified check - in reality we'd need more context
                target_node = self.functions_by_name[parts[1]]
                target_fqn = target_node.fully_qualified_name
                print(f"     Method call: {caller_fqn} -> {target_fqn}")

        # Strategy 3: Check imports (simplified - would need proper import resolution)
        elif call.callee in self.known_imports.get(self._get_module_name(Path(func_element.file_path)), {}):
            imported_fqn = self.known_imports[self._get_module_name(Path(func_element.file_path))][call.callee]
            if imported_fqn in self.functions:
                target_node = self.functions[imported_fqn]
                target_fqn = imported_fqn
                print(f"     Imported call: {caller_fqn} -> {target_fqn}")

        if target_fqn and target_node:
            edge = CallEdge(
                caller=caller_fqn,
                callee=target_fqn,
                file_path=call.file_path,
                line_number=call.line_number,
                call_type='direct',
                parameters=call.parameters,
                is_static_call=True,
                confidence=0.9
            )

            if target_fqn not in self.functions:
                print(f"     Warning: Target function {target_fqn} not in graph")
                return

            self.edges.append(edge)
            self.functions[caller_fqn].outgoing_edges.append(edge)
            self.functions[target_fqn].incoming_edges.append(edge)
            print(f"     ✓ Created edge: {caller_fqn} -> {target_fqn}")

    def _extract_calls_regex_fallback(self, func_element: FunctionElement) -> None:
        """Fallback regex-based call extraction if AST fails."""
        print(f"     Using regex fallback for {func_element.name}")

        # Read the source code around this function
        try:
            with open(func_element.file_path, 'r') as f:
                lines = f.readlines()

            # Extract the function body lines (approximate)
            func_lines = lines[func_element.line_start-1:func_element.line_end]
            func_source = ''.join(func_lines)

            # Find potential function calls using regex patterns
            calls = self._find_function_calls_regex(func_source, func_element.file_path,
                                                  func_element.line_start)

            # Create edges for valid calls
            module_name = self._get_module_name(Path(func_element.file_path))
            caller_fqn = f"{module_name}.{func_element.name}"

            for call in calls:
                self._create_edge_if_valid_regex(caller_fqn, call, func_element)

        except Exception as e:
            print(f"     Warning: Regex fallback failed for {func_element.name}: {e}")

    def _find_function_calls_regex(self, source: str, file_path: str,
                                 start_line: int) -> List[PotentialCall]:
        """Find function calls using regex as fallback."""
        calls = []
        lines = source.split('\n')

        # Improved call pattern that avoids keywords
        call_pattern = r'\b(?!if|for|while|with|try|except|class|def|import|from|return|yield)\w+\s*\('

        for i, line in enumerate(lines, start_line):
            matches = re.finditer(call_pattern, line)
            for match in matches:
                func_name = match.group(0).strip('(').strip()

                # Additional filtering
                if (len(func_name) > 1 and
                    func_name not in {'in', 'is', 'or', 'and', 'not', 'True', 'False', 'None'} and
                    not func_name.startswith('_')):

                    # Extract some context about parameters
                    param_match = re.search(r'\((\s*[^)]+)?\)', line[match.start():])
                    params = []
                    if param_match:
                        param_text = param_match.group(1)
                        # Simple parameter extraction
                        param_names = re.findall(r'\b\w+\b', param_text)
                        params = param_names[:3]  # First 3 parameter names
                        if not params:  # If no names, indicate it's a call
                            params = ['...']

                    call = PotentialCall(
                        callee=func_name,
                        file_path=str(file_path),
                        line_number=i,
                        parameters=params,
                        context=line.strip()
                    )
                    calls.append(call)

        return calls

    def _create_edge_if_valid_regex(self, caller_fqn: str, call: PotentialCall,
                                  func_element: FunctionElement) -> None:
        """Create edge from regex-based call if valid."""
        # Same validation logic as AST-based extraction
        if call.callee in self.functions_by_name:
            target_node = self.functions_by_name[call.callee]
            target_fqn = target_node.fully_qualified_name

            edge = CallEdge(
                caller=caller_fqn,
                callee=target_fqn,
                file_path=call.file_path,
                line_number=call.line_number,
                call_type='direct',
                parameters=call.parameters,
                is_static_call=True,
                confidence=0.7  # Lower confidence for regex
            )

            if target_fqn not in self.functions:
                print(f"     Warning: Target function {target_fqn} not in graph")
                return

            self.edges.append(edge)
            self.functions[caller_fqn].outgoing_edges.append(edge)
            self.functions[target_fqn].incoming_edges.append(edge)
            print(f"     ✓ Regex edge: {caller_fqn} -> {target_fqn}")

    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            # Remove .py extension and convert path separators to dots
            relative_path = file_path.relative_to(Path.cwd())
            module_path = str(relative_path).replace('.py', '').replace('/', '.').replace('\\', '.')
            return module_path
        except ValueError:
            # If not relative to cwd, use absolute path
            module_path = str(file_path).replace('.py', '').replace('/', '.').replace('\\', '.')
            return module_path

    def _identify_entry_points(self) -> None:
        """Identify entry points using multiple strategies."""
        entry_point_patterns = {
            'python': ['main', 'run', '__main__', 'app', 'server', 'start', 'init', 'create'],
            'typescript': ['main', 'run', 'start', 'app', 'server', 'bootstrap']
        }

        python_patterns = entry_point_patterns['python']

        print("   Identifying entry points using multiple strategies...")
        entry_point_count = 0

        # Strategy 1: Common entry point names
        for fqn, func in self.functions.items():
            func_name = func.name.lower()

            if any(pattern in func_name for pattern in python_patterns):
                func.is_entry_point = True
                entry_point_count += 1
                print(f"     ✓ Entry point (name): {func.name} ({fqn})")

            # Strategy 2: Functions with no incoming edges (likely entry points), but not methods
            if (len(func.incoming_edges) == 0 and
                not func.is_method and
                not func.is_entry_point):  # Avoid double-marking
                func.is_entry_point = True
                entry_point_count += 1
                print(f"     ✓ Entry point (no incoming): {func.name} ({fqn})")

            # Strategy 3: Functions in __main__ blocks or main modules
            if ('main' in func.file_path.lower() or
                func.name == '__main__' or
                'main' in fqn.lower()) and not func.is_entry_point:
                func.is_entry_point = True
                entry_point_count += 1
                print(f"     ✓ Entry point (location): {func.name} ({fqn})")

        # Strategy 4: Mark exported/public functions
        for fqn, func in self.functions.items():
            func_name = func.name.lower()
            if (func_name.startswith('public_') or
                'export' in func_name or
                func.access_modifier == 'public') and not func.is_entry_point:
                func.is_entry_point = True
                func.is_exported = True
                entry_point_count += 1
                print(f"     ✓ Entry point (exported): {func.name} ({fqn})")

        # Strategy 5: Module-level entry points
        for module_name, module in self.modules.items():
            if any(ep in module_name.lower() for ep in ['main', 'app', 'server', '__main__']):
                module.is_entry_point = True
                for func in module.functions:
                    if func.name in ['main', 'run', 'start', 'init'] and not func.is_entry_point:
                        func.is_entry_point = True
                        entry_point_count += 1
                        print(f"     ✓ Entry point (module): {func.name} ({func.fully_qualified_name})")

        # Strategy 6: Top-level functions in small modules (likely entry points)
        for module_name, module in self.modules.items():
            if len(module.functions) <= 3:  # Small modules
                for func in module.functions:
                    if (not func.is_method and
                        func.name not in ['__init__', '__post_init__'] and
                        not func.is_entry_point):
                        func.is_entry_point = True
                        entry_point_count += 1
                        print(f"     ✓ Entry point (small module): {func.name} ({func.fully_qualified_name})")

        entry_points = self.get_entry_points()
        print(f"   Total entry points identified: {len(entry_points)}")

    def get_entry_points(self) -> List[FunctionNode]:
        """Get all identified entry points."""
        return [f for f in self.functions.values() if f.is_entry_point]

    def export_graph(self, format: str = 'json') -> Optional[Dict]:
        """Export the call graph in various formats."""
        if not self.functions:
            print("Warning: No functions in graph to export")
            return None

        if format == 'json':
            graph_data = {
                'functions': {
                    f.fully_qualified_name: {
                        'name': f.name,
                        'file_path': f.file_path,
                        'line_start': f.line_start,
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
                        'access_modifier': f.access_modifier or 'public'
                    }
                    for f in self.functions.values()
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
                    for edge in self.edges
                ],
                'entry_points': [f.fully_qualified_name for f in self.get_entry_points()],
                'summary': {
                    'total_functions': len(self.functions),
                    'total_methods': sum(1 for f in self.functions.values() if f.is_method),
                    'total_async': sum(1 for f in self.functions.values() if f.is_async),
                    'total_edges': len(self.edges),
                    'total_modules': len(self.modules),
                    'entry_points_count': len(self.get_entry_points()),
                    'avg_degree': round(len(self.edges) / len(self.functions), 2) if self.functions else 0
                }
            }
            return graph_data
        return None

    def print_summary(self) -> None:
        """Print a human-readable summary of the call graph."""
        print("\n📊 Call Graph Summary:")
        print(f"   Functions: {len(self.functions)}")
        print(f"   Methods: {sum(1 for f in self.functions.values() if f.is_method)}")
        print(f"   Async functions: {sum(1 for f in self.functions.values() if f.is_async)}")
        print(f"   Modules: {len(self.modules)}")
        print(f"   Edges: {len(self.edges)}")
        print(f"   Entry points: {len(self.get_entry_points())}")

        if self.edges and self.functions:
            avg_degree = len(self.edges) / len(self.functions)
            print(f"   Average degree: {avg_degree:.1f}")

        print("\n🏁 Top Entry Points:")
        entry_points = self.get_entry_points()
        for ep in entry_points[:5]:  # Show top 5
            print(f"   • {ep.name} ({ep.fully_qualified_name})")
            print(f"     Location: {ep.file_path}:{ep.line_start}")
            print(f"     Type: {'Method' if ep.is_method else 'Function'}{' (async)' if ep.is_async else ''}")
            print(f"     Class: {ep.class_name if ep.class_name else 'N/A'}")
            print(f"     Connections: {len(ep.incoming_edges)} in, {len(ep.outgoing_edges)} out")
            if ep.docstring:
                print(f"     Docstring: {ep.docstring[:80]}{'...' if len(ep.docstring) > 80 else ''}")
            print()
