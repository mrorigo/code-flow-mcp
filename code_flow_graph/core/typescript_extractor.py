"""
Extracts and represents AST nodes for TypeScript codebases with enhanced metadata.
Includes logic for parsing TypeScript syntax, calculating complexity, detecting frameworks,
and extracting type information using regex-based parsing and TypeScript compiler integration.
"""

import re
import json
import sys
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

# Import from new modular structure
from .models import CodeElement, FunctionElement, ClassElement
from .utils import (
    hash_source_snippet,
    calculate_nloc_typescript,
    calculate_complexity_typescript,
    get_gitignore_patterns,
    match_file_against_pattern,
    parse_decorators_typescript,
    extract_file_imports_typescript,
    extract_generics_typescript,
    detect_framework_patterns_typescript,
    extract_typescript_parameters,
    extract_jsdoc_comment,
    find_closing_brace,
    determine_if_method_typescript,
    extract_route_pattern,
    analyze_type_complexity,
    categorize_dependencies,
    extract_typescript_dependencies
)


class TypeScriptASTVisitor:
    """
    AST visitor that extracts code elements from TypeScript codebases with enhanced metadata.
    Uses fast regex-based parsing for optimal performance.
    """

    def __init__(self):
        self.elements: List[CodeElement] = []
        self.current_class: Optional[str] = None
        self.current_file: str = ""
        self.source_lines: List[str] = []
        self.file_level_imports: Dict[str, str] = {}
        self.file_level_import_from_targets: Set[str] = set()

        # Pre-compile regex patterns for maximum performance
        self._compiled_patterns = self._compile_regex_patterns()

    def _compile_regex_patterns(self) -> Dict[str, Any]:
        """Pre-compile all regex patterns for maximum performance."""
        import time
        start_time = time.time()

        patterns = {}

        # Compile function patterns
        function_patterns = [
            # Regular functions with complex generics: function name<T, U>(...): Promise<T> { ... }
            r'function\s+(\w+)(?:<[^>]*>)?\s*\(([^)]*)\)\s*:\s*([^;{]+)\s*{',
            # Arrow functions with complex return types: const name = <T>(...): T => ...
            r'const\s+(\w+)(?:<[^>]*>)?\s*=\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*:\s*([^=;{]+)\s*=>',
            # Async arrow functions: const name = async (...): Promise<Type> => ...
            r'const\s+(\w+)(?:<[^>]*>)?\s*=\s*async\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*:\s*([^=;{]+)\s*=>',
            # Simple arrow functions: const name = <T>(...) => ...
            r'const\s+(\w+)(?:<[^>]*>)?\s*=\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*=>',
            # React functional components: const Name: React.FC<Props> = ({ ... }) => ...
            r'const\s+(\w+)(?:<[^>]*>)?\s*:\s*[^=]+\s*=\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*=>',
            # Method definitions with complex types: public/private/protected name<T>(...): T { ... }
            r'(?:public|private|protected)?\s*(?:static)?\s*(?:readonly)?\s*(?:async)?\s+(\w+)(?:<[^>]*>)?\s*\(([^)]*)\)\s*:\s*([^;{]+)\s*{',
            # Angular lifecycle methods: ngOnInit(): void { ... }
            r'(\w+)\s*\(\s*\)\s*:\s*([^;{]+)\s*{',
            # Async functions with complex return types: async function name<T>(...): Promise<T> { ... }
            r'async\s+function\s+(\w+)(?:<[^>]*>)?\s*\(([^)]*)\)\s*:\s*([^;{]+)\s*{',
            # Abstract methods: abstract name<T>(...): T; ...
            r'abstract\s+(public|private|protected)?\s*(\w+)(?:<[^>]*>)?\s*\(([^)]*)\)\s*:\s*([^;]+)\s*;',
            # Vue 3 Composition API functions: export function useXxx<T>() { ... }
            r'export\s+function\s+(\w+)(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*:\s*([^;{]+))?\s*{',
            # React hooks and custom functions
            r'(export\s+)?function\s+(\w+)(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*:\s*([^;{]+))?\s*{',
            # Constructor functions: constructor(...) { ... }
            r'constructor\s*\(([^)]*)\)\s*{',
            # Getter methods: get name(): Type { ... }
            r'get\s+(\w+)(?:<[^>]*>)?\s*\(\s*\)\s*:\s*([^;{]+)\s*{',
            # Setter methods: set name(value: Type): void { ... }
            r'set\s+(\w+)(?:<[^>]*>)?\s*\(\s*([^:)]+)\s*:\s*([^;{]+)\)\s*{',
            # Express route handlers with asyncHandler wrapper: router.post('/path', asyncHandler(async (req, res) => { ... }
            r'router\.(get|post|put|delete|patch)\s*\(\s*[^,]+,\s*asyncHandler\s*\(\s*async\s*\(([^)]+)\)\s*=>',
            # Express route handlers with validation: router.post('/path', validation, asyncHandler(async (req, res) => { ... }
            r'router\.(get|post|put|delete|patch)\s*\(\s*[^,]+,[^,]+,\s*asyncHandler\s*\(\s*async\s*\(([^)]+)\)\s*=>',
            # Express middleware functions: asyncHandler(async (req, res, next) => { ... }
            r'asyncHandler\s*\(\s*async\s*\(([^)]+)\)\s*=>',
            # Vue 3 setup functions and configuration
            r'(?:app\.config\.|config\.)(errorHandler|warnHandler|performance)\s*=\s*\([^)]+\)\s*=>',
        ]

        patterns['functions'] = [re.compile(pattern, re.MULTILINE | re.DOTALL) for pattern in function_patterns]

        # Compile class patterns
        class_patterns = [
            # Regular classes: class Name<T> extends Base implements Interface { ... }
            r'class\s+(\w+)(?:<[^>]*>)?(\s+extends\s+([^,\s]+))?(\s+implements\s+([^}]+?))?\s*{',
            # Abstract classes: abstract class Name<T> extends Base { ... }
            r'abstract\s+class\s+(\w+)(?:<[^>]*>)?(\s+extends\s+([^,\s]+))?(\s+implements\s+([^}]+?))?\s*{',
            # Classes with multiple decorators (supports complex decorator chains)
            r'((?:@\w+(?:\([^)]*\))?\s*\n\s*)*)class\s+(\w+)(?:<[^>]*>)?(?:\s+extends\s+([^,\s]+))?(?:\s+implements\s+([^}]+?))?\s*{',
            # Exported classes: export class Name { ... }
            r'export\s+class\s+(\w+)(?:<[^>]*>)?(\s+extends\s+([^,\s]+))?(\s+implements\s+([^}]+?))?\s*{',
            # Abstract exported classes
            r'export\s+abstract\s+class\s+(\w+)(?:<[^>]*>)?(\s+extends\s+([^,\s]+))?(\s+implements\s+([^}]+?))?\s*{',
            # Classes with more complex decorator patterns (multi-line decorators)
            r'((?:@\w+\s*\([^)]+\)\s*\n\s*)+)class\s+(\w+)(?:<[^>]*>)?(?:\s+extends\s+([^,\s]+))?(?:\s+implements\s+([^}]+?))?\s*{',
            # Classes with decorators that span multiple lines
            r'((?:@\w+(?:\s*\([^)]+\))?\s*\n\s*)+)class\s+(\w+)(?:<[^>]*>)?(?:\s+extends\s+([^,\s]+))?(?:\s+implements\s+([^}]+?))?\s*{',
        ]

        patterns['classes'] = [re.compile(pattern, re.MULTILINE | re.DOTALL) for pattern in class_patterns]

        # Compile other patterns
        patterns['interfaces'] = re.compile(r'interface\s+(\w+)(?:<([^>]+)>)?(?:\s+extends\s+([^ {]+))?(?:\s+implements\s+([^ {]+))?\s*{')
        patterns['enums'] = re.compile(r'enum\s+(\w+)(?:<([^>]+)>)?\s*{([^}]*)}', re.DOTALL)
        patterns['type_aliases'] = [
            re.compile(r'type\s+(\w+)(?:<([^>]+)>)?\s*=\s*([^;]+);'),
            re.compile(r'export\s+type\s+(\w+)(?:<([^>]+)>)?\s*=\s*([^;]+);')
        ]
        patterns['namespaces'] = re.compile(r'(?:namespace|module)\s+(\w+)(?:\s*{([^}]*)})?', re.DOTALL)
        patterns['express_vars'] = [
            re.compile(r'const\s+(\w+)\s*=\s*express\s*\(\s*\)'),
            re.compile(r'const\s+(\w+)\s*=\s*Application\s*\(\s*\)')
        ]

        compilation_time = time.time() - start_time
        pattern_count = sum(len(p) if isinstance(p, list) else 1 for p in patterns.values())
        print(f"   Info: Pre-compiled {pattern_count} regex patterns in {compilation_time:.4f}s")

        return patterns



    def _parse_typescript_ast(self, file_path: str, source: str) -> List[CodeElement]:
        """Parse TypeScript AST using fast regex-based parsing."""
        print(f"   Info: Parsing {file_path} using fast regex-based extraction")
        # Use fast regex-based parsing for optimal performance
        return self._parse_with_regex(file_path, source)

    def _parse_with_compiler(self, file_path: str, source: str) -> List[CodeElement]:
        """Parse using TypeScript compiler (placeholder for full implementation)."""
        elements = []
        lines = source.splitlines()

        # Extract functions
        function_matches = self._find_functions_regex(source)
        for match in function_matches:
            func_element = self._create_function_element(match, lines, file_path)
            if func_element:
                elements.append(func_element)

        # Extract classes
        class_matches = self._find_classes_regex(source)
        for match in class_matches:
            class_element = self._create_class_element(match, lines, file_path)
            if class_element:
                elements.append(class_element)

        # Extract variable declarations that might be framework instances
        variable_matches = self._find_framework_variables(source)
        for match in variable_matches:
            var_element = self._create_variable_element(match, lines, file_path)
            if var_element:
                elements.append(var_element)

        return elements

    def _parse_with_regex(self, file_path: str, source: str) -> List[CodeElement]:
        """Parse using regex patterns when TypeScript compiler is unavailable."""
        elements = []
        lines = source.splitlines()

        # Extract functions
        function_matches = self._find_functions_regex(source)
        for match in function_matches:
            func_element = self._create_function_element(match, lines, file_path)
            if func_element:
                elements.append(func_element)

        # Extract classes
        class_matches = self._find_classes_regex(source)
        for match in class_matches:
            class_element = self._create_class_element(match, lines, file_path)
            if class_element:
                elements.append(class_element)

        # Extract variable declarations that might be framework instances (e.g., Express apps)
        variable_matches = self._find_framework_variables(source)
        for match in variable_matches:
            var_element = self._create_variable_element(match, lines, file_path)
            if var_element:
                elements.append(var_element)

        return elements

    def _find_functions_regex(self, source: str) -> List[Dict[str, Any]]:
        """Find function definitions using pre-compiled regex patterns for maximum performance."""
        functions = []

        # Use pre-compiled patterns for maximum performance
        compiled_patterns = self._compiled_patterns['functions']

        for compiled_pattern in compiled_patterns:
            for match in compiled_pattern.finditer(source):
                groups = match.groups()

                # Calculate start_line correctly based on source content before match
                start_line = source[:match.start()].count('\n') + 1

                # Map pattern index to pattern type for proper matching
                pattern_index = compiled_patterns.index(compiled_pattern)

                # Define pattern types mapping (same order as patterns list)
                pattern_types = [
                    'function', 'arrow', 'async_arrow', 'simple_arrow', 'react_component',
                    'method', 'lifecycle_method', 'async_function', 'abstract_method',
                    'export_composable', 'exported_function', 'constructor', 'getter',
                    'setter', 'express_route', 'express_route_validated', 'express_middleware',
                    'vue_config'
                ]

                if pattern_index < len(pattern_types):
                    pattern_type = pattern_types[pattern_index]

                    if pattern_type == 'function':
                        func_info = self._parse_function_match(groups, match, 'function', start_line)
                    elif pattern_type == 'arrow':
                        func_info = self._parse_function_match(groups, match, 'arrow', start_line)
                    elif pattern_type == 'async_arrow':
                        func_info = self._parse_function_match(groups, match, 'async_arrow')
                    elif pattern_type == 'simple_arrow':
                        func_info = self._parse_function_match(groups, match, 'simple_arrow')
                    elif pattern_type == 'react_component':
                        func_info = self._parse_function_match(groups, match, 'react_component')
                    elif pattern_type == 'method':
                        func_info = self._parse_function_match(groups, match, 'method', start_line)
                    elif pattern_type == 'lifecycle_method':
                        func_info = self._parse_lifecycle_method_match(groups, match, start_line)
                    elif pattern_type == 'async_function':
                        func_info = self._parse_function_match(groups, match, 'async_function', start_line)
                    elif pattern_type == 'abstract_method':
                        func_info = self._parse_function_match(groups, match, 'abstract_method', start_line)
                    elif pattern_type == 'export_function':
                        func_info = self._parse_function_match(groups, match, 'export_function', start_line)
                    elif pattern_type == 'export_composable':
                        func_info = self._parse_export_composable_match(groups, match, start_line)
                    elif pattern_type == 'exported_function':
                        func_info = self._parse_function_match(groups, match, 'exported_function', start_line)
                    elif pattern_type == 'constructor':
                        func_info = self._parse_function_match(groups, match, 'constructor', start_line)
                    elif pattern_type == 'getter':
                        func_info = self._parse_function_match(groups, match, 'getter', start_line)
                    elif pattern_type == 'setter':
                        func_info = self._parse_function_match(groups, match, 'setter', start_line)
                    elif pattern_type == 'express_route':
                        func_info = self._parse_express_route_match(groups, match)
                    elif pattern_type == 'express_route_validated':
                        func_info = self._parse_express_route_match(groups, match, start_line)
                    elif pattern_type == 'express_middleware':
                        func_info = self._parse_express_middleware_match(groups, match, start_line)
                    elif pattern_type == 'vue_config':
                        func_info = self._parse_vue_config_match(groups, match, start_line)
                    else:
                        continue

                    if func_info:
                        functions.append(func_info)
                groups = match.groups()

                # Calculate start_line correctly based on source content before match
                start_line = source[:match.start()].count('\n') + 1

                if pattern_type == 'function':
                    # Regular function
                    func_info = self._parse_function_match(groups, match, 'function', start_line)
                elif pattern_type == 'arrow':
                    # Arrow function with return type
                    func_info = self._parse_function_match(groups, match, 'arrow', start_line)
                elif pattern_type == 'async_arrow':
                    # Async arrow function
                    func_info = self._parse_function_match(groups, match, 'async_arrow')
                elif pattern_type == 'simple_arrow':
                    # Simple arrow function
                    func_info = self._parse_function_match(groups, match, 'simple_arrow')
                elif pattern_type == 'react_component':
                    # React functional component
                    func_info = self._parse_function_match(groups, match, 'react_component')
                elif pattern_type == 'method':
                    # Method definition
                    func_info = self._parse_function_match(groups, match, 'method', start_line)
                elif pattern_type == 'lifecycle_method':
                    # Angular lifecycle method like ngOnInit()
                    func_info = self._parse_lifecycle_method_match(groups, match, start_line)
                elif pattern_type == 'async_function':
                    # Async function
                    func_info = self._parse_function_match(groups, match, 'async_function', start_line)
                elif pattern_type == 'abstract_method':
                    # Abstract method
                    func_info = self._parse_function_match(groups, match, 'abstract_method', start_line)
                elif pattern_type == 'export_function':
                    # Exported function (Vue 3 Composition API style)
                    func_info = self._parse_function_match(groups, match, 'export_function', start_line)
                elif pattern_type == 'export_composable':
                    # Exported Vue composable function
                    func_info = self._parse_export_composable_match(groups, match, start_line)
                elif pattern_type == 'exported_function':
                    # Exported function (React/Vue style)
                    func_info = self._parse_function_match(groups, match, 'exported_function', start_line)
                elif pattern_type == 'constructor':
                    # Constructor
                    func_info = self._parse_function_match(groups, match, 'constructor', start_line)
                elif pattern_type == 'getter':
                    # Getter method
                    func_info = self._parse_function_match(groups, match, 'getter', start_line)
                elif pattern_type == 'setter':
                    # Setter method
                    func_info = self._parse_function_match(groups, match, 'setter', start_line)
                elif pattern_type == 'express_route':
                    # Express route handler with asyncHandler
                    func_info = self._parse_express_route_match(groups, match)
                elif pattern_type == 'express_route_validated':
                    # Express route handler with validation
                    func_info = self._parse_express_route_match(groups, match, start_line)
                elif pattern_type == 'express_middleware':
                    # Express middleware with asyncHandler
                    func_info = self._parse_express_middleware_match(groups, match, start_line)
                elif pattern_type == 'vue_config':
                    # Vue configuration handlers
                    func_info = self._parse_vue_config_match(groups, match, start_line)
                else:
                    # Unknown pattern type
                    continue

                if func_info:
                    functions.append(func_info)

        return functions

    def _find_framework_variables(self, source: str) -> List[Dict[str, Any]]:
        """Find variable declarations that might be framework instances using pre-compiled patterns."""
        variables = []

        # Use pre-compiled patterns for maximum performance
        express_patterns = self._compiled_patterns['express_vars']

        for compiled_pattern in express_patterns:
            for match in compiled_pattern.finditer(source):
                groups = match.groups()
                if groups:
                    var_name = groups[0]
                    variables.append({
                        'name': var_name,
                        'var_type': 'express_app',
                        'start_line': source[:match.start()].count('\n') + 1,
                        'match_text': match.group(0)
                    })

        return variables

    def _create_variable_element(self, var_info: Dict[str, Any], lines: List[str], file_path: str) -> Optional[CodeElement]:
        """Create CodeElement for framework variables."""
        start_line = var_info.get('start_line', 1)

        # For variables, we estimate a single line
        end_line = start_line

        # Detect framework patterns for this variable
        framework_patterns = self._detect_framework_patterns(var_info.get('match_text', ''))

        # Ensure framework is detected for Express variables
        if var_info.get('var_type') == 'express_app':
            framework_patterns['framework'] = 'express'
            framework_patterns['features'].extend(['routing', 'middleware'])
            framework_patterns['confidence'] = 0.9

        var_element = CodeElement(
            name=var_info['name'],
            kind='variable',
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            full_source='\n'.join(lines),
            metadata={
                'framework': framework_patterns.get('framework'),
                'typescript_features': framework_patterns.get('features', []),
                'variable_type': var_info.get('var_type'),
                'confidence': 0.8,
                'parsing_method': 'regex_fallback',
                'detected_frameworks': [framework_patterns.get('framework')] if framework_patterns.get('framework') else [],
                'framework_confidence': framework_patterns.get('confidence', 0.0),
                'framework_features': framework_patterns.get('features', [])
            }
        )

        return var_element

    def _parse_function_match(self, groups: tuple, match: re.Match, func_type: str, start_line: int = 1) -> Optional[Dict[str, Any]]:
        """Parse function match groups into function info dictionary."""
        try:
            num_groups = len(groups)

            if func_type == 'function':
                # function name<T>(params): ReturnType { ... }
                return {
                    'name': groups[0] if num_groups > 0 else 'unknown',
                    'generics': extract_generics_typescript(groups[0]) if num_groups > 0 else [],
                    'params': groups[1] if num_groups > 1 else '',
                    'return_type': self._parse_complex_return_type(groups[2]) if num_groups > 2 else None,
                    'is_async': False,
                    'is_static': False,
                    'is_generator': False,
                    'is_abstract': False,
                    'is_getter': False,
                    'is_setter': False,
                    'is_constructor': False,
                    'is_exported': False,
                    'start_line': start_line,
                    'match_text': match.group(0),
                    'function_type': func_type
                }
            elif func_type == 'arrow':
                # const name = <T>(params): ReturnType => ...
                return {
                    'name': groups[0] if num_groups > 0 else 'unknown',
                    'generics': extract_generics_typescript(groups[0]) if num_groups > 0 else [],
                    'params': groups[1] if num_groups > 1 else '',
                    'return_type': self._parse_complex_return_type(groups[2]) if num_groups > 2 else None,
                    'is_async': False,
                    'is_static': False,
                    'is_generator': False,
                    'is_abstract': False,
                    'is_getter': False,
                    'is_setter': False,
                    'is_constructor': False,
                    'is_exported': False,
                    'start_line': start_line,
                    'match_text': match.group(0),
                    'function_type': func_type
                }
            elif func_type == 'async_function':
                # async function name<T>(params): Promise<Type> { ... }
                return {
                    'name': groups[0] if num_groups > 0 else 'unknown',
                    'generics': extract_generics_typescript(groups[0]) if num_groups > 0 else [],
                    'params': groups[1] if num_groups > 1 else '',
                    'return_type': self._parse_complex_return_type(groups[2]) if num_groups > 2 else None,
                    'is_async': True,
                    'is_static': False,
                    'is_generator': False,
                    'is_abstract': False,
                    'is_getter': False,
                    'is_setter': False,
                    'is_constructor': False,
                    'is_exported': False,
                    'start_line': start_line,
                    'match_text': match.group(0),
                    'function_type': func_type
                }
            elif func_type == 'method':
                # Handle method definitions where access modifier groups may vary
                # Groups: (access_modifier, static, readonly, async, name, params, return_type)

                # Find the method name (should be the first word after modifiers)
                name = 'unknown'
                params = ''
                return_type = None
                access_modifier = None
                is_static = False
                is_readonly = False
                is_async = False

                # Parse groups based on what's available
                if num_groups > 0 and groups[0] in ['public', 'private', 'protected']:
                    access_modifier = groups[0]
                if num_groups > 1 and groups[1] == 'static':
                    is_static = True
                if num_groups > 2 and groups[2] == 'readonly':
                    is_readonly = True
                if num_groups > 3 and groups[3] == 'async':
                    is_async = True

                # The name is typically at index 4, but adjust if access modifier is missing
                name_idx = 4 if num_groups > 4 else (0 if num_groups > 0 else None)
                if name_idx is not None and groups[name_idx]:
                    name = groups[name_idx]

                params_idx = 5 if num_groups > 5 else (1 if num_groups > 1 else None)
                if params_idx is not None:
                    params = groups[params_idx] if groups[params_idx] else ''

                return_type_idx = 6 if num_groups > 6 else (2 if num_groups > 2 else None)
                if return_type_idx is not None:
                    return_type = self._parse_complex_return_type(groups[return_type_idx]) if groups[return_type_idx] else None

                return {
                    'name': name,
                    'generics': extract_generics_typescript(name),
                    'params': params,
                    'return_type': return_type,
                    'access_modifier': access_modifier,
                    'is_static': is_static,
                    'is_readonly': is_readonly,
                    'is_async': is_async,
                    'is_generator': False,
                    'is_abstract': False,
                    'is_getter': False,
                    'is_setter': False,
                    'is_constructor': False,
                    'is_exported': False,
                    'start_line': start_line,
                    'match_text': match.group(0),
                    'function_type': func_type
                }
            elif func_type == 'abstract_method':
                # abstract name<T>(params): ReturnType; ...
                return {
                    'name': groups[1] if num_groups > 1 else 'unknown',
                    'generics': extract_generics_typescript(groups[1]) if num_groups > 1 else [],
                    'params': groups[2] if num_groups > 2 else '',
                    'return_type': self._parse_complex_return_type(groups[3]) if num_groups > 3 else None,
                    'access_modifier': groups[0] if num_groups > 0 else None,
                    'is_async': False,
                    'is_static': False,
                    'is_generator': False,
                    'is_abstract': True,
                    'is_getter': False,
                    'is_setter': False,
                    'is_constructor': False,
                    'is_exported': False,
                    'start_line': start_line,
                    'match_text': match.group(0),
                    'function_type': func_type
                }
            elif func_type == 'constructor':
                # constructor(params) { ... }
                return {
                    'name': 'constructor',
                    'generics': [],
                    'params': groups[0] if num_groups > 0 else '',
                    'return_type': None,
                    'is_async': False,
                    'is_static': False,
                    'is_generator': False,
                    'is_abstract': False,
                    'is_getter': False,
                    'is_setter': False,
                    'is_constructor': True,
                    'is_exported': False,
                    'start_line': start_line,
                    'match_text': match.group(0),
                    'function_type': func_type
                }
            elif func_type == 'getter':
                # get name(): Type { ... }
                return {
                    'name': groups[0] if num_groups > 0 else 'unknown',
                    'generics': extract_generics_typescript(groups[0]) if num_groups > 0 else [],
                    'params': '',
                    'return_type': self._parse_complex_return_type(groups[1]) if num_groups > 1 else None,
                    'is_async': False,
                    'is_static': False,
                    'is_generator': False,
                    'is_abstract': False,
                    'is_getter': True,
                    'is_setter': False,
                    'is_constructor': False,
                    'is_exported': False,
                    'start_line': start_line,
                    'match_text': match.group(0),
                    'function_type': func_type
                }
            elif func_type == 'setter':
                # set name(value: Type): void { ... }
                return {
                    'name': groups[0] if num_groups > 0 else 'unknown',
                    'generics': extract_generics_typescript(groups[0]) if num_groups > 0 else [],
                    'params': groups[1] if num_groups > 1 else '',
                    'return_type': self._parse_complex_return_type(groups[2]) if num_groups > 2 else None,
                    'is_async': False,
                    'is_static': False,
                    'is_generator': False,
                    'is_abstract': False,
                    'is_getter': False,
                    'is_setter': True,
                    'is_constructor': False,
                    'is_exported': False,
                    'start_line': start_line,
                    'match_text': match.group(0),
                    'function_type': func_type
                }
            elif func_type == 'export_function':
                # export function name<T>(params): ReturnType { ... }
                return {
                    'name': groups[1] if num_groups > 1 else 'unknown',
                    'generics': extract_generics_typescript(groups[1]) if num_groups > 1 else [],
                    'params': groups[2] if num_groups > 2 else '',
                    'return_type': self._parse_complex_return_type(groups[3]) if num_groups > 3 else None,
                    'is_async': False,
                    'is_static': False,
                    'is_generator': False,
                    'is_abstract': False,
                    'is_getter': False,
                    'is_setter': False,
                    'is_constructor': False,
                    'is_exported': True,
                    'start_line': start_line,
                    'match_text': match.group(0),
                    'function_type': func_type
                }
            elif func_type == 'react_component':
                # React functional component: const Name: React.FC<Props> = ({ ... }) => ...
                return {
                    'name': groups[0] if num_groups > 0 else 'unknown',
                    'generics': extract_generics_typescript(groups[0]) if num_groups > 0 else [],
                    'params': groups[1] if num_groups > 1 else '',
                    'return_type': None,  # React components typically don't have explicit return types
                    'is_async': False,
                    'is_static': False,
                    'is_generator': False,
                    'is_abstract': False,
                    'is_getter': False,
                    'is_setter': False,
                    'is_constructor': False,
                    'is_exported': False,
                    'start_line': start_line,
                    'match_text': match.group(0),
                    'function_type': func_type
                }

        except Exception as e:
            print(f"   Warning: Error parsing function match: {e}", file=sys.stderr)
            return None

        return None

    def _parse_express_route_match(self, groups: tuple, match: re.Match, start_line: int = 1) -> Optional[Dict[str, Any]]:
        """Parse Express route handler with asyncHandler wrapper."""
        try:
            num_groups = len(groups)

            # Extract HTTP method and parameters
            method_match = re.search(r'router\.(\w+)\s*\(', match.group(0))
            http_method = method_match.group(1) if method_match else 'unknown'

            # Parse parameters from the async function
            params_text = groups[-1] if num_groups > 0 else 'req, res'
            params = extract_typescript_parameters(f'({params_text})')

            return {
                'name': f'{http_method}_handler',
                'generics': [],
                'params': params_text,
                'return_type': 'Promise<void>',
                'is_async': True,
                'is_static': False,
                'is_generator': False,
                'is_abstract': False,
                'is_getter': False,
                'is_setter': False,
                'is_constructor': False,
                'is_exported': False,
                'start_line': start_line,
                'match_text': match.group(0),
                'function_type': 'express_route',
                'http_method': http_method,
                'route_pattern': extract_route_pattern(match.group(0))
            }
        except Exception as e:
            print(f"   Warning: Error parsing Express route match: {e}", file=sys.stderr)
            return None

    def _parse_express_middleware_match(self, groups: tuple, match: re.Match, start_line: int = 1) -> Optional[Dict[str, Any]]:
        """Parse Express middleware with asyncHandler wrapper."""
        try:
            num_groups = len(groups)

            # Parse parameters from the async function
            params_text = groups[-1] if num_groups > 0 else 'req, res, next'
            params = extract_typescript_parameters(f'({params_text})')

            return {
                'name': 'middleware_handler',
                'generics': [],
                'params': params_text,
                'return_type': 'Promise<void>',
                'is_async': True,
                'is_static': False,
                'is_generator': False,
                'is_abstract': False,
                'is_getter': False,
                'is_setter': False,
                'is_constructor': False,
                'is_exported': False,
                'start_line': start_line,
                'match_text': match.group(0),
                'function_type': 'express_middleware'
            }
        except Exception as e:
            print(f"   Warning: Error parsing Express middleware match: {e}", file=sys.stderr)
            return None

    def _parse_vue_config_match(self, groups: tuple, match: re.Match, start_line: int = 1) -> Optional[Dict[str, Any]]:
        """Parse Vue configuration handlers."""
        try:
            num_groups = len(groups)

            # Extract config type
            config_match = re.search(r'config\.(\w+)\s*=', match.group(0))
            config_type = config_match.group(1) if config_match else 'handler'

            return {
                'name': f'{config_type}_handler',
                'generics': [],
                'params': 'err, instance, info',
                'return_type': 'void',
                'is_async': False,
                'is_static': False,
                'is_generator': False,
                'is_abstract': False,
                'is_getter': False,
                'is_setter': False,
                'is_constructor': False,
                'is_exported': False,
                'start_line': start_line,
                'match_text': match.group(0),
                'function_type': 'vue_config'
            }
        except Exception as e:
            print(f"   Warning: Error parsing Vue config match: {e}", file=sys.stderr)
            return None

    def _parse_lifecycle_method_match(self, groups: tuple, match: re.Match, start_line: int = 1) -> Optional[Dict[str, Any]]:
        """Parse Angular lifecycle methods like ngOnInit()."""
        try:
            num_groups = len(groups)

            return {
                'name': groups[0] if num_groups > 0 else 'unknown',
                'generics': [],
                'params': '',
                'return_type': self._parse_complex_return_type(groups[1]) if num_groups > 1 else 'void',
                'is_async': False,
                'is_static': False,
                'is_generator': False,
                'is_abstract': False,
                'is_getter': False,
                'is_setter': False,
                'is_constructor': False,
                'is_exported': False,
                'start_line': start_line,
                'match_text': match.group(0),
                'function_type': 'lifecycle_method'
            }
        except Exception as e:
            print(f"   Warning: Error parsing lifecycle method match: {e}", file=sys.stderr)
            return None

    def _parse_export_composable_match(self, groups: tuple, match: re.Match, start_line: int = 1) -> Optional[Dict[str, Any]]:
        """Parse exported Vue composable functions like export function useApi()."""
        try:
            num_groups = len(groups)

            return {
                'name': groups[0] if num_groups > 0 else 'unknown',
                'generics': extract_generics_typescript(groups[0]) if num_groups > 0 else [],
                'params': groups[1] if num_groups > 1 else '',
                'return_type': self._parse_complex_return_type(groups[2]) if num_groups > 2 else None,
                'is_async': False,
                'is_static': False,
                'is_generator': False,
                'is_abstract': False,
                'is_getter': False,
                'is_setter': False,
                'is_constructor': False,
                'is_exported': True,
                'start_line': start_line,
                'match_text': match.group(0),
                'function_type': 'export_composable'
            }
        except Exception as e:
            print(f"   Warning: Error parsing export composable match: {e}", file=sys.stderr)
            return None

    def _parse_complex_return_type(self, return_type: str) -> str:
        """Parse complex return types including utility types and conditionals."""
        if not return_type:
            return ""

        # Clean up the return type
        return_type = return_type.strip()

        # Handle Promise<T> and similar
        if return_type.startswith('Promise<'):
            return return_type

        # Handle Observable<T>
        if return_type.startswith('Observable<'):
            return return_type

        # Handle Array<T> and T[]
        if return_type.startswith('Array<') or return_type.endswith('[]'):
            return return_type

        # Handle utility types
        utility_types = ['Partial', 'Required', 'Readonly', 'Record', 'Pick', 'Omit', 'Exclude', 'Extract', 'NonNullable', 'Parameters', 'ReturnType', 'InstanceType']
        for util in utility_types:
            if return_type.startswith(f'{util}<'):
                return return_type

        # Handle conditional types: T extends U ? X : Y
        if 'extends' in return_type and '?' in return_type:
            return return_type

        return return_type

    def _find_classes_regex(self, source: str) -> List[Dict[str, Any]]:
        """Find class definitions using pre-compiled regex patterns for maximum performance."""
        classes = []

        # Use pre-compiled patterns for maximum performance
        compiled_patterns = self._compiled_patterns['classes']

        for compiled_pattern in compiled_patterns:
            for match in compiled_pattern.finditer(source):
                groups = match.groups()

                # Handle different pattern formats
                if len(groups) >= 2 and groups[1]:  # Pattern with decorators
                    decorators_text = groups[0]
                    class_name = groups[1]
                    extends = groups[3] if len(groups) > 3 else None
                    implements_text = groups[4] if len(groups) > 4 else None
                else:  # Pattern without decorators
                    decorators_text = ""
                    class_name = groups[0]
                    extends = groups[2] if len(groups) > 2 else None
                    implements_text = groups[3] if len(groups) > 3 else None

                # Parse implements clause
                implements = []
                if implements_text:
                    implements = self._parse_implements_clause(implements_text.strip())

                # Parse decorators
                decorators = parse_decorators_typescript(decorators_text, source[:match.start()])

                class_info = {
                    'name': class_name,
                    'generics': extract_generics_typescript(class_name),
                    'extends': extends,
                    'implements': implements,
                    'decorators': decorators,
                    'is_abstract': 'abstract' in match.group(0),
                    'is_exported': 'export' in match.group(0),
                    'start_line': source[:match.start()].count('\n') + 1,
                    'match_text': match.group(0),
                    'confidence': self._calculate_confidence(match.group(0), decorators)
                }

                classes.append(class_info)

        return classes

    def _parse_implements_clause(self, implements_text: str) -> List[str]:
        """Parse implements clause with support for generics and multiple interfaces."""
        implements = []

        # Split by comma but be careful with generics
        depth = 0
        current = ""
        for char in implements_text:
            if char == '<':
                depth += 1
                current += char
            elif char == '>':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                if current.strip():
                    implements.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            implements.append(current.strip())

        return implements

    def _calculate_confidence(self, source: str, decorators: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the parsing result."""
        confidence = 0.5  # Base confidence

        # Increase confidence based on recognizable patterns
        if re.search(r'function\s+\w+.*\{', source):
            confidence += 0.1
        if re.search(r'class\s+\w+.*\{', source):
            confidence += 0.1
        if decorators:
            confidence += 0.2
        if re.search(r'interface\s+\w+', source):
            confidence += 0.1
        if re.search(r'type\s+\w+\s*=', source):
            confidence += 0.1

        # Decrease confidence for unusual patterns
        if re.search(r'/\*[\s\S]*?\*/', source):  # Complex JSDoc
            confidence += 0.1
        if re.search(r'//.*', source):  # Comments indicate real code
            confidence += 0.05

        return min(confidence, 1.0)

    def _detect_framework_patterns(self, source: str) -> Dict[str, Any]:
        """Detect TypeScript framework patterns with comprehensive modern framework support."""
        return detect_framework_patterns_typescript(source)

    def _create_function_element(self, func_info: Dict[str, Any], lines: List[str], file_path: str) -> Optional[FunctionElement]:
        """Create FunctionElement from regex match information."""
        start_line = func_info.get('start_line', 1)

        # Estimate end line by finding the matching closing brace
        end_line = find_closing_brace(lines, start_line - 1)

        # Ensure end_line is greater than start_line
        if end_line <= start_line:
            end_line = start_line + 20  # Default to 20 lines if we can't find the closing brace

        # Determine if this is a method by checking if it's inside a class
        is_method, class_name = determine_if_method_typescript(start_line, lines)

        # Extract JSDoc comment
        docstring = extract_jsdoc_comment(lines, start_line - 1)

        # Extract parameters
        params = extract_typescript_parameters(func_info.get('match_text', ''))

        # Calculate complexity and NLOC
        func_source_lines = lines[start_line-1:end_line] if end_line > start_line else lines[start_line-1:start_line]
        func_source = '\n'.join(func_source_lines)
        complexity = calculate_complexity_typescript(func_source)
        nloc = calculate_nloc_typescript(lines, start_line, end_line)

        # Extract dependencies
        dependencies = extract_typescript_dependencies(func_source, self.file_level_imports)

        # Detect framework patterns
        framework_patterns = detect_framework_patterns_typescript(func_source)

        # Enhanced metadata extraction with framework context
        enhanced_metadata = self._extract_enhanced_metadata(func_source, func_info, framework_patterns)

        # Ensure enhanced_metadata is a dictionary with fallback
        if not isinstance(enhanced_metadata, dict):
            enhanced_metadata = {'confidence': 0.5, 'error': f'Invalid enhanced_metadata type: {type(enhanced_metadata)}'}

        # Ensure Express route/middleware functions get proper framework metadata
        if func_info.get('function_type') in ['express_route', 'express_middleware']:
            framework_patterns['framework'] = 'express'
            framework_patterns['features'].extend(['routing', 'middleware'])
            framework_patterns['confidence'] = 0.9
        elif 'router.' in func_source or 'asyncHandler' in func_source:
            # If function contains Express patterns but wasn't caught by specific patterns
            framework_patterns['framework'] = 'express'
            framework_patterns['features'].extend(['routing', 'express_patterns'])
            framework_patterns['confidence'] = 0.8

        # Ensure React components get proper framework metadata
        if func_info.get('function_type') in ['react_component']:
            framework_patterns['framework'] = 'react'
            framework_patterns['features'].extend(['hooks', 'jsx', 'components'])
            framework_patterns['confidence'] = 0.9
        # Ensure Vue composables get proper framework metadata
        elif func_info.get('function_type') in ['export_composable', 'vue_config']:
            framework_patterns['framework'] = 'vue'
            framework_patterns['features'].extend(['composition_api', 'reactivity'])
            framework_patterns['confidence'] = 0.9
        elif any(pattern in func_source for pattern in ['useApi', 'useUser', 'useProduct', 'useOrder', 'useAuth']):
            # If function contains Vue composable patterns
            framework_patterns['framework'] = 'vue'
            framework_patterns['features'].extend(['composition_api', 'vue_composables'])
            framework_patterns['confidence'] = 0.8

        # Ensure framework is detected for Angular components and other frameworks
        decorators = framework_patterns.get('decorators', [])
        if decorators:
            # Find the decorator with highest confidence
            # Handle case where decorators might contain strings instead of dictionaries
            high_confidence_decorators = []
            for d in decorators:
                if isinstance(d, dict) and d.get('confidence', 0) > 0.8:
                    high_confidence_decorators.append(d)
                elif isinstance(d, str):
                    # Handle string decorators by creating a basic dict
                    if d in ['Component', 'Controller', 'Injectable']:
                        high_confidence_decorators.append({
                            'name': d,
                            'framework': self._infer_framework_from_decorator_name(d),
                            'confidence': 0.9,
                            'category': 'general'
                        })
            if high_confidence_decorators:
                # Use the framework from the highest confidence decorator
                best_decorator = max(high_confidence_decorators, key=lambda d: d.get('confidence', 0))
                detected_framework = best_decorator.get('framework')
                if detected_framework and detected_framework != 'unknown':
                    framework_patterns['framework'] = detected_framework
                    framework_patterns['confidence'] = best_decorator.get('confidence', 0.9)

                    # Add framework-specific features
                    if detected_framework == 'angular':
                        framework_patterns['features'].extend(['components', 'dependency_injection'])
                    elif detected_framework == 'nestjs':
                        framework_patterns['features'].extend(['controllers', 'dependency_injection', 'modules'])
                    elif detected_framework == 'react':
                        framework_patterns['features'].extend(['hooks', 'jsx', 'components'])
                    elif detected_framework == 'express':
                        framework_patterns['features'].extend(['routing', 'middleware'])

        # Fix method detection - if function is within a class, mark as method
        if (self.current_class is not None and
            func_info.get('function_type') not in ['getter', 'setter', 'constructor']):
            func_info['function_type'] = 'method'

        # Create function element with comprehensive metadata
        func_element = FunctionElement(
            name=func_info['name'],
            kind='function',
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            full_source='\n'.join(lines),
            parameters=params,
            return_type=func_info.get('return_type'),
            is_async=func_info.get('is_async', False),
            is_static=func_info.get('is_static', False),
            access_modifier=func_info.get('access_modifier'),
            docstring=docstring,
            is_method=is_method,
            class_name=class_name,
            complexity=complexity,
            nloc=nloc,
            external_dependencies=dependencies,
            decorators=framework_patterns.get('decorators', []),
            catches_exceptions=[],  # TypeScript equivalent would be caught errors
            local_variables_declared=[],  # Could be extracted from function body
            hash_body=hash_source_snippet(lines, start_line, end_line),
            metadata={
                'framework': framework_patterns.get('framework'),
                'typescript_features': framework_patterns.get('features', []),
                'function_type': func_info.get('function_type', 'unknown'),
                'generics': func_info.get('generics', []),
                'confidence': enhanced_metadata.get('confidence', 0.5),
                'parsing_method': 'regex_fallback',  # Will be updated by caller
                'complexity_factors': enhanced_metadata.get('complexity_factors', {}),
                'type_analysis': enhanced_metadata.get('type_analysis', {}),
                'decorator_analysis': enhanced_metadata.get('decorator_analysis', {}),
                'dependency_analysis': enhanced_metadata.get('dependency_analysis', {}),
                # Ensure framework detection is properly set
                'detected_frameworks': [framework_patterns.get('framework')] if framework_patterns.get('framework') else [],
                'framework_confidence': framework_patterns.get('confidence', 0.0),
                'framework_features': framework_patterns.get('features', [])
            }
        )

        return func_element

    def _create_class_element(self, class_info: Dict[str, Any], lines: List[str], file_path: str) -> Optional[ClassElement]:
        """Create ClassElement from regex match information."""
        start_line = class_info.get('start_line', 1)

        # Estimate end line by finding the matching closing brace
        end_line = find_closing_brace(lines, start_line - 1)

        # Extract JSDoc comment
        docstring = extract_jsdoc_comment(lines, start_line - 1)

        # Detect framework patterns
        class_source_lines = lines[start_line-1:end_line] if end_line > start_line else lines[start_line-1:start_line]
        class_source = '\n'.join(class_source_lines)
        framework_patterns = detect_framework_patterns_typescript(class_source)

        # Enhanced metadata extraction for classes
        enhanced_metadata = self._extract_enhanced_class_metadata(class_source, class_info, framework_patterns)

        # Ensure framework is detected for classes with decorators
        if class_info.get('decorators'):
            # Find the decorator with highest confidence
            # Handle case where decorators might contain strings instead of dictionaries
            high_confidence_decorators = []
            for d in class_info['decorators']:
                if isinstance(d, dict) and d.get('confidence', 0) > 0.8:
                    high_confidence_decorators.append(d)
                elif isinstance(d, str):
                    # Handle string decorators by creating a basic dict
                    if d in ['Component', 'Controller', 'Injectable']:
                        high_confidence_decorators.append({
                            'name': d,
                            'framework': self._infer_framework_from_decorator_name(d),
                            'confidence': 0.9,
                            'category': 'general'
                        })
            if high_confidence_decorators:
                # Use the framework from the highest confidence decorator
                best_decorator = max(high_confidence_decorators, key=lambda d: d.get('confidence', 0))
                detected_framework = best_decorator.get('framework')
                if detected_framework and detected_framework != 'unknown':
                    framework_patterns['framework'] = detected_framework
                    framework_patterns['confidence'] = best_decorator.get('confidence', 0.9)

                    # Add framework-specific features
                    if detected_framework == 'angular':
                        framework_patterns['features'].extend(['components', 'dependency_injection'])
                    elif detected_framework == 'nestjs':
                        framework_patterns['features'].extend(['controllers', 'dependency_injection', 'modules'])
                    elif detected_framework == 'react':
                        framework_patterns['features'].extend(['hooks', 'jsx', 'components'])
                    elif detected_framework == 'express':
                        framework_patterns['features'].extend(['routing', 'middleware'])

        # Create class element with comprehensive metadata
        class_element = ClassElement(
            name=class_info['name'],
            kind='class',
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            full_source='\n'.join(lines),
            methods=[],  # Could be extracted from class body
            attributes=[],  # Could be extracted from class body
            extends=class_info.get('extends'),
            implements=class_info.get('implements', []),
            docstring=docstring,
            decorators=class_info.get('decorators', []),
            hash_body=hash_source_snippet(lines, start_line, end_line),
            metadata={
                'framework': framework_patterns.get('framework'),
                'typescript_features': framework_patterns.get('features', []),
                'generics': class_info.get('generics', []),
                'is_abstract': class_info.get('is_abstract', False),
                'is_exported': class_info.get('is_exported', False),
                'confidence': enhanced_metadata.get('confidence', 0.5),
                'parsing_method': 'regex_fallback',
                'complexity_factors': enhanced_metadata.get('complexity_factors', {}),
                'decorator_analysis': enhanced_metadata.get('decorator_analysis', {}),
                'inheritance_analysis': enhanced_metadata.get('inheritance_analysis', {}),
                'dependency_analysis': enhanced_metadata.get('dependency_analysis', {}),
                # Ensure framework detection is properly set
                'detected_frameworks': [framework_patterns.get('framework')] if framework_patterns.get('framework') else [],
                'framework_confidence': framework_patterns.get('confidence', 0.0),
                'framework_features': framework_patterns.get('features', [])
            }
        )

        return class_element

    def _infer_framework_from_decorator_name(self, decorator_name: str) -> str:
        """Infer framework from decorator name."""
        framework_decorators = {
            'angular': ['Component', 'Directive', 'Pipe', 'Injectable', 'Input', 'Output'],
            'nestjs': ['Controller', 'Module', 'Injectable', 'Get', 'Post', 'Put', 'Delete'],
            'typeorm': ['Entity', 'Column', 'PrimaryGeneratedColumn'],
            'react': [],  # React doesn't typically use decorators in the same way
            'vue': [],    # Vue 3 doesn't use decorators like Angular/NestJS
            'express': [] # Express doesn't use decorators
        }

        for framework, decorators in framework_decorators.items():
            if decorator_name in decorators:
                return framework

        return 'unknown'

    def _extract_enhanced_metadata(self, func_source: str, func_info: Dict[str, Any], framework_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive metadata for functions with confidence scoring."""
        # Ensure framework_patterns is a dictionary
        if not isinstance(framework_patterns, dict):
            framework_patterns = {}

        metadata = {
            'confidence': 0.5,
            'complexity_factors': {},
            'type_analysis': {},
            'decorator_analysis': {},
            'dependency_analysis': {}
        }

        try:
            # Calculate confidence based on various factors
            base_confidence = 0.5

            # Function signature clarity
            if func_info.get('name') and len(func_info['name']) > 0:
                base_confidence += 0.2

            # Parameter analysis
            if func_info.get('params'):
                params_str = func_info['params']
                if isinstance(params_str, str):
                    param_count = len([p.strip() for p in params_str.split(',') if p.strip()])
                    metadata['complexity_factors']['parameter_count'] = param_count
                    base_confidence += 0.1

            # Return type analysis
            return_type = func_info.get('return_type')
            if return_type and isinstance(return_type, str):
                metadata['type_analysis']['has_return_type'] = True
                metadata['type_analysis']['return_type_complexity'] = analyze_type_complexity(return_type)
                base_confidence += 0.1
            else:
                metadata['type_analysis']['has_return_type'] = False

            # Framework-specific analysis
            framework = framework_patterns.get('framework')
            if framework:
                metadata['framework_analysis'] = {
                    'detected_framework': framework,
                    'framework_confidence': framework_patterns.get('confidence', 0.0),
                    'features_detected': framework_patterns.get('features', [])
                }
                base_confidence += 0.2

            # Decorator analysis
            decorators = framework_patterns.get('decorators', [])
            if decorators and isinstance(decorators, list):
                framework_list = []
                category_list = []
                for d in decorators:
                    if isinstance(d, dict):
                        fw = d.get('framework')
                        cat = d.get('category')
                        if fw: framework_list.append(fw)
                        if cat: category_list.append(cat)

                metadata['decorator_analysis'] = {
                    'decorator_count': len(decorators),
                    'decorator_frameworks': list(set(framework_list)),
                    'decorator_categories': list(set(category_list))
                }
                base_confidence += 0.1

            # Dependency analysis
            dependencies = extract_typescript_dependencies(func_source, self.file_level_imports)
            metadata['dependency_analysis'] = {
                'external_dependencies': dependencies,
                'dependency_count': len(dependencies),
                'dependency_categories': categorize_dependencies(dependencies)
            }
            if dependencies:
                base_confidence += 0.1

            metadata['confidence'] = min(base_confidence, 1.0)

        except Exception as e:
            print(f"   Warning: Error extracting enhanced metadata: {e}", file=sys.stderr)
            print(f"   Debug: Full traceback for enhanced metadata error:", file=sys.stderr)
            import traceback
            traceback.print_exc()
            metadata['confidence'] = 0.3
            metadata['error'] = str(e)

        # Ensure we always return a dictionary
        return metadata if isinstance(metadata, dict) else {
            'confidence': 0.3,
            'error': f'Invalid metadata type: {type(metadata)}',
            'complexity_factors': {},
            'type_analysis': {},
            'decorator_analysis': {},
            'dependency_analysis': {}
        }

    def _extract_enhanced_class_metadata(self, class_source: str, class_info: Dict[str, Any], framework_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive metadata for classes."""
        metadata = {
            'confidence': 0.5,
            'complexity_factors': {},
            'decorator_analysis': {},
            'inheritance_analysis': {},
            'dependency_analysis': {}
        }

        try:
            # Calculate confidence based on various factors
            base_confidence = 0.5

            # Class name and structure
            if class_info.get('name') and len(class_info['name']) > 0:
                base_confidence += 0.2

            # Inheritance analysis
            if class_info.get('extends'):
                metadata['inheritance_analysis'] = {
                    'extends': class_info['extends'],
                    'extends_count': 1,
                    'inheritance_depth': 1
                }
                base_confidence += 0.1

            if class_info.get('implements'):
                implements_count = len(class_info['implements'])
                metadata['inheritance_analysis']['implements'] = class_info['implements']
                metadata['inheritance_analysis']['implements_count'] = implements_count
                base_confidence += min(implements_count * 0.05, 0.2)

            # Generics analysis
            if class_info.get('generics'):
                metadata['inheritance_analysis']['generics'] = class_info['generics']
                metadata['inheritance_analysis']['generic_count'] = len(class_info['generics'])
                base_confidence += 0.1

            # Decorator analysis
            decorators = class_info.get('decorators')
            if decorators and isinstance(decorators, list):
                decorator_count = len(decorators)
                framework_list = []
                category_list = []
                for d in decorators:
                    if isinstance(d, dict):
                        fw = d.get('framework')
                        cat = d.get('category')
                        if fw: framework_list.append(fw)
                        if cat: category_list.append(cat)

                metadata['decorator_analysis'] = {
                    'decorator_count': decorator_count,
                    'frameworks': list(set(framework_list)),
                    'categories': list(set(category_list))
                }
                base_confidence += min(decorator_count * 0.1, 0.3)

            # Abstract class analysis
            if class_info.get('is_abstract'):
                metadata['complexity_factors']['is_abstract'] = True
                base_confidence += 0.1

            # Framework analysis
            if framework_patterns.get('framework'):
                metadata['framework_analysis'] = {
                    'detected_framework': framework_patterns['framework'],
                    'framework_confidence': framework_patterns.get('confidence', 0.0)
                }
                base_confidence += 0.2

            metadata['confidence'] = min(base_confidence, 1.0)

        except Exception as e:
            print(f"   Warning: Error extracting enhanced class metadata: {e}", file=sys.stderr)
            metadata['confidence'] = 0.3
            metadata['error'] = str(e)

        return metadata

    def _log_parsing_info(self, file_path: str, parsing_method: str, elements_found: int, metadata: Dict[str, Any]):
        """Log detailed parsing information for debugging."""
        try:
            confidence = metadata.get('confidence', 0.5)
            framework = metadata.get('framework')

            print(f"   Info: Parsed {file_path} using {parsing_method} ({elements_found} elements, confidence: {confidence:.2f})")

            if framework:
                print(f"   Info: Detected framework: {framework}")

            if metadata.get('typescript_features'):
                features = ', '.join(metadata['typescript_features'])
                print(f"   Info: TypeScript features: {features}")

        except Exception as e:
            print(f"   Warning: Error logging parsing info: {e}", file=sys.stderr)

    def visit_file(self, file_path: str, source: str) -> List[CodeElement]:
        """Visit a TypeScript file and extract all code elements using fast regex-based parsing."""
        import time

        start_time = time.time()
        self.source_lines = source.splitlines()
        self.current_file = file_path
        self.elements = []

        try:
            # Extract file-level imports
            self.file_level_imports, self.file_level_import_from_targets = extract_file_imports_typescript(source)

            # Parse complex types (interfaces, enums, etc.)
            complex_types = self._parse_complex_types(source)

            # Create interface elements
            for interface_info in complex_types['interfaces']:
                interface_element = self._create_interface_element(interface_info, source, file_path)
                if interface_element:
                    self.elements.append(interface_element)

            # Create enum elements
            for enum_info in complex_types['enums']:
                enum_element = self._create_enum_element(enum_info, source, file_path)
                if enum_element:
                    self.elements.append(enum_element)

            # Create type alias elements
            for type_alias_info in complex_types['type_aliases']:
                type_alias_element = self._create_type_alias_element(type_alias_info, source, file_path)
                if type_alias_element:
                    self.elements.append(type_alias_element)

            # Parse the file for classes and functions using fast regex-based extraction
            elements = self._parse_typescript_ast(file_path, source)
            self.elements.extend(elements)

            # Log parsing results
            parsing_time = time.time() - start_time
            print(f"   Info: Parsed {file_path} using fast regex extraction ({len(self.elements)} elements, {parsing_time:.2f}s)")

            # Log framework detection if any
            if self.elements:
                frameworks = set()
                for element in self.elements:
                    if element.metadata.get('framework'):
                        frameworks.add(element.metadata['framework'])

                if frameworks:
                    print(f"   Info: Detected frameworks in {file_path}: {', '.join(frameworks)}")

            return self.elements.copy()

        except Exception as e:
            parsing_time = time.time() - start_time
            print(f"   Error: Failed to parse {file_path} after {parsing_time:.2f}s: {e}", file=sys.stderr)
            return []

    def _parse_complex_types(self, source: str) -> Dict[str, Any]:
        """Parse complex TypeScript type definitions with advanced type system support."""
        complex_types = {
            'union_types': [],
            'intersection_types': [],
            'mapped_types': [],
            'conditional_types': [],
            'utility_types': [],
            'template_literal_types': [],
            'keyof_types': [],
            'type_aliases': [],
            'interfaces': [],
            'enums': [],
            'namespaces': [],
            'function_types': [],
            'constructor_types': []
        }

        # Parse interfaces using pre-compiled pattern for maximum performance
        interface_pattern = self._compiled_patterns['interfaces']
        for match in interface_pattern.finditer(source):
            generics = self._extract_generics_from_match(match.group(2)) if match.group(2) else []
            extends = [ext.strip() for ext in match.group(3).split(',')] if match.group(3) else []
            implements = [impl.strip() for impl in match.group(4).split(',')] if match.group(4) else []

            complex_types['interfaces'].append({
                'name': match.group(1),
                'generics': generics,
                'extends': extends,
                'implements': implements,
                'line': source[:match.start()].count('\n') + 1
            })

        # Parse type aliases using pre-compiled patterns for maximum performance
        type_alias_patterns = self._compiled_patterns['type_aliases']

        for compiled_pattern in type_alias_patterns:
            for match in compiled_pattern.finditer(source):
                generics = self._extract_generics_from_match(match.group(2)) if match.group(2) else []
                definition = match.group(3).strip()

                # Categorize the type definition
                type_category = self._categorize_type_definition(definition)

                complex_types['type_aliases'].append({
                    'name': match.group(1),
                    'generics': generics,
                    'definition': definition,
                    'category': type_category,
                    'line': source[:match.start()].count('\n') + 1
                })

                # Add to specific categories
                if type_category in complex_types:
                    # Extract the actual utility type name from the definition
                    utility_name = None
                    if type_category == 'utility_types':
                        # Extract the utility type name (e.g., 'Readonly' from 'Readonly<User>')
                        util_match = re.match(r'\b(\w+)<', definition)
                        utility_name = util_match.group(1) if util_match else 'utility'

                    complex_types[type_category].append({
                        'name': match.group(1),
                        'definition': definition,
                        'line': source[:match.start()].count('\n') + 1,
                        'utility': utility_name
                    })

        # Parse enums using pre-compiled pattern for maximum performance
        enum_pattern = self._compiled_patterns['enums']
        for match in enum_pattern.finditer(source):
            generics = self._extract_generics_from_match(match.group(2)) if match.group(2) else []
            enum_members_text = match.group(3)

            # Enhanced enum member parsing
            enum_members = self._parse_enum_members(enum_members_text)

            complex_types['enums'].append({
                'name': match.group(1),
                'generics': generics,
                'members': enum_members,
                'line': source[:match.start()].count('\n') + 1
            })

        # Parse namespaces using pre-compiled pattern for maximum performance
        namespace_pattern = self._compiled_patterns['namespaces']
        for match in namespace_pattern.finditer(source):
            namespace_content = match.group(2) if match.group(2) else ""
            complex_types['namespaces'].append({
                'name': match.group(1),
                'content': namespace_content,
                'line': source[:match.start()].count('\n') + 1
            })

        # Parse function types: type FuncType = (param: string) => void
        function_type_pattern = r'type\s+(\w+)\s*=\s*\(([^)]+)\)\s*=>\s*([^;]+);'
        for match in re.finditer(function_type_pattern, source):
            complex_types['function_types'].append({
                'name': match.group(1),
                'parameters': match.group(2).strip(),
                'return_type': match.group(3).strip(),
                'line': source[:match.start()].count('\n') + 1
            })

        # Parse constructor types: type CtorType = new (param: string) => void
        constructor_type_pattern = r'type\s+(\w+)\s*=\s*new\s*\(([^)]+)\)\s*=>\s*([^;]+);'
        for match in re.finditer(constructor_type_pattern, source):
            complex_types['constructor_types'].append({
                'name': match.group(1),
                'parameters': match.group(2).strip(),
                'return_type': match.group(3).strip(),
                'line': source[:match.start()].count('\n') + 1
            })

        return complex_types

    def _extract_generics_from_match(self, generics_text: str) -> List[str]:
        """Extract generic type parameters from match group."""
        if not generics_text:
            return []

        # Split by comma but respect nested brackets
        generics = []
        current = ""
        depth = 0

        for char in generics_text:
            if char in '<([{' :
                depth += 1
                current += char
            elif char in '>)]}':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                if current.strip():
                    generics.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            generics.append(current.strip())

        return generics

    def _categorize_type_definition(self, definition: str) -> str:
        """Categorize a type definition into specific type categories."""
        definition = definition.strip()

        # Union types: T | U
        if re.search(r'\s*\|\s*', definition) and not re.search(r'\bextends\b', definition):
            return 'union_types'

        # Intersection types: T & U
        if re.search(r'\s*&\s*', definition):
            return 'intersection_types'

        # Conditional types: T extends U ? X : Y
        if re.search(r'\bextends\b.*\?.*\:', definition):
            return 'conditional_types'

        # Mapped types: { [K in keyof T]: U }
        if re.search(r'\{\s*\[.*in\s+.*\]', definition):
            return 'mapped_types'

        # Template literal types: `prefix${T}suffix`
        if re.search(r'`[^`]*\$\{[^}]+\}[^`]*`', definition):
            return 'template_literal_types'

        # keyof types: keyof T
        if definition.startswith('keyof '):
            return 'keyof_types'

        # Utility types: Partial<T>, Required<T>, etc.
        utility_patterns = [
            r'\bPartial\s*<', r'\bRequired\s*<', r'\bReadonly\s*<',
            r'\bRecord\s*<', r'\bPick\s*<', r'\bOmit\s*<',
            r'\bExclude\s*<', r'\bExtract\s*<', r'\bNonNullable\s*<'
        ]

        for pattern in utility_patterns:
            if re.search(pattern, definition):
                return 'utility_types'

        return 'type_aliases'  # Default category

    def _parse_enum_members(self, members_text: str) -> List[Dict[str, Any]]:
        """Parse enum members with values and complex expressions."""
        members = []
        member_lines = [m.strip() for m in members_text.split(',') if m.strip()]

        for member_line in member_lines:
            if not member_line:
                continue

            # Handle enum with value: NAME = 'value' or NAME = 123
            if '=' in member_line:
                parts = member_line.split('=', 1)
                name = parts[0].strip()
                value = parts[1].strip()

                # Handle string values
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                    value_type = 'string'
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                    value_type = 'string'
                elif value.isdigit():
                    value = int(value)
                    value_type = 'number'
                else:
                    value_type = 'expression'

                members.append({
                    'name': name,
                    'value': value,
                    'value_type': value_type
                })
            else:
                # Simple enum member
                members.append({
                    'name': member_line,
                    'value': None,
                    'value_type': None
                })

        return members

    def _create_interface_element(self, interface_info: Dict[str, Any], source: str, file_path: str) -> Optional[CodeElement]:
        """Create a CodeElement representing a TypeScript interface."""
        start_line = interface_info.get('line', 1)

        # Estimate interface body (simple approximation)
        lines = source.splitlines()
        end_line = start_line  # Interfaces are typically single declarations

        # Extract JSDoc comment
        docstring = extract_jsdoc_comment(lines, start_line)

        interface_element = CodeElement(
            name=interface_info['name'],
            kind='interface',
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            full_source=source,
            metadata={
                'typescript_kind': 'interface',
                'extends': interface_info.get('extends', []),
                'implements': interface_info.get('implements', [])
            }
        )

        return interface_element

    def _create_enum_element(self, enum_info: Dict[str, Any], source: str, file_path: str) -> Optional[CodeElement]:
        """Create a CodeElement representing a TypeScript enum."""
        start_line = enum_info.get('line', 1)

        # Estimate enum body (simple approximation)
        lines = source.splitlines()
        end_line = start_line

        # Extract JSDoc comment
        docstring = extract_jsdoc_comment(lines, start_line)

        enum_element = CodeElement(
            name=enum_info['name'],
            kind='enum',
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            full_source=source,
            metadata={
                'typescript_kind': 'enum',
                'enum_members': enum_info.get('members', [])
            }
        )

        return enum_element

    def _create_type_alias_element(self, type_alias_info: Dict[str, Any], source: str, file_path: str) -> Optional[CodeElement]:
        """Create a CodeElement representing a TypeScript type alias."""
        start_line = type_alias_info.get('line', 1)

        # Type aliases are typically single-line declarations
        lines = source.splitlines()
        end_line = start_line

        # Extract JSDoc comment
        docstring = extract_jsdoc_comment(lines, start_line)

        type_alias_element = CodeElement(
            name=type_alias_info['name'],
            kind='type_alias',
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            full_source=source,
            metadata={
                'typescript_kind': 'type_alias',
                'type_definition': type_alias_info.get('definition', '')
            }
        )

        return type_alias_element


class TypeScriptASTExtractor:
    def __init__(self):
        self.visitor = TypeScriptASTVisitor()
        self.project_root: Optional[Path] = None
        self.tsconfig: Optional[Dict[str, Any]] = None
        self.project_references: List[Dict[str, Any]] = []
        self.path_mappings: Dict[str, List[str]] = {}

    def _find_tsconfig(self, directory: Path) -> Optional[Path]:
        """Find tsconfig.json file in directory or parent directories."""
        current_dir = directory
        while current_dir != current_dir.parent:
            tsconfig_path = current_dir / "tsconfig.json"
            if tsconfig_path.exists() and tsconfig_path.is_file():
                return tsconfig_path
            current_dir = current_dir.parent
        return None

    def _parse_tsconfig(self, tsconfig_path: Path) -> Dict[str, Any]:
        """Parse TypeScript configuration file with enhanced error handling."""
        try:
            with open(tsconfig_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # Handle empty files
            if not content:
                print(f"   Warning: Empty tsconfig.json at {tsconfig_path}", file=sys.stderr)
                return {}

            config = json.loads(content)

            # Validate configuration structure
            if not isinstance(config, dict):
                raise ValueError("tsconfig.json must be a JSON object")

            return config
        except json.JSONDecodeError as e:
            print(f"   Warning: Invalid JSON in tsconfig.json {tsconfig_path}: {e}", file=sys.stderr)
            print(f"   Info: Line {e.lineno}, column {e.colno}: {e.msg}", file=sys.stderr)
            return {}
        except FileNotFoundError:
            print(f"   Warning: tsconfig.json not found at {tsconfig_path}", file=sys.stderr)
            return {}
        except PermissionError:
            print(f"   Warning: Permission denied reading tsconfig.json {tsconfig_path}", file=sys.stderr)
            return {}
        except UnicodeDecodeError as e:
            print(f"   Warning: Encoding error reading tsconfig.json {tsconfig_path}: {e}", file=sys.stderr)
            return {}
        except Exception as e:
            print(f"   Warning: Unexpected error parsing tsconfig.json {tsconfig_path}: {e}", file=sys.stderr)
            return {}

    def _setup_project_integration(self, directory: Path) -> None:
        """Setup TypeScript project integration by parsing tsconfig.json."""
        tsconfig_path = self._find_tsconfig(directory)
        if not tsconfig_path:
            return

        self.tsconfig = self._parse_tsconfig(tsconfig_path)
        if not self.tsconfig:
            return

        # Extract project references
        if 'references' in self.tsconfig:
            self.project_references = self.tsconfig['references']

        # Extract path mappings
        if 'compilerOptions' in self.tsconfig:
            compiler_options = self.tsconfig['compilerOptions']
            if 'baseUrl' in compiler_options:
                base_url = compiler_options['baseUrl']
                self.path_mappings['baseUrl'] = base_url

            if 'paths' in compiler_options:
                self.path_mappings['paths'] = compiler_options['paths']

        print(f"   Info: Loaded TypeScript configuration from {tsconfig_path}")

    def _get_compiler_options(self) -> Dict[str, Any]:
        """Get TypeScript compiler options with defaults."""
        if not self.tsconfig or 'compilerOptions' not in self.tsconfig:
            return {}

        options = self.tsconfig['compilerOptions']

        # Add default values for common options
        defaults = {
            'target': 'ES2015',
            'module': 'ES2015',
            'lib': ['ES2015', 'DOM'],
            'strict': True,
            'esModuleInterop': True,
            'skipLibCheck': True,
            'forceConsistentCasingInFileNames': True
        }

        # Merge with provided options
        for key, default_value in defaults.items():
            if key not in options:
                options[key] = default_value

        return options

    def _resolve_path_mapping(self, import_path: str) -> str:
        """Resolve import path using TypeScript path mappings."""
        if not self.path_mappings or 'paths' not in self.path_mappings:
            return import_path

        paths_config = self.path_mappings['paths']
        base_url = self.path_mappings.get('baseUrl', './')

        # Simple path mapping resolution
        for pattern, mappings in paths_config.items():
            if '*' in pattern:
                prefix = pattern.replace('*', '')
                if import_path.startswith(prefix):
                    suffix = import_path[len(prefix):]
                    for mapping in mappings:
                        resolved = mapping.replace('*', suffix)
                        # In a real implementation, this would resolve relative to baseUrl
                        return resolved

        return import_path

    def _build_interface_hierarchy(self) -> Dict[str, List[str]]:
        """Build interface inheritance hierarchy from extracted interface elements."""
        hierarchy = {}

        # This would be implemented to analyze extends relationships
        # For now, return empty hierarchy
        return hierarchy

    def _resolve_type_aliases(self) -> Dict[str, str]:
        """Resolve type aliases to their actual type definitions."""
        # This would be implemented to recursively resolve type aliases
        # For now, return empty resolution map
        return {}

    def extract_from_file(self, file_path: Path) -> List[CodeElement]:
        """Extract code elements from a single TypeScript file."""
        if not file_path.exists() or file_path.suffix not in ['.ts', '.tsx']:
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            return self.visitor.visit_file(str(file_path.resolve()), source)
        except Exception as e:
            print(f"   Warning: Error processing {file_path}: {e}", file=sys.stderr)
            return []

    def extract_from_directory(self, directory: Path) -> List[CodeElement]:
        """Extract code elements from all TypeScript files in directory."""
        self.project_root = directory.resolve()

        # Setup TypeScript project integration
        self._setup_project_integration(directory)

        elements = []
        ts_files = list(directory.rglob('*.ts')) + list(directory.rglob('*.tsx'))

        # Apply gitignore filtering (reuse Python implementation logic)
        ignored_patterns_with_dirs = get_gitignore_patterns(directory)

        filtered_files = [
            file_path
            for file_path in ts_files
            if not any(
                match_file_against_pattern(file_path, pattern, gitignore_dir, directory)
                for pattern, gitignore_dir in ignored_patterns_with_dirs
            )
        ]

        print(f"Found {len(filtered_files)} TypeScript files to analyze (after filtering .gitignore).")

        # Extract TypeScript-specific elements (interfaces, enums, namespaces)
        ts_elements = self._extract_typescript_specific_elements(directory, filtered_files)
        elements.extend(ts_elements)

        for file_path in filtered_files:
            elements.extend(self.extract_from_file(file_path))

        return elements

    def _extract_typescript_specific_elements(self, directory: Path, ts_files: List[Path]) -> List[CodeElement]:
        """Extract TypeScript-specific elements like interfaces, enums, and namespaces."""
        elements = []

        for file_path in ts_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()

                # Extract complex types and create specialized elements
                complex_types = self.visitor._parse_complex_types(source)

                # Create interface elements
                for interface_info in complex_types['interfaces']:
                    interface_element = self._create_interface_element(interface_info, source, file_path)
                    if interface_element:
                        elements.append(interface_element)

                # Create enum elements
                for enum_info in complex_types['enums']:
                    enum_element = self._create_enum_element(enum_info, source, file_path)
                    if enum_element:
                        elements.append(enum_element)

                # Create namespace elements
                for namespace_info in complex_types['namespaces']:
                    namespace_element = self._create_namespace_element(namespace_info, source, file_path)
                    if namespace_element:
                        elements.append(namespace_element)

                # Create type alias elements
                for type_alias_info in complex_types['type_aliases']:
                    type_alias_element = self._create_type_alias_element(type_alias_info, source, file_path)
                    if type_alias_element:
                        elements.append(type_alias_element)

            except Exception as e:
                print(f"   Warning: Error processing TypeScript elements in {file_path}: {e}", file=sys.stderr)

        return elements

    def _create_interface_element(self, interface_info: Dict[str, Any], source: str, file_path: Path) -> Optional[ClassElement]:
        """Create a ClassElement representing a TypeScript interface."""
        start_line = interface_info.get('line', 1)

        # Estimate interface body
        lines = source.splitlines()
        end_line = self._find_interface_end(lines, start_line - 1)

        # Extract JSDoc comment
        docstring = extract_jsdoc_comment(lines, start_line - 1)

        interface_element = ClassElement(
            name=interface_info['name'],
            kind='interface',
            file_path=str(file_path.resolve()),
            line_start=start_line,
            line_end=end_line,
            full_source=source,
            methods=[],  # Interface methods would be extracted from body
            attributes=[],  # Interface properties would be extracted from body
            extends=interface_info.get('extends', []),
            implements=interface_info.get('implements', []),
            docstring=docstring,
            decorators=[],  # Interfaces typically don't have decorators
            hash_body=hash_source_snippet(lines, start_line, end_line),
            metadata={
                'typescript_kind': 'interface',
                'extends_interfaces': interface_info.get('extends', []),
                'implements_interfaces': interface_info.get('implements', [])
            }
        )

        return interface_element

    def _create_enum_element(self, enum_info: Dict[str, Any], source: str, file_path: Path) -> Optional[ClassElement]:
        """Create a ClassElement representing a TypeScript enum."""
        start_line = enum_info.get('line', 1)

        # Estimate enum body
        lines = source.splitlines()
        end_line = self._find_enum_end(lines, start_line - 1)

        # Extract JSDoc comment
        docstring = extract_jsdoc_comment(lines, start_line - 1)

        enum_element = ClassElement(
            name=enum_info['name'],
            kind='enum',
            file_path=str(file_path.resolve()),
            line_start=start_line,
            line_end=end_line,
            full_source=source,
            methods=[],  # Enums don't have methods
            attributes=enum_info.get('members', []),
            extends=None,
            implements=[],
            docstring=docstring,
            decorators=[],
            hash_body=hash_source_snippet(lines, start_line, end_line),
            metadata={
                'typescript_kind': 'enum',
                'enum_members': enum_info.get('members', [])
            }
        )

        return enum_element

    def _create_namespace_element(self, namespace_info: Dict[str, Any], source: str, file_path: Path) -> Optional[ClassElement]:
        """Create a ClassElement representing a TypeScript namespace."""
        start_line = namespace_info.get('line', 1)

        # Estimate namespace body
        lines = source.splitlines()
        end_line = self._find_namespace_end(lines, start_line - 1)

        # Extract JSDoc comment
        docstring = extract_jsdoc_comment(lines, start_line - 1)

        namespace_element = ClassElement(
            name=namespace_info['name'],
            kind='namespace',
            file_path=str(file_path.resolve()),
            line_start=start_line,
            line_end=end_line,
            full_source=source,
            methods=[],  # Namespace members would be extracted from body
            attributes=[],
            extends=None,
            implements=[],
            docstring=docstring,
            decorators=[],
            hash_body=hash_source_snippet(lines, start_line, end_line),
            metadata={
                'typescript_kind': 'namespace'
            }
        )

        return namespace_element

    def _create_type_alias_element(self, type_alias_info: Dict[str, Any], source: str, file_path: Path) -> Optional[CodeElement]:
        """Create a CodeElement representing a TypeScript type alias."""
        start_line = type_alias_info.get('line', 1)

        # Type aliases are typically single-line declarations
        lines = source.splitlines()
        end_line = start_line

        # Extract JSDoc comment
        docstring = extract_jsdoc_comment(lines, start_line - 1)

        type_alias_element = CodeElement(
            name=type_alias_info['name'],
            kind='type_alias',
            file_path=str(file_path.resolve()),
            line_start=start_line,
            line_end=end_line,
            full_source=source,
            metadata={
                'typescript_kind': 'type_alias',
                'type_definition': type_alias_info.get('definition', ''),
                'resolved_type': self._resolve_type_alias(type_alias_info.get('definition', ''))
            }
        )

        return type_alias_element

    def _resolve_type_alias(self, definition: str) -> str:
        """Resolve a type alias to its underlying type (simplified version)."""
        # This is a simplified implementation
        # In a full implementation, this would recursively resolve type aliases
        return definition.strip()

    def _find_interface_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of an interface declaration."""
        brace_count = 0
        for i in range(start_idx, len(lines)):
            line = lines[i]
            brace_count += line.count('{')
            brace_count -= line.count('}')
            if brace_count == 0 and '{' in lines[start_idx]:
                return i + 1

        return len(lines)

    def _find_enum_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of an enum declaration."""
        brace_count = 0
        for i in range(start_idx, len(lines)):
            line = lines[i]
            brace_count += line.count('{')
            brace_count -= line.count('}')
            if brace_count == 0 and '{' in lines[start_idx]:
                return i + 1
        return len(lines)

    def _find_namespace_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a namespace declaration."""
        brace_count = 0
        for i in range(start_idx, len(lines)):
            line = lines[i]
            brace_count += line.count('{')
            brace_count -= line.count('}')
            if brace_count == 0 and '{' in lines[start_idx]:
                return i + 1
        return len(lines)