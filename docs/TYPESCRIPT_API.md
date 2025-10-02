# TypeScript API Documentation

This document provides comprehensive API documentation for CodeFlow's TypeScript-specific functionality, including class references, method signatures, and integration points.

## Core Classes

### TypeScriptASTVisitor

The main AST visitor class that extracts code elements from TypeScript source code using sophisticated regex-based parsing.

#### Constructor

```python
visitor = TypeScriptASTVisitor()
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `elements` | `List[CodeElement]` | List of extracted code elements |
| `current_class` | `Optional[str]` | Currently parsed class name |
| `current_file` | `str` | Currently parsed file path |
| `source_lines` | `List[str]` | Source code lines being parsed |
| `file_level_imports` | `Dict[str, str]` | File-level import mappings |
| `file_level_import_from_targets` | `Set[str]` | Direct import targets |
| `regex_patterns` | `Dict[str, Any]` | Pre-compiled regex patterns for parsing |

#### Methods

##### Regex Pattern Compilation

```python
def _compile_regex_patterns(self) -> Dict[str, Any]:
    """Pre-compile all regex patterns for maximum performance."""
    patterns = {}
    # Compile function patterns, class patterns, etc.
    print(f"Pre-compiled {len(patterns)} regex patterns")
    return patterns

def _find_functions_regex(self, source: str) -> List[Dict[str, Any]]:
    """Find function definitions using pre-compiled regex patterns."""
    functions = []
    # Use compiled patterns for optimal performance
    print(f"Found {len(functions)} functions using regex patterns")
    return functions
```

##### Parsing Methods

```python
def _parse_typescript_ast(self, file_path: str, source: str) -> List[CodeElement]:
    """Parse TypeScript AST using regex-based parsing."""
    elements = self._parse_with_regex(file_path, source)
    print(f"Extracted {len(elements)} code elements")
    return elements

def _parse_with_regex(self, file_path: str, source: str) -> List[CodeElement]:
    """Parse using sophisticated regex patterns for TypeScript analysis."""
    elements = []
    # Extract functions, classes, interfaces, etc.
    return elements
```

##### Function Detection

```python
def _find_functions_regex(self, source: str) -> List[Dict[str, Any]]:
    """Find function definitions using regex patterns."""
    functions = self._find_functions_regex(source)
    return [
        {
            'name': 'functionName',
            'params': 'param1: Type1, param2: Type2',
            'return_type': 'ReturnType',
            'is_async': False,
            'is_static': False,
            'is_abstract': False,
            'start_line': 10,
            'match_text': 'function functionName(...): ReturnType { ... }'
        }
    ]

def _create_function_element(self, func_info: Dict[str, Any], lines: List[str], file_path: str) -> Optional[FunctionElement]:
    """Create FunctionElement from regex match information."""
    return FunctionElement(
        name=func_info['name'],
        kind='function',
        file_path=file_path,
        line_start=func_info['start_line'],
        line_end=func_info.get('end_line', func_info['start_line']),
        full_source='\n'.join(lines),
        parameters=self._extract_typescript_parameters(func_info.get('match_text', '')),
        return_type=func_info.get('return_type'),
        is_async=func_info.get('is_async', False),
        is_static=func_info.get('is_static', False),
        access_modifier=func_info.get('access_modifier'),
        docstring=self._extract_jsdoc_comment(lines, func_info['start_line'] - 1),
        is_method=self.current_class is not None,
        class_name=self.current_class,
        complexity=self._calculate_typescript_complexity(func_source),
        nloc=self._calculate_nloc(start_line, end_line, lines),
        external_dependencies=self._extract_typescript_dependencies(func_source),
        decorators=framework_patterns.get('decorators', []),
        catches_exceptions=[],
        local_variables_declared=[],
        hash_body=self._hash_source_snippet(start_line, end_line, lines),
        metadata={'framework': framework_patterns.get('framework')}
    )
```

##### Class Detection

```python
def _find_classes_regex(self, source: str) -> List[Dict[str, Any]]:
    """Find class definitions using regex patterns."""
    classes = self._find_classes_regex(source)
    return [
        {
            'name': 'ClassName',
            'extends': 'BaseClass',
            'implements': ['Interface1', 'Interface2'],
            'decorators': [{'name': '@Component', 'args': []}],
            'start_line': 5,
            'match_text': 'class ClassName extends BaseClass implements Interface1, Interface2 { ... }'
        }
    ]

def _create_class_element(self, class_info: Dict[str, Any], lines: List[str], file_path: str) -> Optional[ClassElement]:
    """Create ClassElement from regex match information."""
    return ClassElement(
        name=class_info['name'],
        kind='class',
        file_path=file_path,
        line_start=class_info['start_line'],
        line_end=self._find_closing_brace(lines, class_info['start_line'] - 1),
        full_source='\n'.join(lines),
        methods=[],
        attributes=[],
        extends=class_info.get('extends'),
        implements=class_info.get('implements', []),
        docstring=self._extract_jsdoc_comment(lines, class_info['start_line'] - 1),
        decorators=class_info.get('decorators', []),
        hash_body=self._hash_source_snippet(start_line, end_line, lines),
        metadata={'framework': framework_patterns.get('framework')}
    )
```

##### TypeScript-Specific Extraction

```python
def _extract_jsdoc_comment(self, lines: List[str], start_idx: int) -> Optional[str]:
    """Extract JSDoc comments preceding a function or class."""
    return "/**\n * Function description\n * @param param1 Description\n * @returns Description\n */"

def _extract_typescript_parameters(self, func_source: str) -> List[str]:
    """Extract TypeScript function parameters with types."""
    return ["param1: string", "param2: number", "param3?: boolean"]

def _calculate_typescript_complexity(self, func_source: str) -> int:
    """Calculate cyclomatic complexity for TypeScript functions."""
    return 3  # Example: if + while + function itself

def _calculate_nloc(self, start_line: int, end_line: int, lines: List[str]) -> int:
    """Calculate Non-Comment Lines of Code for TypeScript."""
    return 15  # Example: 15 lines of actual code

def _extract_typescript_dependencies(self, func_source: str) -> List[str]:
    """Extract external dependencies from TypeScript function."""
    return ['react', 'lodash', 'axios']

def _extract_file_imports(self, source: str) -> None:
    """Extract file-level imports from TypeScript source."""
    self.file_level_imports = {
        'React': 'react',
        'Component': 'react',
        'useState': 'react'
    }
    self.file_level_import_from_targets = {'React', 'Component', 'useState'}
```

##### Framework Detection

```python
def _detect_framework_patterns(self, source: str) -> Dict[str, Any]:
    """Detect TypeScript framework patterns and decorators."""
    return {
        'decorators': [
            {'name': '@Component', 'framework': 'angular'},
            {'name': '@Injectable', 'framework': 'angular'}
        ],
        'framework': 'angular',
        'features': ['components', 'dependency_injection']
    }
```

##### Complex Type Parsing

```python
def _parse_complex_types(self, source: str) -> Dict[str, Any]:
    """Parse complex TypeScript type definitions."""
    return {
        'union_types': ['string | number | null'],
        'intersection_types': ['Type1 & Type2'],
        'mapped_types': ['{[K in keyof T]: T[K]}'],
        'conditional_types': ['T extends U ? X : Y'],
        'utility_types': ['Partial<T>', 'Required<T>'],
        'type_aliases': [{'name': 'UserId', 'definition': 'string', 'line': 10}],
        'interfaces': [{'name': 'User', 'extends': [], 'implements': [], 'line': 5}],
        'enums': [{'name': 'Status', 'members': ['Active', 'Inactive'], 'line': 15}],
        'namespaces': []
    }
```

##### Utility Methods

```python
def _find_closing_brace(self, lines: List[str], start_idx: int) -> int:
    """Find the matching closing brace for a given opening brace."""
    return 25  # Example: closes at line 25

def _hash_source_snippet(self, start_line: int, end_line: int, lines: List[str]) -> str:
    """Generate MD5 hash of TypeScript source snippet."""
    return "a1b2c3d4e5f6..."  # Example hash

def visit_file(self, file_path: str, source: str) -> List[CodeElement]:
    """Visit a TypeScript file and extract all code elements."""
    elements = self.visit_file(file_path, source)
    print(f"Extracted {len(elements)} elements from {file_path}")
    return elements
```

### TypeScriptASTExtractor

The main extractor class that handles file processing, project integration, and TypeScript configuration management.

#### Constructor

```python
extractor = TypeScriptASTExtractor()
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `visitor` | `TypeScriptASTVisitor` | The underlying AST visitor |
| `project_root` | `Optional[Path]` | Project root directory |
| `tsconfig` | `Optional[Dict[str, Any]]` | Parsed tsconfig.json |
| `project_references` | `List[Dict[str, Any]]` | TypeScript project references |
| `path_mappings` | `Dict[str, List[str]]` | TypeScript path mappings |

#### Methods

##### Project Configuration

```python
def _find_tsconfig(self, directory: Path) -> Optional[Path]:
    """Find tsconfig.json file in directory or parent directories."""
    return Path("/path/to/project/tsconfig.json")

def _parse_tsconfig(self, tsconfig_path: Path) -> Dict[str, Any]:
    """Parse TypeScript configuration file."""
    return {
        'compilerOptions': {
            'target': 'ES2020',
            'module': 'commonjs',
            'strict': True
        },
        'include': ['src/**/*'],
        'exclude': ['node_modules', 'dist']
    }

def _setup_project_integration(self, directory: Path) -> None:
    """Setup TypeScript project integration by parsing tsconfig.json."""
    self._setup_project_integration(directory)
    if self.tsconfig:
        print(f"Loaded TypeScript configuration from {directory}/tsconfig.json")

def _get_compiler_options(self) -> Dict[str, Any]:
    """Get TypeScript compiler options with defaults."""
    return {
        'target': 'ES2020',
        'module': 'commonjs',
        'strict': True,
        'esModuleInterop': True
    }
```

##### Path Resolution

```python
def _resolve_path_mapping(self, import_path: str) -> str:
    """Resolve import path using TypeScript path mappings."""
    return import_path  # Resolved path

def _build_interface_hierarchy(self) -> Dict[str, List[str]]:
    """Build interface inheritance hierarchy from extracted interface elements."""
    return {
        'User': ['BaseEntity'],
        'Admin': ['User', 'BaseEntity']
    }

def _resolve_type_aliases(self) -> Dict[str, str]:
    """Resolve type aliases to their actual type definitions."""
    return {
        'UserId': 'string',
        'UserRole': 'admin | user | guest'
    }
```

##### File Processing

```python
def extract_from_file(self, file_path: Path) -> List[CodeElement]:
    """Extract code elements from a single TypeScript file."""
    elements = self.extract_from_file(file_path)
    print(f"Processed {file_path}: {len(elements)} elements")
    return elements

def extract_from_directory(self, directory: Path) -> List[CodeElement]:
    """Extract code elements from all TypeScript files in directory."""
    elements = self.extract_from_directory(directory)
    print(f"Total extracted: {len(elements)} code elements")
    return elements

def _extract_typescript_specific_elements(self, directory: Path, ts_files: List[Path]) -> List[CodeElement]:
    """Extract TypeScript-specific elements like interfaces, enums, and namespaces."""
    return []
```

##### Element Creation

```python
def _create_interface_element(self, interface_info: Dict[str, Any], source: str, file_path: Path) -> Optional[ClassElement]:
    """Create a ClassElement representing a TypeScript interface."""
    return ClassElement(
        name=interface_info['name'],
        kind='interface',
        file_path=str(file_path),
        line_start=interface_info['line'],
        line_end=self._find_interface_end(source.splitlines(), interface_info['line'] - 1),
        full_source=source,
        methods=[],
        attributes=[],
        extends=interface_info.get('extends', []),
        implements=interface_info.get('implements', []),
        docstring=self.visitor._extract_jsdoc_comment(source.splitlines(), interface_info['line'] - 1),
        decorators=[],
        hash_body=self.visitor._hash_source_snippet(start_line, end_line, source.splitlines()),
        metadata={'typescript_kind': 'interface'}
    )

def _create_enum_element(self, enum_info: Dict[str, Any], source: str, file_path: Path) -> Optional[ClassElement]:
    """Create a ClassElement representing a TypeScript enum."""
    return ClassElement(
        name=enum_info['name'],
        kind='enum',
        file_path=str(file_path),
        line_start=enum_info['line'],
        line_end=self._find_enum_end(source.splitlines(), enum_info['line'] - 1),
        full_source=source,
        methods=[],
        attributes=enum_info.get('members', []),
        extends=None,
        implements=[],
        docstring=self.visitor._extract_jsdoc_comment(source.splitlines(), enum_info['line'] - 1),
        decorators=[],
        hash_body=self.visitor._hash_source_snippet(start_line, end_line, source.splitlines()),
        metadata={'typescript_kind': 'enum'}
    )

def _create_namespace_element(self, namespace_info: Dict[str, Any], source: str, file_path: Path) -> Optional[ClassElement]:
    """Create a ClassElement representing a TypeScript namespace."""
    return ClassElement(
        name=namespace_info['name'],
        kind='namespace',
        file_path=str(file_path),
        line_start=namespace_info['line'],
        line_end=self._find_namespace_end(source.splitlines(), namespace_info['line'] - 1),
        full_source=source,
        methods=[],
        attributes=[],
        extends=None,
        implements=[],
        docstring=self.visitor._extract_jsdoc_comment(source.splitlines(), namespace_info['line'] - 1),
        decorators=[],
        hash_body=self.visitor._hash_source_snippet(start_line, end_line, source.splitlines()),
        metadata={'typescript_kind': 'namespace'}
    )

def _create_type_alias_element(self, type_alias_info: Dict[str, Any], source: str, file_path: Path) -> Optional[CodeElement]:
    """Create a CodeElement representing a TypeScript type alias."""
    return CodeElement(
        name=type_alias_info['name'],
        kind='type_alias',
        file_path=str(file_path),
        line_start=type_alias_info['line'],
        line_end=type_alias_info['line'],
        full_source=source,
        metadata={
            'typescript_kind': 'type_alias',
            'type_definition': type_alias_info.get('definition', ''),
            'resolved_type': self._resolve_type_alias(type_alias_info.get('definition', ''))
        }
    )
```

## Data Structures

### TypeScript-Specific Metadata

#### FunctionElement Metadata

```python
function_element = FunctionElement(
    # ... standard fields ...
    metadata={
        'framework': 'angular',  # Detected framework
        'typescript_features': [  # TypeScript-specific features
            'generics',
            'async_await',
            'decorators'
        ],
        'type_parameters': ['T', 'U'],  # Generic type parameters
        'overloads': 2,  # Number of function overloads
        'accessibility': 'public'  # Access modifier
    }
)
```

#### ClassElement Metadata

```python
class_element = ClassElement(
    # ... standard fields ...
    metadata={
        'framework': 'nestjs',  # Detected framework
        'typescript_kind': 'class',  # TypeScript-specific element type
        'implements_interfaces': ['Injectable', 'OnModuleInit'],
        'abstract': False,  # Whether class is abstract
        'generic_base': False,  # Whether extends a generic class
        'decorator_args': {  # Decorator arguments
            '@Controller': ['users'],
            '@Injectable': []
        }
    }
)
```

### Supported TypeScript Features

#### Type Annotations

| Type Category | Examples | Support |
|---------------|----------|---------|
| Primitive Types | `string`, `number`, `boolean` | ✅ Full |
| Object Types | `object`, `Record<K,V>`, `any` | ✅ Full |
| Array Types | `string[]`, `Array<T>` | ✅ Full |
| Union Types | `string \| number \| null` | ✅ Full |
| Intersection Types | `Type1 & Type2` | ✅ Full |
| Function Types | `(param: T) => U` | ✅ Full |
| Generic Types | `Map<K, V>`, `Promise<T>` | ✅ Full |
| Utility Types | `Partial<T>`, `Required<T>` | ✅ Full |
| Conditional Types | `T extends U ? X : Y` | ✅ Full |

#### Decorators

| Framework | Decorators | Detection | Metadata |
|-----------|------------|-----------|----------|
| Angular | `@Component`, `@Directive`, `@Pipe` | ✅ Full | Arguments, options |
| NestJS | `@Controller`, `@Injectable`, `@Module` | ✅ Full | HTTP methods, paths |
| Custom | Any custom decorators | ✅ Pattern | Name, arguments |

#### Interfaces and Types

| Element | Parsing | Hierarchy | Resolution |
|---------|---------|-----------|------------|
| Interfaces | ✅ Regex + Compiler | ✅ Extends chains | ✅ Partial |
| Type Aliases | ✅ Full parsing | ❌ N/A | ✅ Basic |
| Enums | ✅ Full parsing | ❌ N/A | ✅ Members |
| Namespaces | ✅ Structure | ✅ Internal | ❌ External |

## Integration Points

### CallGraphBuilder Integration

```python
from core.call_graph_builder import CallGraphBuilder

# TypeScript elements work seamlessly with call graph builder
elements = typescript_extractor.extract_from_directory(project_path)
graph_builder = CallGraphBuilder()
graph_builder.build_from_elements(elements)

# Access TypeScript-specific information
for node in graph_builder.functions.values():
    if hasattr(node, 'metadata') and node.metadata.get('framework'):
        framework = node.metadata['framework']
        print(f"Function {node.name} uses {framework}")
```

### VectorStore Integration

```python
from core.vector_store import CodeVectorStore

# TypeScript elements are stored with full metadata
vector_store = CodeVectorStore()
elements = typescript_extractor.extract_from_directory(project_path)

# TypeScript-specific queries work with semantic search
results = vector_store.query_functions("Angular components with forms", n_results=10)

for result in results:
    metadata = result['metadata']
    if metadata.get('framework') == 'angular':
        print(f"Found Angular component: {metadata['name']}")
```

### MCP Server Integration

```python
# TypeScript analysis available through MCP tools
{
    "tool": "semantic_search",
    "input": {
        "query": "TypeScript functions with async/await",
        "filters": {"language": "typescript"},
        "n_results": 5
    }
}

# Framework-specific queries
{
    "tool": "get_function_metadata",
    "input": {
        "fqn": "myapp.UserService.createUser",
        "include_typescript_info": true
    }
}
```

## Usage Examples

### Basic TypeScript Analysis

```python
from code_flow_graph.core.ast_extractor import TypeScriptASTExtractor

# Initialize extractor
extractor = TypeScriptASTExtractor()

# Extract from single file
elements = extractor.extract_from_file(Path("src/component.ts"))
print(f"Extracted {len(elements)} elements")

# Extract from entire project
project_elements = extractor.extract_from_directory(Path("./my-ts-project"))
print(f"Project total: {len(project_elements)} elements")
```

### Framework-Specific Analysis

```python
# Analyze Angular project
angular_elements = extractor.extract_from_directory(Path("./angular-app"))

# Filter Angular components
angular_components = [
    elem for elem in angular_elements
    if elem.metadata.get('framework') == 'angular' and
       elem.metadata.get('decorators', {}).get('Component')
]

print(f"Found {len(angular_components)} Angular components")
```

### Type System Analysis

```python
# Extract TypeScript type information
typescript_elements = extractor.extract_from_directory(Path("./ts-project"))

# Find interfaces
interfaces = [elem for elem in typescript_elements if elem.kind == 'interface']
print(f"Found {len(interfaces)} interfaces")

# Find enums
enums = [elem for elem in typescript_elements if elem.kind == 'enum']
print(f"Found {len(enums)} enums")

# Find type aliases
type_aliases = [elem for elem in typescript_elements if elem.kind == 'type_alias']
print(f"Found {len(type_aliases)} type aliases")
```

## Error Handling

### TypeScript Analysis

```python
try:
    elements = extractor.extract_from_directory(project_path)
    print("Analysis completed successfully using regex-based parsing")
except Exception as e:
    print(f"Analysis failed: {e}")
    print("Note: CodeFlow uses regex-based parsing - no external dependencies required")
```

### Malformed TypeScript

```python
# The extractor gracefully handles syntax errors
try:
    elements = extractor.extract_from_file(malformed_file)
    print(f"Extracted {len(elements)} elements despite syntax issues")
except SyntaxError as e:
    print(f"Syntax error in {malformed_file}: {e}")
```

## Performance Considerations

### Large Projects

```python
# For large TypeScript projects, consider:
# 1. Use project-specific vector stores
vector_store = CodeVectorStore(persist_directory="./ts_project_vectors")

# 2. Process files incrementally
for file_path in Path("./src").rglob("*.ts"):
    elements = extractor.extract_from_file(file_path)
    # Process in batches
```

### Memory Management

```python
# TypeScript parsing can be memory-intensive for large files
# Consider processing files individually for large projects
import gc

elements = []
for file_path in ts_files:
    file_elements = extractor.extract_from_file(file_path)
    elements.extend(file_elements)
    gc.collect()  # Free memory after each file
```

This API documentation provides comprehensive information for integrating with and extending CodeFlow's TypeScript functionality, from basic usage to advanced framework-specific analysis.