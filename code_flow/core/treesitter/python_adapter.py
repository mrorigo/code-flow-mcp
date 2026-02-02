"""
Tree-sitter adapter for Python source.

Produces CodeElement / FunctionElement / ClassElement objects from Tree-sitter nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from tree_sitter import Node, Tree

from ..models import ClassElement, CodeElement, FunctionElement
from ..utils import calculate_nloc_python, hash_source_snippet


@dataclass
class PythonImportContext:
    file_imports: Dict[str, str]
    import_from_targets: Set[str]


def extract_elements(tree: Tree, source: str, file_path: str) -> List[CodeElement]:
    lines = source.splitlines()
    import_context = _extract_file_imports(tree.root_node, source)
    elements: List[CodeElement] = []
    _walk_module(tree.root_node, source, file_path, lines, import_context, elements, current_class=None)
    return elements


def _walk_module(
    node: Node,
    source: str,
    file_path: str,
    lines: List[str],
    import_context: PythonImportContext,
    elements: List[CodeElement],
    current_class: Optional[str],
) -> None:
    for child in node.children:
        if child.type in {"function_definition", "async_function_definition"}:
            func = _build_function(child, source, file_path, lines, import_context, current_class)
            elements.append(func)
            # Walk nested functions
            _walk_module(child, source, file_path, lines, import_context, elements, current_class=current_class)
        elif child.type == "class_definition":
            class_element = _build_class(child, source, file_path, lines)
            elements.append(class_element)
            # Walk class body to add methods
            _walk_class_body(child, source, file_path, lines, import_context, elements, class_element.name)
        else:
            # Recurse to find nested defs
            _walk_module(child, source, file_path, lines, import_context, elements, current_class=current_class)


def _walk_class_body(
    class_node: Node,
    source: str,
    file_path: str,
    lines: List[str],
    import_context: PythonImportContext,
    elements: List[CodeElement],
    class_name: str,
) -> None:
    body = class_node.child_by_field_name("body")
    if not body:
        return
    for child in body.children:
        if child.type in {"function_definition", "async_function_definition"}:
            func = _build_function(child, source, file_path, lines, import_context, class_name)
            elements.append(func)


def _build_function(
    node: Node,
    source: str,
    file_path: str,
    lines: List[str],
    import_context: PythonImportContext,
    current_class: Optional[str],
) -> FunctionElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(source, name_node) if name_node else "<anonymous>"

    start_line, end_line = _line_span(node)
    parameters = _extract_parameters(node, source)
    return_type = _extract_return_type(node, source)
    is_async = node.type == "async_function_definition"
    docstring = _extract_docstring(node, source)
    decorators = _extract_decorators(node, source)

    # Best-effort metrics from Tree-sitter node kinds
    complexity = _calculate_complexity(node)
    nloc = calculate_nloc_python(lines, start_line, end_line)
    hash_body = hash_source_snippet(lines, start_line, end_line)

    external_dependencies = _extract_external_dependencies(node, source, import_context)
    catches_exceptions = _extract_exception_handlers(node, source)
    local_variables = _extract_local_variables(node, source)

    return FunctionElement(
        name=name,
        kind="function",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        parameters=parameters,
        return_type=return_type,
        is_async=is_async,
        docstring=docstring,
        is_method=current_class is not None,
        class_name=current_class,
        complexity=complexity,
        nloc=nloc,
        external_dependencies=external_dependencies,
        decorators=decorators,
        catches_exceptions=catches_exceptions,
        local_variables_declared=local_variables,
        hash_body=hash_body,
        metadata={},
    )


def _build_class(node: Node, source: str, file_path: str, lines: List[str]) -> ClassElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(source, name_node) if name_node else "<anonymous>"
    start_line, end_line = _line_span(node)
    decorators = _extract_decorators(node, source)
    docstring = _extract_docstring(node, source)
    methods = _extract_class_methods(node, source)
    attributes = _extract_class_attributes(node, source)
    extends = _extract_class_extends(node, source)
    hash_body = hash_source_snippet(lines, start_line, end_line)

    return ClassElement(
        name=name,
        kind="class",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        methods=methods,
        attributes=attributes,
        extends=extends,
        docstring=docstring,
        decorators=decorators,
        hash_body=hash_body,
        metadata={},
    )


def _extract_parameters(node: Node, source: str) -> List[str]:
    params_node = node.child_by_field_name("parameters")
    if not params_node:
        return []
    params: List[str] = []
    for child in params_node.children:
        if child.type in {"identifier", "typed_parameter", "default_parameter", "list_splat", "dictionary_splat"}:
            params.append(_node_text(source, child))
    return [p for p in params if p and p not in ["(", ")", ","]]


def _extract_return_type(node: Node, source: str) -> Optional[str]:
    return_node = node.child_by_field_name("return_type")
    return _node_text(source, return_node) if return_node else None


def _extract_docstring(node: Node, source: str) -> Optional[str]:
    body = node.child_by_field_name("body")
    if not body or not body.children:
        return None
    first_stmt = body.children[0]
    if first_stmt.type == "expression_statement":
        string_node = first_stmt.children[0] if first_stmt.children else None
        if string_node and string_node.type == "string":
            text = _node_text(source, string_node)
            return _strip_quotes(text)
    return None


def _extract_decorators(node: Node, source: str) -> List[Dict[str, Any]]:
    decorators: List[Dict[str, Any]] = []
    for child in node.children:
        if child.type == "decorator":
            name = _node_text(source, child.child_by_field_name("name")) or _node_text(source, child)
            args = _extract_decorator_args(child, source)
            decorators.append({"name": name, "args": args.get("args", []), "kwargs": args.get("kwargs", {})})
    return decorators


def _extract_decorator_args(node: Node, source: str) -> Dict[str, object]:
    args_node = node.child_by_field_name("arguments")
    if not args_node:
        return {"args": [], "kwargs": {}}
    args: List[str] = []
    kwargs: Dict[str, str] = {}
    for child in args_node.children:
        if child.type == "keyword_argument":
            key_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if key_node and value_node:
                kwargs[_node_text(source, key_node)] = _node_text(source, value_node)
        elif child.type not in {",", "(", ")"}:
            args.append(_node_text(source, child))
    return {"args": args, "kwargs": kwargs}


def _extract_class_methods(node: Node, source: str) -> List[str]:
    body = node.child_by_field_name("body")
    if not body:
        return []
    methods: List[str] = []
    for child in body.children:
        if child.type == "function_definition":
            name_node = child.child_by_field_name("name")
            if name_node:
                methods.append(_node_text(source, name_node))
    return methods


def _extract_class_attributes(node: Node, source: str) -> List[str]:
    body = node.child_by_field_name("body")
    if not body:
        return []
    attributes: Set[str] = set()
    for child in body.children:
        if child.type in {"assignment", "annotated_assignment"}:
            target = child.child_by_field_name("left") or child.child_by_field_name("target")
            if target and target.type == "identifier":
                attributes.add(_node_text(source, target))
        if child.type == "function_definition":
            if _node_text(source, child.child_by_field_name("name")) == "__init__":
                for sub in child.children:
                    if sub.type == "assignment":
                        target = sub.child_by_field_name("left")
                        if target and target.type == "attribute":
                            attr_name = target.child_by_field_name("attribute")
                            if attr_name:
                                attributes.add(_node_text(source, attr_name))
    return sorted(attributes)


def _extract_class_extends(node: Node, source: str) -> Optional[str]:
    bases = node.child_by_field_name("superclasses")
    if not bases:
        return None
    for child in bases.children:
        if child.type == "identifier":
            return _node_text(source, child)
    return None


def _extract_external_dependencies(
    node: Node,
    source: str,
    import_context: PythonImportContext,
) -> List[str]:
    dependencies: Set[str] = set()
    for child in _walk(node):
        if child.type == "identifier":
            name = _node_text(source, child)
            if name in import_context.file_imports:
                module = import_context.file_imports[name]
                module_root = module.split(".")[0].lstrip(".")
                if module_root:
                    dependencies.add(module_root)
            elif name in import_context.import_from_targets:
                module = import_context.file_imports.get(name, name)
                module_root = module.split(".")[0].lstrip(".")
                if module_root:
                    dependencies.add(module_root)
        if child.type == "attribute":
            value = child.child_by_field_name("object")
            if value and value.type == "identifier":
                base = _node_text(source, value)
                if base in import_context.file_imports:
                    module = import_context.file_imports[base]
                    module_root = module.split(".")[0].lstrip(".")
                    if module_root:
                        dependencies.add(module_root)
    return sorted(dependencies)


def _extract_exception_handlers(node: Node, source: str) -> List[str]:
    exceptions: Set[str] = set()
    for child in _walk(node):
        if child.type == "except_clause":
            exc_node = child.child_by_field_name("type")
            if exc_node:
                exceptions.add(_node_text(source, exc_node))
            else:
                exceptions.add("Exception")
    return sorted(exceptions)


def _extract_local_variables(node: Node, source: str) -> List[str]:
    locals_set: Set[str] = set()
    for child in _walk(node):
        if child.type in {"assignment", "augmented_assignment", "annotated_assignment"}:
            target = child.child_by_field_name("left") or child.child_by_field_name("target")
            if target and target.type == "identifier":
                locals_set.add(_node_text(source, target))
        if child.type in {"for_statement", "for_in_statement"}:
            target = child.child_by_field_name("left")
            if target and target.type == "identifier":
                locals_set.add(_node_text(source, target))
    return sorted(locals_set - {"self", "cls"})


def _extract_file_imports(root: Node, source: str) -> PythonImportContext:
    import re

    file_imports: Dict[str, str] = {}
    import_from_targets: Set[str] = set()
    for child in root.children:
        if child.type == "import_statement":
            statement = _node_text(source, child)
            for match in re.finditer(r"import\s+([\w\.]+)(?:\s+as\s+(\w+))?", statement):
                module = match.group(1)
                alias = match.group(2) or module.split(".")[-1]
                file_imports[alias] = module
        if child.type == "import_from_statement":
            statement = _node_text(source, child)
            module_match = re.search(r"from\s+([\w\.]+)\s+import\s+(.+)", statement)
            if not module_match:
                continue
            module_name = module_match.group(1)
            imports_part = module_match.group(2)
            for item in imports_part.split(','):
                name_alias = item.strip()
                if not name_alias:
                    continue
                if " as " in name_alias:
                    name, alias = [part.strip() for part in name_alias.split(" as ", 1)]
                else:
                    name, alias = name_alias, name_alias
                file_imports[alias] = f"{module_name}.{name}"
                import_from_targets.add(alias)
    return PythonImportContext(file_imports=file_imports, import_from_targets=import_from_targets)


def _line_span(node: Node) -> Tuple[int, int]:
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    return start_line, end_line


def _node_text(source: str, node: Optional[Node]) -> str:
    if not node:
        return ""
    source_bytes = source.encode("utf-8")
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


def _strip_quotes(text: str) -> str:
    if text.startswith(('"""', "'''")) and text.endswith(('"""', "'''")):
        return text[3:-3]
    if text.startswith(("\"", "'")) and text.endswith(("\"", "'")):
        return text[1:-1]
    return text


def _walk(node: Node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _calculate_complexity(node: Node) -> int:
    complexity = 1
    for child in _walk(node):
        if child.type in {
            "if_statement",
            "elif_clause",
            "for_statement",
            "while_statement",
            "try_statement",
            "except_clause",
            "with_statement",
            "comprehension",
        }:
            complexity += 1
    return complexity
