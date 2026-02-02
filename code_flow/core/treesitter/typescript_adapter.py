"""
Tree-sitter adapter for TypeScript/TSX source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from tree_sitter import Node, Tree

from ..models import ClassElement, CodeElement, FunctionElement
from ..utils import (
    calculate_complexity_typescript,
    calculate_nloc_typescript,
    hash_source_snippet,
)


@dataclass
class TypeScriptImportContext:
    file_imports: Dict[str, str]
    import_from_targets: Set[str]


def extract_elements(tree: Tree, source: str, file_path: str) -> List[CodeElement]:
    lines = source.splitlines()
    import_context = _extract_file_imports(tree.root_node, source)
    elements: List[CodeElement] = []

    _walk_root(tree.root_node, source, file_path, lines, import_context, elements)
    return elements


def _walk_root(
    node: Node,
    source: str,
    file_path: str,
    lines: List[str],
    import_context: TypeScriptImportContext,
    elements: List[CodeElement],
) -> None:
    for child in node.children:
        if child.type == "expression_statement":
            elements.extend(_extract_route_handlers(child, source, file_path, lines, import_context))
        if child.type == "lexical_declaration":
            elements.extend(_extract_variable_functions(child, source, file_path, lines, import_context))
        if child.type == "function_declaration":
            elements.append(_build_function(child, source, file_path, lines, import_context))
        elif child.type == "class_declaration":
            elements.append(_build_class(child, source, file_path, lines))
            _walk_class_body(child, source, file_path, lines, import_context, elements)
        elif child.type == "interface_declaration":
            elements.append(_build_interface(child, source, file_path, lines))
        elif child.type == "enum_declaration":
            elements.append(_build_enum(child, source, file_path, lines))
        elif child.type == "type_alias_declaration":
            elements.append(_build_type_alias(child, source, file_path, lines))
        else:
            _walk_root(child, source, file_path, lines, import_context, elements)


def _walk_class_body(
    class_node: Node,
    source: str,
    file_path: str,
    lines: List[str],
    import_context: TypeScriptImportContext,
    elements: List[CodeElement],
) -> None:
    body = class_node.child_by_field_name("body")
    if not body:
        return
    class_name = _node_text(source, class_node.child_by_field_name("name"))
    for child in body.children:
        if child.type in {"method_definition", "public_field_definition"}:
            func = _build_method(child, source, file_path, lines, import_context, class_name)
            if func:
                elements.append(func)


def _build_function(
    node: Node,
    source: str,
    file_path: str,
    lines: List[str],
    import_context: TypeScriptImportContext,
) -> FunctionElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(source, name_node) if name_node else "<anonymous>"
    start_line, end_line = _line_span(node)
    params = _extract_parameters(node, source)
    return_type = _extract_return_type(node, source)
    docstring = _extract_jsdoc(node, source)
    decorators = _extract_decorators(node, source)

    func_source = _slice_source(lines, start_line, end_line)
    complexity = calculate_complexity_typescript(func_source)
    nloc = calculate_nloc_typescript(lines, start_line, end_line)
    external_deps = _extract_external_dependencies(node, source, import_context)
    hash_body = hash_source_snippet(lines, start_line, end_line)
    metadata = _build_typescript_metadata(node, source)

    return FunctionElement(
        name=name,
        kind="function",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        parameters=params,
        return_type=return_type,
        is_async=_is_async(node),
        docstring=docstring,
        is_method=False,
        class_name=None,
        complexity=complexity,
        nloc=nloc,
        external_dependencies=external_deps,
        decorators=decorators,
        hash_body=hash_body,
        metadata=metadata,
    )


def _build_method(
    node: Node,
    source: str,
    file_path: str,
    lines: List[str],
    import_context: TypeScriptImportContext,
    class_name: str,
) -> Optional[FunctionElement]:
    if node.type == "public_field_definition":
        value = node.child_by_field_name("value")
        if not value or value.type != "arrow_function":
            return None
        func_node = value
        name_node = node.child_by_field_name("name")
    else:
        func_node = node
        name_node = node.child_by_field_name("name")

    name = _node_text(source, name_node) if name_node else "<anonymous>"
    start_line, end_line = _line_span(node)
    params = _extract_parameters(func_node, source)
    return_type = _extract_return_type(func_node, source)
    docstring = _extract_jsdoc(node, source)
    decorators = _extract_decorators(node, source)
    func_source = _slice_source(lines, start_line, end_line)
    complexity = calculate_complexity_typescript(func_source)
    nloc = calculate_nloc_typescript(lines, start_line, end_line)
    external_deps = _extract_external_dependencies(node, source, import_context)
    hash_body = hash_source_snippet(lines, start_line, end_line)
    metadata = _build_typescript_metadata(node, source)

    return FunctionElement(
        name=name,
        kind="function",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        parameters=params,
        return_type=return_type,
        is_async=_is_async(node),
        docstring=docstring,
        is_method=True,
        class_name=class_name,
        complexity=complexity,
        nloc=nloc,
        external_dependencies=external_deps,
        decorators=decorators,
        hash_body=hash_body,
        metadata=metadata,
    )


def _build_class(node: Node, source: str, file_path: str, lines: List[str]) -> ClassElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(source, name_node) if name_node else "<anonymous>"
    start_line, end_line = _line_span(node)
    decorators = _extract_decorators(node, source)
    docstring = _extract_jsdoc(node, source)
    methods = _extract_class_methods(node, source)
    implements, extends = _extract_heritage(node, source)
    hash_body = hash_source_snippet(lines, start_line, end_line)

    metadata = _build_typescript_metadata(node, source)
    return ClassElement(
        name=name,
        kind="class",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        methods=methods,
        attributes=[],
        extends=extends,
        implements=implements,
        docstring=docstring,
        decorators=decorators,
        hash_body=hash_body,
        metadata=metadata,
    )


def _build_interface(node: Node, source: str, file_path: str, lines: List[str]) -> ClassElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(source, name_node) if name_node else "<anonymous>"
    start_line, end_line = _line_span(node)
    hash_body = hash_source_snippet(lines, start_line, end_line)
    return ClassElement(
        name=name,
        kind="interface",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        methods=[],
        attributes=[],
        extends=None,
        implements=[],
        docstring=_extract_jsdoc(node, source),
        decorators=[],
        hash_body=hash_body,
        metadata={"typescript_kind": "interface"},
    )


def _build_enum(node: Node, source: str, file_path: str, lines: List[str]) -> ClassElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(source, name_node) if name_node else "<anonymous>"
    start_line, end_line = _line_span(node)
    hash_body = hash_source_snippet(lines, start_line, end_line)
    members = []
    body = node.child_by_field_name("body")
    if body:
        for child in body.named_children:
            if child.type == "enum_assignment" or child.type == "property_identifier":
                members.append(_node_text(source, child))
    return ClassElement(
        name=name,
        kind="enum",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        methods=[],
        attributes=members,
        extends=None,
        implements=[],
        docstring=_extract_jsdoc(node, source),
        decorators=[],
        hash_body=hash_body,
        metadata={"typescript_kind": "enum", "enum_members": members},
    )


def _build_type_alias(node: Node, source: str, file_path: str, lines: List[str]) -> CodeElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(source, name_node) if name_node else "<anonymous>"
    start_line, end_line = _line_span(node)
    definition_node = node.child_by_field_name("value")
    definition = _node_text(source, definition_node)
    hash_body = hash_source_snippet(lines, start_line, end_line)
    return CodeElement(
        name=name,
        kind="type_alias",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        metadata={"typescript_kind": "type_alias", "type_definition": definition, "hash_body": hash_body},
    )


def _extract_variable_functions(
    node: Node,
    source: str,
    file_path: str,
    lines: List[str],
    import_context: TypeScriptImportContext,
) -> List[FunctionElement]:
    functions: List[FunctionElement] = []
    for declarator in node.named_children:
        if declarator.type != "variable_declarator":
            continue
        name_node = declarator.child_by_field_name("name")
        value_node = declarator.child_by_field_name("value")
        if not value_node or value_node.type != "arrow_function":
            continue
        name = _node_text(source, name_node) if name_node else "<anonymous>"
        start_line, end_line = _line_span(declarator)
        params = _extract_parameters(value_node, source)
        return_type = _extract_return_type(value_node, source)
        docstring = _extract_jsdoc(declarator, source)
        decorators = _extract_decorators(declarator, source)
        func_source = _slice_source(lines, start_line, end_line)
        complexity = calculate_complexity_typescript(func_source)
        nloc = calculate_nloc_typescript(lines, start_line, end_line)
        external_deps = _extract_external_dependencies(declarator, source, import_context)
        hash_body = hash_source_snippet(lines, start_line, end_line)
        metadata = _build_typescript_metadata(declarator, source)
        functions.append(
            FunctionElement(
                name=name,
                kind="function",
                file_path=file_path,
                line_start=start_line,
                line_end=end_line,
                full_source=source,
                parameters=params,
                return_type=return_type,
                is_async=_is_async(value_node),
                docstring=docstring,
                is_method=False,
                class_name=None,
                complexity=complexity,
                nloc=nloc,
                external_dependencies=external_deps,
                decorators=decorators,
                hash_body=hash_body,
                metadata=metadata,
            )
        )
    return functions


def _extract_route_handlers(
    node: Node,
    source: str,
    file_path: str,
    lines: List[str],
    import_context: TypeScriptImportContext,
) -> List[FunctionElement]:
    call_node = node.child_by_field_name("expression") if node.type == "expression_statement" else node
    if not call_node or call_node.type != "call_expression":
        return []

    callee = call_node.child_by_field_name("function")
    if not callee or callee.type != "member_expression":
        return []

    object_node = callee.child_by_field_name("object")
    property_node = callee.child_by_field_name("property")
    if not object_node or not property_node:
        return []

    object_name = _node_text(source, object_node)
    method_name = _node_text(source, property_node)
    if object_name not in {"router", "app"}:
        return []

    if method_name not in {"get", "post", "put", "delete", "patch", "options", "head", "use"}:
        return []

    args_node = call_node.child_by_field_name("arguments")
    if not args_node:
        return []

    path_text = None
    handler_node = None
    for arg in args_node.named_children:
        if arg.type in {"string", "template_string"} and path_text is None:
            path_text = _node_text(source, arg)
            continue
        if arg.type in {"arrow_function", "function", "function_expression"}:
            handler_node = arg
            break
        if arg.type == "call_expression" and handler_node is None:
            nested_args = arg.child_by_field_name("arguments")
            if not nested_args:
                continue
            for nested in nested_args.named_children:
                if nested.type in {"arrow_function", "function", "function_expression"}:
                    handler_node = nested
                    break
        if handler_node:
            break

    if not handler_node:
        return []

    start_line, end_line = _line_span(handler_node)
    params = _extract_parameters(handler_node, source)
    return_type = _extract_return_type(handler_node, source)
    docstring = _extract_jsdoc(handler_node, source)
    func_source = _slice_source(lines, start_line, end_line)
    complexity = calculate_complexity_typescript(func_source)
    nloc = calculate_nloc_typescript(lines, start_line, end_line)
    external_deps = _extract_external_dependencies(handler_node, source, import_context)
    hash_body = hash_source_snippet(lines, start_line, end_line)
    metadata = _build_typescript_metadata(handler_node, source)
    metadata.setdefault("typescript_features", [])
    if "routing" not in metadata["typescript_features"]:
        metadata["typescript_features"].append("routing")
    metadata["framework"] = "express"

    handler_name = f"{object_name}.{method_name}"
    if path_text:
        handler_name = f"{handler_name}({path_text})"

    return [
        FunctionElement(
            name=handler_name,
            kind="function",
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            full_source=source,
            parameters=params,
            return_type=return_type,
            is_async=_is_async(handler_node),
            docstring=docstring,
            is_method=False,
            class_name=None,
            complexity=complexity,
            nloc=nloc,
            external_dependencies=external_deps,
            decorators=[],
            hash_body=hash_body,
            metadata=metadata,
        )
    ]


def _extract_parameters(node: Node, source: str) -> List[str]:
    params_node = node.child_by_field_name("parameters")
    if not params_node:
        return []
    params: List[str] = []
    for child in params_node.named_children:
        if child.type in {"required_parameter", "optional_parameter", "rest_parameter", "parameter"}:
            params.append(_node_text(source, child))
    return params


def _extract_return_type(node: Node, source: str) -> Optional[str]:
    return_node = node.child_by_field_name("return_type")
    return _node_text(source, return_node) if return_node else None


def _extract_decorators(node: Node, source: str) -> List[Dict[str, Any]]:
    decorators: List[Dict[str, Any]] = []

    def _add_decorator(decorator_node: Node) -> None:
        name_node = decorator_node.child_by_field_name("name")
        if name_node:
            name = _node_text(source, name_node)
        else:
            text = _node_text(source, decorator_node)
            name = text.lstrip("@").split("(")[0].strip()
        if name:
            decorators.append({"name": name})

    for child in _walk(node):
        if child.type == "decorator":
            _add_decorator(child)

    return decorators


def _extract_jsdoc(node: Node, source: str) -> Optional[str]:
    prev = node.prev_named_sibling
    if prev and prev.type == "comment" and _node_text(source, prev).startswith("/**"):
        return _node_text(source, prev)
    return None


def _extract_class_methods(node: Node, source: str) -> List[str]:
    body = node.child_by_field_name("body")
    if not body:
        return []
    methods: List[str] = []
    for child in body.named_children:
        if child.type == "method_definition":
            name_node = child.child_by_field_name("name")
            if name_node:
                methods.append(_node_text(source, name_node))
        if child.type == "public_field_definition":
            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if name_node and value_node and value_node.type == "arrow_function":
                methods.append(_node_text(source, name_node))
    return methods


def _extract_heritage(node: Node, source: str) -> Tuple[List[str], Optional[str]]:
    implements: List[str] = []
    extends: Optional[str] = None
    heritage = node.child_by_field_name("heritage")
    if heritage:
        for child in heritage.named_children:
            if child.type == "extends_clause":
                target = child.child_by_field_name("value")
                extends = _node_text(source, target) if target else None
            if child.type == "implements_clause":
                for item in child.named_children:
                    if item.type == "type_identifier":
                        implements.append(_node_text(source, item))
    return implements, extends


def _build_typescript_metadata(node: Node, source: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"typescript_features": []}

    # Framework heuristics (best-effort)
    decorators = _extract_decorators(node, source)
    if decorators:
        metadata["typescript_features"].append("decorators")
        metadata["typescript_features"].append("orm")
        for deco in decorators:
            name = deco.get("name", "")
            if name in {"Controller", "Injectable"}:
                metadata["framework"] = "nestjs"
            if name in {"Component"}:
                metadata["framework"] = "angular"
            if name in {"Entity", "Column", "PrimaryGeneratedColumn"}:
                metadata["framework"] = "typeorm"

    if node.type == "class_declaration" and "framework" not in metadata:
        class_text = _node_text(source, node)
        if "extends React.Component" in class_text or "React.FC" in class_text:
            metadata["framework"] = "react"

    if "framework" not in metadata:
        source_text = source
        if "@angular/core" in source_text or "@Component" in source_text:
            metadata["framework"] = "angular"
        elif "@nestjs/" in source_text or "@Controller" in source_text:
            metadata["framework"] = "nestjs"
        elif "from 'react'" in source_text or "from \"react\"" in source_text or "React.FC" in source_text:
            metadata["framework"] = "react"
        else:
            metadata["framework"] = None

    return metadata


def _extract_external_dependencies(
    node: Node,
    source: str,
    import_context: TypeScriptImportContext,
) -> List[str]:
    dependencies: Set[str] = set()
    for child in _walk(node):
        if child.type == "identifier":
            name = _node_text(source, child)
            if name in import_context.file_imports:
                module = import_context.file_imports[name]
                module_root = module.split("/")[0].lstrip(".")
                if module_root:
                    dependencies.add(module_root)
    return sorted(dependencies)


def _extract_file_imports(root: Node, source: str) -> TypeScriptImportContext:
    file_imports: Dict[str, str] = {}
    import_from_targets: Set[str] = set()
    for child in root.children:
        if child.type == "import_statement":
            module_node = child.child_by_field_name("source")
            module_name = _strip_quotes(_node_text(source, module_node)) if module_node else ""
            for name_node in child.named_children:
                if name_node.type == "import_clause":
                    for clause in name_node.named_children:
                        if clause.type == "identifier":
                            local_name = _node_text(source, clause)
                            file_imports[local_name] = module_name
                        if clause.type == "named_imports":
                            for imp in clause.named_children:
                                if imp.type == "import_specifier":
                                    name = imp.child_by_field_name("name")
                                    if name:
                                        local = _node_text(source, name)
                                        file_imports[local] = module_name
                                        if module_name and not module_name.startswith("."):
                                            import_from_targets.add(local)
    return TypeScriptImportContext(file_imports=file_imports, import_from_targets=import_from_targets)


def _line_span(node: Node) -> Tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _node_text(source: str, node: Optional[Node]) -> str:
    if not node:
        return ""
    return source[node.start_byte:node.end_byte]


def _strip_quotes(text: str) -> str:
    if text.startswith(("\"", "'")) and text.endswith(("\"", "'")):
        return text[1:-1]
    return text


def _walk(node: Node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _slice_source(lines: List[str], start_line: int, end_line: int) -> str:
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    return "\n".join(lines[start_idx:end_idx])


def _is_async(node: Node) -> bool:
    for child in node.children:
        if child.type == "async":
            return True
    return False
