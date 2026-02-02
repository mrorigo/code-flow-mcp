"""
Tree-sitter adapter for Rust source.

Produces CodeElement / FunctionElement / ClassElement objects from Tree-sitter nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from tree_sitter import Node, Tree

from ..models import ClassElement, CodeElement, FunctionElement
from ..utils import calculate_nloc_typescript, hash_source_snippet


@dataclass
class RustImportContext:
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
    import_context: RustImportContext,
    elements: List[CodeElement],
) -> None:
    for child in node.children:
        if child.type == "function_item":
            elements.append(_build_function(child, source, file_path, lines, import_context, current_impl=None))
        elif child.type == "impl_item":
            _walk_impl(child, source, file_path, lines, import_context, elements)
        elif child.type == "struct_item":
            elements.append(_build_struct(child, source, file_path, lines))
        elif child.type == "enum_item":
            elements.append(_build_enum(child, source, file_path, lines))
        elif child.type == "trait_item":
            elements.append(_build_trait(child, source, file_path, lines))
        else:
            _walk_root(child, source, file_path, lines, import_context, elements)


def _walk_impl(
    node: Node,
    source: str,
    file_path: str,
    lines: List[str],
    import_context: RustImportContext,
    elements: List[CodeElement],
) -> None:
    impl_type = _impl_type_name(node, source)
    for child in node.children:
        if child.type == "function_item":
            elements.append(_build_function(child, source, file_path, lines, import_context, current_impl=impl_type))


def _build_function(
    node: Node,
    source: str,
    file_path: str,
    lines: List[str],
    import_context: RustImportContext,
    current_impl: Optional[str],
) -> FunctionElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(source, name_node) if name_node else "<anonymous>"
    start_line, end_line = _line_span(node)
    parameters = _extract_parameters(node, source)
    return_type = _extract_return_type(node, source)
    docstring = _extract_docstring(node, source)
    hash_body = hash_source_snippet(lines, start_line, end_line)
    nloc = calculate_nloc_typescript(lines, start_line, end_line)
    external_dependencies = _extract_external_dependencies(node, source, import_context)

    return FunctionElement(
        name=name,
        kind="function",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        parameters=parameters,
        return_type=return_type,
        is_async=_is_async(node),
        docstring=docstring,
        is_method=current_impl is not None,
        class_name=current_impl,
        complexity=_calculate_complexity(node),
        nloc=nloc,
        external_dependencies=external_dependencies,
        decorators=[],
        hash_body=hash_body,
        metadata={"rust": True},
    )


def _build_struct(node: Node, source: str, file_path: str, lines: List[str]) -> ClassElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(source, name_node) if name_node else "<anonymous>"
    start_line, end_line = _line_span(node)
    hash_body = hash_source_snippet(lines, start_line, end_line)
    return ClassElement(
        name=name,
        kind="struct",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        methods=[],
        attributes=_extract_struct_fields(node, source),
        extends=None,
        implements=[],
        docstring=_extract_docstring(node, source),
        decorators=[],
        hash_body=hash_body,
        metadata={"rust": True},
    )


def _build_enum(node: Node, source: str, file_path: str, lines: List[str]) -> ClassElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(source, name_node) if name_node else "<anonymous>"
    start_line, end_line = _line_span(node)
    hash_body = hash_source_snippet(lines, start_line, end_line)
    return ClassElement(
        name=name,
        kind="enum",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        methods=[],
        attributes=_extract_enum_variants(node, source),
        extends=None,
        implements=[],
        docstring=_extract_docstring(node, source),
        decorators=[],
        hash_body=hash_body,
        metadata={"rust": True},
    )


def _build_trait(node: Node, source: str, file_path: str, lines: List[str]) -> ClassElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(source, name_node) if name_node else "<anonymous>"
    start_line, end_line = _line_span(node)
    hash_body = hash_source_snippet(lines, start_line, end_line)
    return ClassElement(
        name=name,
        kind="trait",
        file_path=file_path,
        line_start=start_line,
        line_end=end_line,
        full_source=source,
        methods=_extract_trait_methods(node, source),
        attributes=[],
        extends=None,
        implements=[],
        docstring=_extract_docstring(node, source),
        decorators=[],
        hash_body=hash_body,
        metadata={"rust": True},
    )


def _extract_parameters(node: Node, source: str) -> List[str]:
    params_node = node.child_by_field_name("parameters")
    if not params_node:
        return []
    params: List[str] = []
    for child in params_node.named_children:
        params.append(_node_text(source, child))
    return params


def _extract_return_type(node: Node, source: str) -> Optional[str]:
    return_node = node.child_by_field_name("return_type")
    return _node_text(source, return_node) if return_node else None


def _extract_docstring(node: Node, source: str) -> Optional[str]:
    for child in node.children:
        if child.type == "doc_comment":
            return _node_text(source, child)
    return None


def _extract_struct_fields(node: Node, source: str) -> List[str]:
    body = node.child_by_field_name("body")
    if not body:
        return []
    fields: List[str] = []
    for child in body.named_children:
        if child.type == "field_declaration":
            name_node = child.child_by_field_name("name")
            if name_node:
                fields.append(_node_text(source, name_node))
    return fields


def _extract_enum_variants(node: Node, source: str) -> List[str]:
    body = node.child_by_field_name("body")
    if not body:
        return []
    variants: List[str] = []
    for child in body.named_children:
        if child.type == "enum_variant":
            name_node = child.child_by_field_name("name")
            if name_node:
                variants.append(_node_text(source, name_node))
    return variants


def _extract_trait_methods(node: Node, source: str) -> List[str]:
    body = node.child_by_field_name("body")
    if not body:
        return []
    methods: List[str] = []
    for child in body.named_children:
        if child.type == "function_item":
            name_node = child.child_by_field_name("name")
            if name_node:
                methods.append(_node_text(source, name_node))
    return methods


def _impl_type_name(node: Node, source: str) -> Optional[str]:
    type_node = node.child_by_field_name("type")
    return _node_text(source, type_node) if type_node else None


def _extract_external_dependencies(
    node: Node,
    source: str,
    import_context: RustImportContext,
) -> List[str]:
    dependencies: Set[str] = set()
    for child in _walk(node):
        if child.type == "identifier":
            name = _node_text(source, child)
            if name in import_context.file_imports:
                module = import_context.file_imports[name]
                module_root = module.split("::")[0].lstrip("::")
                if module_root:
                    dependencies.add(module_root)
    return sorted(dependencies)


def _extract_file_imports(root: Node, source: str) -> RustImportContext:
    file_imports: Dict[str, str] = {}
    import_from_targets: Set[str] = set()
    for child in root.children:
        if child.type == "use_declaration":
            statement = _node_text(source, child)
            path_text = statement.replace("use", "").replace(";", "").strip()
            if " as " in path_text:
                path, alias = [part.strip() for part in path_text.split(" as ", 1)]
            else:
                path = path_text
                alias = path.split("::")[-1]
            if alias:
                file_imports[alias] = path
                import_from_targets.add(alias)
    return RustImportContext(file_imports=file_imports, import_from_targets=import_from_targets)


def _line_span(node: Node) -> Tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _node_text(source: str, node: Optional[Node]) -> str:
    if not node:
        return ""
    source_bytes = source.encode("utf-8")
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


def _walk(node: Node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _is_async(node: Node) -> bool:
    for child in node.children:
        if child.type == "async":
            return True
    return False


def _calculate_complexity(node: Node) -> int:
    complexity = 1
    for child in _walk(node):
        if child.type in {
            "if_expression",
            "match_expression",
            "while_expression",
            "for_expression",
            "loop_expression",
        }:
            complexity += 1
    return complexity
