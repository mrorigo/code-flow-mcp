"""
Tree-sitter parser facade with cached parser instances.
"""

from functools import lru_cache

from tree_sitter import Parser, Tree

from .languages import get_py_language, get_ts_language, get_tsx_language, get_rust_language


@lru_cache(maxsize=3)
def get_parser(language_id: str) -> Parser:
    parser = Parser()
    if language_id == "python":
        parser.language = get_py_language()
    elif language_id == "typescript":
        parser.language = get_ts_language()
    elif language_id == "tsx":
        parser.language = get_tsx_language()
    elif language_id == "rust":
        parser.language = get_rust_language()
    else:
        raise ValueError(f"Unsupported language: {language_id}")
    return parser


def parse_source(source: str, language_id: str) -> Tree:
    parser = get_parser(language_id)
    return parser.parse(bytes(source, "utf-8"))
