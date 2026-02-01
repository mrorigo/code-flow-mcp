"""
Tree-sitter language loaders.

These helpers return Tree-sitter Language objects for Python/TypeScript/TSX.
"""

from functools import lru_cache

from tree_sitter import Language

from tree_sitter_python import language as py_language
from tree_sitter_typescript import language_typescript, language_tsx


@lru_cache(maxsize=1)
def get_py_language() -> Language:
    return Language(py_language())


@lru_cache(maxsize=1)
def get_ts_language() -> Language:
    return Language(language_typescript())


@lru_cache(maxsize=1)
def get_tsx_language() -> Language:
    return Language(language_tsx())
