import pytest
from pathlib import Path

from code_flow_graph.core.treesitter.rust_extractor import TreeSitterRustExtractor


def test_rust_extractor_parses_sample():
    fixture = Path("tests/core/rust/fixtures/sample.rs")
    extractor = TreeSitterRustExtractor()

    elements = extractor.extract_from_file(fixture)

    names = {element.name for element in elements}
    assert "User" in names
    assert "make_user" in names
    assert "Printable" in names
