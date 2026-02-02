"""
Tree-sitter-based Python extractor.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import List

from ..models import CodeElement
from ..utils import get_gitignore_patterns, match_file_against_pattern
from .extractor_base import TreeSitterExtractorBase
from .python_adapter import extract_elements


class TreeSitterPythonExtractor(TreeSitterExtractorBase):
    def __init__(self, enable_performance_monitoring: bool = True):
        super().__init__(language_id="python", enable_performance_monitoring=enable_performance_monitoring)
        self.performance_metrics = {
            "total_files": 0,
            "total_elements": 0,
            "processing_time": 0.0,
            "parse_time": 0.0,
            "io_time": 0.0,
        }

    def extract_from_file(self, file_path: Path) -> List[CodeElement]:
        if not file_path.exists() or file_path.suffix != ".py":
            return []
        start_time = time.time()
        try:
            io_start = time.time()
            source = file_path.read_text(encoding="utf-8")
            io_time = time.time() - io_start

            parse_start = time.time()
            tree = self._parse(source)
            parse_time = time.time() - parse_start

            elements = extract_elements(tree, source, str(file_path.resolve()))

            if self.enable_performance_monitoring:
                total_time = time.time() - start_time
                self.performance_metrics["total_files"] += 1
                self.performance_metrics["total_elements"] += len(elements)
                self.performance_metrics["processing_time"] += total_time
                self.performance_metrics["parse_time"] += parse_time
                self.performance_metrics["io_time"] += io_time

            return elements
        except Exception as e:
            if self.enable_performance_monitoring:
                self.performance_metrics["total_files"] += 1
            logging.info(f"   Warning: Error processing {file_path}: {e}")
            return []

    def extract_from_directory(self, directory: Path) -> List[CodeElement]:
        self.project_root = directory.resolve()
        python_files = list(directory.rglob("*.py"))
        ignored_patterns_with_dirs = get_gitignore_patterns(directory)

        filtered_files = [
            file_path
            for file_path in python_files
            if not any(
                match_file_against_pattern(file_path, pattern, gitignore_dir, directory)
                for pattern, gitignore_dir in ignored_patterns_with_dirs
            )
        ]

        logging.info(f"Found {len(filtered_files)} Python files to analyze (after filtering .gitignore).")
        elements: List[CodeElement] = []
        for file_path in filtered_files:
            elements.extend(self.extract_from_file(file_path))

        return elements
