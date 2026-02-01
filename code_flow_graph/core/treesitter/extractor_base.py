"""
Shared Tree-sitter extractor base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pathlib import Path

from .parser import parse_source


@dataclass
class TreeSitterExtractorBase:
    language_id: str
    project_root: Optional[Path] = None
    enable_performance_monitoring: bool = True
    performance_metrics: dict = field(default_factory=dict)

    def _parse(self, source: str):
        return parse_source(source, self.language_id)
