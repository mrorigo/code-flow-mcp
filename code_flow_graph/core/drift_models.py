"""Data models for drift detection output and intermediate features."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class DriftFeatureVector:
    entity_id: str
    granularity: str  # file | module | package
    features_numeric: Dict[str, float] = field(default_factory=dict)
    features_categorical: Dict[str, int] = field(default_factory=dict)
    features_textual: Dict[str, List[str]] = field(default_factory=dict)
    source_hash: Optional[str] = None


@dataclass
class DriftCluster:
    cluster_id: str
    member_ids: List[str] = field(default_factory=list)
    centroid: Dict[str, float] = field(default_factory=dict)
    dominant_patterns: Dict[str, Any] = field(default_factory=dict)
    cohesion_score: float = 0.0


@dataclass
class DriftFinding:
    entity_id: str
    drift_type: str  # structural | topological
    confidence: float
    summary: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    similar_entities: List[str] = field(default_factory=list)


@dataclass
class TopologyViolation:
    edge: Dict[str, Any]
    violation_type: str  # cycle | layer_inversion | unexpected_dependency
    context_path: List[str] = field(default_factory=list)
    confidence: float = 0.0
