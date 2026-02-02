"""High-level drift analyzer orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

from code_flow.core.call_graph_builder import FunctionNode, CallEdge
from code_flow.core.drift_clusterer import DriftClusterer
from code_flow.core.drift_features import DriftFeatureExtractor
from code_flow.core.drift_report import DriftReportBuilder
from code_flow.core.drift_topology import TopologyAnalyzer


@dataclass
class DriftAnalyzer:
    project_root: str
    config: Dict[str, Any]

    def analyze(self, functions: List[FunctionNode], edges: List[CallEdge]) -> Dict[str, Any]:
        extractor = DriftFeatureExtractor(
            project_root=self._project_root_path(),
            granularity=self.config.get("drift_granularity", "module"),
            min_entity_size=int(self.config.get("drift_min_entity_size", 3)),
        )
        feature_vectors = extractor.build_feature_vectors(functions, edges)

        clusterer = DriftClusterer(
            confidence_threshold=float(self.config.get("drift_confidence_threshold", 0.6))
        )
        clusters, structural_findings = clusterer.cluster(feature_vectors)

        topology_analyzer = TopologyAnalyzer()
        _, topological_findings = topology_analyzer.analyze(functions, edges)

        report_builder = DriftReportBuilder()
        return report_builder.build_report(
            structural_clusters=clusters,
            structural_findings=structural_findings,
            topological_findings=topological_findings,
            meta={
                "granularity": self.config.get("drift_granularity", "module"),
                "min_entity_size": self.config.get("drift_min_entity_size", 3),
                "cluster_algorithm": self.config.get("drift_cluster_algorithm", "hdbscan"),
            },
        )

    def _project_root_path(self):
        from pathlib import Path

        return Path(self.project_root).resolve()
