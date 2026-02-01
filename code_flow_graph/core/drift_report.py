"""Drift report composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from code_flow_graph.core.drift_models import DriftCluster, DriftFinding


@dataclass
class DriftReportBuilder:
    def build_report(
        self,
        structural_clusters: List[DriftCluster],
        structural_findings: List[DriftFinding],
        topological_findings: List[DriftFinding],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "summary": {
                "structural_findings": len(structural_findings),
                "topological_findings": len(topological_findings),
            },
            "structural_findings": [self._serialize_finding(f) for f in structural_findings],
            "topological_findings": [self._serialize_finding(f) for f in topological_findings],
            "clusters": [self._serialize_cluster(c) for c in structural_clusters],
            "meta": meta,
        }

    @staticmethod
    def _serialize_finding(finding: DriftFinding) -> Dict[str, Any]:
        return {
            "entity_id": finding.entity_id,
            "drift_type": finding.drift_type,
            "confidence": finding.confidence,
            "summary": finding.summary,
            "evidence": finding.evidence,
            "similar_entities": finding.similar_entities,
        }

    @staticmethod
    def _serialize_cluster(cluster: DriftCluster) -> Dict[str, Any]:
        return {
            "cluster_id": cluster.cluster_id,
            "member_ids": cluster.member_ids,
            "centroid": cluster.centroid,
            "dominant_patterns": cluster.dominant_patterns,
            "cohesion_score": cluster.cohesion_score,
        }
