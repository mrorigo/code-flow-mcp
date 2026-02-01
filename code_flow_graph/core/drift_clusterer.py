"""Structural drift detection using clustering heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from code_flow_graph.core.drift_models import DriftCluster, DriftFinding, DriftFeatureVector


@dataclass
class DriftClusterer:
    confidence_threshold: float = 0.6

    def cluster(self, vectors: List[DriftFeatureVector]) -> Tuple[List[DriftCluster], List[DriftFinding]]:
        if not vectors:
            return [], []

        clusters = [self._single_cluster(vectors)]
        findings = self._find_outliers(vectors, clusters[0])
        return clusters, findings

    def _single_cluster(self, vectors: List[DriftFeatureVector]) -> DriftCluster:
        centroid: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for vector in vectors:
            for key, value in vector.features_numeric.items():
                centroid[key] = centroid.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        for key in centroid:
            centroid[key] = centroid[key] / counts[key]

        dominant_patterns = self._dominant_patterns(vectors)
        return DriftCluster(
            cluster_id="cluster-0",
            member_ids=[v.entity_id for v in vectors],
            centroid=centroid,
            dominant_patterns=dominant_patterns,
            cohesion_score=1.0,
        )

    def _dominant_patterns(self, vectors: List[DriftFeatureVector]) -> Dict[str, List[str]]:
        pattern_counts: Dict[str, Dict[str, int]] = {}
        for vector in vectors:
            for key, values in vector.features_textual.items():
                if key not in pattern_counts:
                    pattern_counts[key] = {}
                for value in values:
                    pattern_counts[key][value] = pattern_counts[key].get(value, 0) + 1

        dominant: Dict[str, List[str]] = {}
        for key, counts in pattern_counts.items():
            sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            dominant[key] = [item[0] for item in sorted_items[:5]]
        return dominant

    def _find_outliers(self, vectors: List[DriftFeatureVector], cluster: DriftCluster) -> List[DriftFinding]:
        findings: List[DriftFinding] = []
        distances = [self._distance(vector.features_numeric, cluster.centroid) for vector in vectors]
        if not distances:
            return findings

        mean_distance = sum(distances) / len(distances)
        std_distance = math.sqrt(sum((d - mean_distance) ** 2 for d in distances) / len(distances))

        for vector, distance in zip(vectors, distances):
            if std_distance == 0:
                continue
            score = (distance - mean_distance) / std_distance
            confidence = min(1.0, max(0.0, (score / 3.0)))
            if confidence < self.confidence_threshold:
                continue

            findings.append(
                DriftFinding(
                    entity_id=vector.entity_id,
                    drift_type="structural",
                    confidence=confidence,
                    summary="Structural drift candidate",
                    evidence={
                        "distance": distance,
                        "score": score,
                        "dominant_patterns": cluster.dominant_patterns,
                    },
                    similar_entities=[],
                )
            )
        return findings

    @staticmethod
    def _distance(vec: Dict[str, float], centroid: Dict[str, float]) -> float:
        keys = set(vec.keys()).union(centroid.keys())
        return math.sqrt(sum((vec.get(k, 0.0) - centroid.get(k, 0.0)) ** 2 for k in keys))
