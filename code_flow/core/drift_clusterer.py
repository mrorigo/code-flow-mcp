"""Structural drift detection using clustering heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None

try:
    from sklearn.cluster import DBSCAN  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DBSCAN = None

from code_flow.core.drift_models import DriftCluster, DriftFinding, DriftFeatureVector


@dataclass
class DriftClusterer:
    confidence_threshold: float = 0.6
    algorithm: str = "hdbscan"
    eps: float = 0.75
    min_samples: int = 5

    def cluster(self, vectors: List[DriftFeatureVector]) -> Tuple[List[DriftCluster], List[DriftFinding]]:
        if not vectors:
            return [], []

        labels = self._cluster_labels(vectors)
        clusters = self._build_clusters(vectors, labels)
        findings: List[DriftFinding] = []
        for cluster in clusters:
            members = [v for v in vectors if v.entity_id in cluster.member_ids]
            findings.extend(self._find_outliers(members, cluster))
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

    def _cluster_labels(self, vectors: List[DriftFeatureVector]) -> List[int]:
        features = [self._vector_to_dense(v) for v in vectors]
        algo = self.algorithm.lower()

        if algo == "hdbscan" and hdbscan is not None:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, self.min_samples))
            return clusterer.fit_predict(features).tolist()

        if DBSCAN is not None:
            clusterer = DBSCAN(eps=self.eps, min_samples=max(2, self.min_samples))
            return clusterer.fit_predict(features).tolist()

        # Fallback to single cluster if no clustering backend is available.
        return [0 for _ in vectors]

    def _build_clusters(self, vectors: List[DriftFeatureVector], labels: List[int]) -> List[DriftCluster]:
        clustered: Dict[int, List[DriftFeatureVector]] = {}
        for vector, label in zip(vectors, labels):
            clustered.setdefault(label, []).append(vector)

        clusters: List[DriftCluster] = []
        for label, members in clustered.items():
            cluster = self._single_cluster(members)
            if label < 0:
                cluster.cluster_id = "cluster-noise"
            else:
                cluster.cluster_id = f"cluster-{label}"
            clusters.append(cluster)
        return clusters

    def _vector_to_dense(self, vector: DriftFeatureVector) -> List[float]:
        numeric = vector.features_numeric
        keys = sorted(numeric.keys())
        return [float(numeric.get(k, 0.0)) for k in keys]

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
