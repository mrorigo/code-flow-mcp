"""Feature extraction for drift detection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple
from pathlib import Path

from code_flow.core.call_graph_builder import FunctionNode, CallEdge
from code_flow.core.drift_models import DriftFeatureVector


@dataclass
class DriftFeatureExtractor:
    project_root: Path
    granularity: str = "module"  # module | file
    min_entity_size: int = 3

    def build_feature_vectors(
        self,
        functions: Iterable[FunctionNode],
        edges: Iterable[CallEdge],
    ) -> List[DriftFeatureVector]:
        functions_by_entity = self._group_functions(functions)
        degree_index = self._build_degree_index(edges)
        vectors: List[DriftFeatureVector] = []

        for entity_id, funcs in functions_by_entity.items():
            if len(funcs) < self.min_entity_size:
                continue
            vectors.append(self._build_vector(entity_id, funcs, degree_index))
        return vectors

    def _group_functions(self, functions: Iterable[FunctionNode]) -> Dict[str, List[FunctionNode]]:
        groups: Dict[str, List[FunctionNode]] = defaultdict(list)
        for fn in functions:
            entity_id = self._entity_id(fn)
            groups[entity_id].append(fn)
        return groups

    def _entity_id(self, fn: FunctionNode) -> str:
        if self.granularity == "file":
            return fn.file_path
        module_path = self._module_name(fn.file_path)
        return module_path

    def _module_name(self, file_path: str) -> str:
        try:
            relative_path = Path(file_path).resolve().relative_to(self.project_root)
            return str(relative_path).replace(".py", "").replace(".ts", "").replace(".tsx", "").replace(".rs", "").replace("/", ".")
        except ValueError:
            return Path(file_path).stem

    @staticmethod
    def _build_degree_index(edges: Iterable[CallEdge]) -> Dict[str, Tuple[int, int]]:
        incoming = Counter()
        outgoing = Counter()
        for edge in edges:
            outgoing[edge.caller] += 1
            incoming[edge.callee] += 1
        index: Dict[str, Tuple[int, int]] = {}
        for fqn in set(incoming.keys()).union(outgoing.keys()):
            index[fqn] = (incoming.get(fqn, 0), outgoing.get(fqn, 0))
        return index

    def _build_vector(
        self,
        entity_id: str,
        funcs: List[FunctionNode],
        degree_index: Dict[str, Tuple[int, int]],
    ) -> DriftFeatureVector:
        numeric_acc = defaultdict(list)
        categorical_counts = Counter()
        textual_sets: Dict[str, List[str]] = defaultdict(list)

        for fn in funcs:
            numeric_acc["complexity"].append(float(fn.complexity or 0.0))
            numeric_acc["nloc"].append(float(fn.nloc or 0.0))
            numeric_acc["decorator_count"].append(float(len(fn.decorators or [])))
            numeric_acc["dependency_count"].append(float(len(fn.external_dependencies or [])))
            numeric_acc["exception_count"].append(float(len(fn.catches_exceptions or [])))

            incoming, outgoing = degree_index.get(fn.fully_qualified_name, (0, 0))
            numeric_acc["incoming_degree"].append(float(incoming))
            numeric_acc["outgoing_degree"].append(float(outgoing))

            categorical_counts["is_async"] += 1 if fn.is_async else 0
            categorical_counts["is_static"] += 1 if fn.is_static else 0
            categorical_counts["is_method"] += 1 if fn.is_method else 0
            categorical_counts["has_docstring"] += 1 if fn.docstring else 0

            for decorator in fn.decorators or []:
                if isinstance(decorator, dict):
                    name = decorator.get("name")
                else:
                    name = str(decorator)
                if name:
                    textual_sets["decorators"].append(name)

            for dep in fn.external_dependencies or []:
                textual_sets["external_dependencies"].append(dep)

            for exc in fn.catches_exceptions or []:
                textual_sets["catches_exceptions"].append(exc)

        features_numeric = self._aggregate_numeric(numeric_acc)
        features_categorical = {
            key: int(value) for key, value in categorical_counts.items()
        }

        return DriftFeatureVector(
            entity_id=entity_id,
            granularity=self.granularity,
            features_numeric=features_numeric,
            features_categorical=features_categorical,
            features_textual={k: list(set(v)) for k, v in textual_sets.items()},
            source_hash=None,
        )

    @staticmethod
    def _aggregate_numeric(values: Dict[str, List[float]]) -> Dict[str, float]:
        aggregated: Dict[str, float] = {}
        for key, series in values.items():
            if not series:
                aggregated[f"{key}_mean"] = 0.0
                aggregated[f"{key}_variance"] = 0.0
                continue
            mean = sum(series) / len(series)
            variance = sum((x - mean) ** 2 for x in series) / len(series)
            aggregated[f"{key}_mean"] = mean
            aggregated[f"{key}_variance"] = variance
        return aggregated
