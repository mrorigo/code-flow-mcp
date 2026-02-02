"""Topological drift detection using call graph structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

from code_flow_graph.core.call_graph_builder import CallEdge, FunctionNode
from code_flow_graph.core.drift_models import DriftFinding, TopologyViolation


@dataclass
class TopologyAnalyzer:
    def analyze(self, functions: List[FunctionNode], edges: List[CallEdge]) -> Tuple[List[TopologyViolation], List[DriftFinding]]:
        module_graph = self._build_module_graph(functions, edges)
        logging.info(
            "Drift topology: modules=%d edges=%d",
            len(module_graph),
            sum(len(targets) for targets in module_graph.values()),
        )
        violations: List[TopologyViolation] = []
        findings: List[DriftFinding] = []

        cycles = self._detect_cycles(module_graph)
        for cycle in cycles:
            violations.append(
                TopologyViolation(
                    edge={"cycle": cycle},
                    violation_type="cycle",
                    context_path=cycle,
                    confidence=0.7,
                )
            )

        layer_map = self._infer_layers(module_graph)
        logging.info("Drift topology: inferred layers for %d modules", len(layer_map))
        for source, targets in module_graph.items():
            for target in targets:
                if layer_map.get(source, 0) > layer_map.get(target, 0):
                    violations.append(
                        TopologyViolation(
                            edge={"caller_module": source, "callee_module": target},
                            violation_type="layer_inversion",
                            context_path=[source, target],
                            confidence=0.6,
                        )
                    )

        for violation in violations:
            findings.append(
                DriftFinding(
                    entity_id=str(violation.edge),
                    drift_type="topological",
                    confidence=violation.confidence,
                    summary=f"Topological drift: {violation.violation_type}",
                    evidence={"edge": violation.edge, "path": violation.context_path},
                    similar_entities=[],
                )
            )

        return violations, findings

    def _build_module_graph(self, functions: List[FunctionNode], edges: List[CallEdge]) -> Dict[str, List[str]]:
        module_by_fqn = {fn.fully_qualified_name: self._module_from_path(fn.file_path) for fn in functions}
        graph: Dict[str, List[str]] = {}
        for edge in edges:
            source_module = module_by_fqn.get(edge.caller)
            target_module = module_by_fqn.get(edge.callee)
            if not source_module or not target_module:
                continue
            if source_module == target_module:
                continue
            graph.setdefault(source_module, [])
            if target_module not in graph[source_module]:
                graph[source_module].append(target_module)
        return graph

    @staticmethod
    def _module_from_path(file_path: str) -> str:
        return file_path.replace("/", ".").replace("\\", ".")

    @staticmethod
    def _detect_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
        cycles: List[List[str]] = []
        visited: set[str] = set()
        stack: set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in stack:
                    try:
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        if len(cycle) > 2:
                            cycles.append(cycle)
                    except ValueError:
                        continue

            stack.remove(node)
            path.pop()

        for node in graph.keys():
            if node not in visited:
                dfs(node)
        return cycles

    @staticmethod
    def _infer_layers(graph: Dict[str, List[str]]) -> Dict[str, int]:
        layer_map: Dict[str, int] = {node: 0 for node in graph.keys()}
        changed = True
        max_iterations = max(1, len(graph) * 2)
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for source, targets in graph.items():
                for target in targets:
                    if layer_map.get(target, 0) <= layer_map.get(source, 0):
                        layer_map[target] = layer_map[source] + 1
                        changed = True
        if iteration >= max_iterations:
            logging.warning(
                "Drift topology: layer inference hit iteration cap (%d). Graph may contain cycles.",
                max_iterations,
            )
        return layer_map
