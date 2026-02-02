from __future__ import annotations

from code_flow.core.call_graph_builder import CallEdge, FunctionNode
from code_flow.core.drift_analyzer import DriftAnalyzer
from code_flow.core.drift_topology import TopologyAnalyzer


def _make_function(name: str, module: str, project_root: str = "/repo") -> FunctionNode:
    file_path = f"{project_root}/{module.replace('.', '/')}.py"
    fqn = f"{module}.{name}"
    return FunctionNode(
        name=name,
        fully_qualified_name=fqn,
        file_path=file_path,
        line_start=1,
        line_end=2,
        parameters=[],
    )


def _make_edge(caller: FunctionNode, callee: FunctionNode) -> CallEdge:
    return CallEdge(
        caller=caller.fully_qualified_name,
        callee=callee.fully_qualified_name,
        file_path=caller.file_path,
        line_number=1,
        call_type="direct",
        parameters=[],
        is_static_call=True,
        confidence=0.95,
    )


def test_topology_analyzer_detects_cycle() -> None:
    project_root = "/repo"
    a = _make_function("fa", "pkg.a", project_root)
    b = _make_function("fb", "pkg.b", project_root)
    c = _make_function("fc", "pkg.c", project_root)

    edges = [
        _make_edge(a, b),
        _make_edge(b, c),
        _make_edge(c, a),
    ]

    analyzer = TopologyAnalyzer(project_root=project_root)
    violations, _ = analyzer.analyze([a, b, c], edges)

    assert any(v.violation_type == "cycle" for v in violations)


def test_topology_analyzer_detects_layer_inversion() -> None:
    project_root = "/repo"
    a = _make_function("fa", "pkg.a", project_root)
    b = _make_function("fb", "pkg.b", project_root)

    edges = [
        _make_edge(a, b),
        _make_edge(b, a),
    ]

    analyzer = TopologyAnalyzer(project_root=project_root)
    violations, _ = analyzer.analyze([a, b], edges)

    assert any(v.violation_type == "layer_inversion" for v in violations)


def test_drift_report_includes_topological_findings() -> None:
    project_root = "/repo"
    a = _make_function("fa", "pkg.a", project_root)
    b = _make_function("fb", "pkg.b", project_root)
    c = _make_function("fc", "pkg.c", project_root)

    edges = [
        _make_edge(a, b),
        _make_edge(b, c),
        _make_edge(c, a),
    ]

    analyzer = DriftAnalyzer(
        project_root=project_root,
        config={
            "drift_granularity": "module",
            "drift_min_entity_size": 1,
            "drift_cluster_algorithm": "hdbscan",
            "drift_cluster_eps": 0.75,
            "drift_cluster_min_samples": 2,
            "drift_confidence_threshold": 0.6,
        },
    )
    report = analyzer.analyze([a, b, c], edges)

    assert report["summary"]["topological_findings"] >= 1
    assert len(report["topological_findings"]) >= 1
