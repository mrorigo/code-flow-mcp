# Improve Call Graph Accuracy Plan

## 1) Establish baseline metrics
- Generate current drift + call graph for this repo and for each sample project under `tests/typescript/sample_projects/*`.
- Track baseline:
  - total edges
  - edges per module
  - number of cycles in drift
  - count of low-confidence edges (from [`CallGraphBuilder._create_edge_if_valid()`](../code_flow/core/call_graph_builder.py:443))
- Store these metrics in a simple table for before/after comparison.

## 2) Improve Python call resolution (AST-based)
Focus on reducing false positives in [`CallGraphBuilder._create_edge_if_valid()`](../code_flow/core/call_graph_builder.py:443):

### 2.1 Scope resolution to the caller’s module
- Build a module-local symbol table per file that maps:
  - locally defined functions/classes
  - imported functions/classes
- Only resolve `foo()` to symbols present in the caller module’s symbol table.

### 2.2 Import-aware resolution
- When `call.callee` is `obj.method`, attempt to resolve `obj` to an imported module/class and then resolve `method` within that scope.
- If no import context, downgrade confidence or skip.

### 2.3 Confidence gating
- Attach confidence rules:
  - module-local direct matches: `0.95`
  - imported symbol match: `0.9`
  - heuristic/global fallback: `0.6`
- Exclude edges below a configurable threshold from topology drift.

## 3) Improve TypeScript call resolution (regex-based)
### 3.1 Use TS import map
- Build a map of `import {foo} from 'x'`, `import * as x from 'y'`.
- Resolve `foo()` only if `foo` is in map or local definition.

### 3.2 Restrict global resolution
- If call doesn’t match local or imported symbols, skip unless inside same file.

## 4) Normalize module naming for topology
- Normalize module names from absolute path to project-relative in [`TopologyAnalyzer._module_from_path()`](../code_flow/core/drift_topology.py:80).
- This avoids the same module being treated as multiple nodes.

## 5) Cycle deduplication in drift output
- Normalize cycles (rotate to canonical minimum node, drop duplicates) inside [`TopologyAnalyzer._detect_cycles()`](../code_flow/core/drift_topology.py:84).
- Keeps report concise and easier to compare.

## 6) Validation plan (using this repo + sample_projects)
### 6.1 Local repo
- Compare before/after:
  - edge count should drop
  - cycles should drop or shrink
  - proportion of low-confidence edges should drop

### 6.2 Sample projects
- For each sample project (`basic`, `express`, `angular`, `nextjs`, `vue`):
  - ensure expected edges still appear (e.g., service → repository calls)
  - verify that “obvious” cycles (e.g., dependency injection loop artifacts) disappear

### 6.3 Regression guard
- Add or extend tests to assert:
  - total edges falls within expected range (not zero, not exploding)
  - confidence distribution shifts upward
  - cycles count decreases

## 7) Deliverables
- Updated call-resolution logic in [`CallGraphBuilder._create_edge_if_valid()`](../code_flow/core/call_graph_builder.py:443)
- Import/symbol-table helpers for Python and TypeScript
- Topology normalization + cycle dedup
- Test/metrics report comparing baseline vs improved

