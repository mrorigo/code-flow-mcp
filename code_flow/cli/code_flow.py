"""
Main orchestrator for code graph analysis with corrected pipeline logic.
"""

from tqdm import tqdm
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import argparse
import asyncio

# Ensure local modules can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.treesitter.python_extractor import TreeSitterPythonExtractor
from core.treesitter.typescript_extractor import TreeSitterTypeScriptExtractor
from core.models import FunctionElement, ClassElement, CodeElement
from code_flow.core.config import load_config
from code_flow.core.drift_topology import TopologyAnalyzer
from core.call_graph_builder import CallGraphBuilder, FunctionNode # Import FunctionNode here
from core.vector_store import CodeVectorStore
from core.drift_analyzer import DriftAnalyzer
from code_flow.core.llm_summary import SummaryGenerator, SummaryProcessor

def resolve_embedding_model(model_name: str) -> str:
    """
    Resolve embedding model shorthand to actual model name.
    
    Args:
        model_name: Either a shorthand ('fast', 'medium', 'accurate') or a specific model name
        
    Returns:
        The actual SentenceTransformer model name
    """
    shortcuts = {
        'fast': 'all-MiniLM-L6-v2',      # 384 dimensions, fastest
        'medium': 'all-MiniLM-L12-v2',   # 384 dimensions, balanced
        'accurate': 'all-mpnet-base-v2'  # 768 dimensions, most accurate
    }
    return shortcuts.get(model_name.lower(), model_name)


def _safe_query(
    analyzer: "CodeGraphAnalyzer",
    query: str,
    n_results: int = 5,
    generate_mermaid: bool = False,
    llm_optimized_mermaid: bool = False,
    min_similarity: float = 0.0,
) -> int:
    try:
        analyzer.query(
            query,
            n_results=n_results,
            generate_mermaid=generate_mermaid,
            llm_optimized_mermaid=llm_optimized_mermaid,
            min_similarity=min_similarity,
        )
        return 0
    except Exception as exc:
        message = str(exc)
        if "embedding with dimension" in message or "InvalidArgumentError" in message:
            print(
                "⚠️  Vector store query failed due to embedding dimension mismatch. "
                "Re-run analysis with a consistent embedding model.",
                file=sys.stderr,
            )
            return 0
        raise


class CodeGraphAnalyzer:
    """Main analyzer that orchestrates the entire pipeline."""

    def __init__(self, root_directory: Path, language: str = "python", embedding_model: str = 'all-MiniLM-L6-v2', max_tokens: int = 256, enable_summaries: bool = False, llm_config: Optional[Dict] = None, chromadb_path: str | None = None):
        """
        Initialize the analyzer.

        Args:
            root_directory: Root of the codebase. This is also used to derive the
                            persistence directory for the vector store.
            language: 'python' or 'typescript'.
            enable_summaries: Enable LLM-driven summary generation.
            llm_config: Configuration for LLM summary generation.
        """
        self.root_directory = root_directory
        self.language = language.lower()
        self.extractor = self._get_extractor()
        self.graph_builder = CallGraphBuilder()
        # Pass the root_directory to the graph_builder for consistent FQN generation
        self.graph_builder.project_root = root_directory

        # Vector store path derived from config (.codeflow/chroma)
        vector_store_path = Path(chromadb_path) if chromadb_path else (self.root_directory / "code_vectors_chroma")
        try:
            self.vector_store = CodeVectorStore(
                persist_directory=str(vector_store_path),
                embedding_model_name=embedding_model,
                max_tokens=max_tokens
            )
        except Exception as e:
            print(
                "Could not start CodeVectorStore at "
                f"{vector_store_path}. Vector-based features will be disabled. Error: {e}",
                file=sys.stderr,
            )
            self.vector_store = None

        self.all_elements: List[CodeElement] = []
        self.classes: Dict[str, ClassElement] = {}
        
        # Summary generation setup
        self.summary_processor = None
        if enable_summaries and self.vector_store:
            config = llm_config or {}
            self.summary_generator = SummaryGenerator({'llm_config': config})
            self.summary_processor = SummaryProcessor(
                generator=self.summary_generator,
                builder=self.graph_builder,
                vector_store=self.vector_store,
                concurrency=config.get('concurrency', 5),
                prioritize_entry_points=config.get('prioritize_entry_points', False)
            )

    def _get_extractor(self):
        """Get the appropriate AST extractor for the language."""
        if self.language == "python":
            return TreeSitterPythonExtractor()
        elif self.language == "typescript":
            return TreeSitterTypeScriptExtractor()
        elif self.language == "rust":
            from code_flow.core.treesitter.rust_extractor import TreeSitterRustExtractor
            return TreeSitterRustExtractor()
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def analyze(self) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.

        Returns:
            Analysis results dictionary.
        """
        print(f"🚀 Starting analysis of {self.language} codebase at {self.root_directory}")

        # Step 1: Extract AST elements
        print("\n📖 Step 1: Extracting AST elements...")
        self.extractor.project_root = self.root_directory # Set project root for extractor too
        self.all_elements = self.extractor.extract_from_directory(self.root_directory)
        print(f"   Found {len(self.all_elements)} code elements")

        # Categorize elements for reporting
        functions = [e for e in self.all_elements if isinstance(e, FunctionElement)]
        classes = [e for e in self.all_elements if isinstance(e, ClassElement)]
        self.classes = {c.name: c for c in classes}

        # Step 2: Build call graph
        print("\n🔗 Step 2: Building call graph...")
        self.graph_builder.build_from_elements(self.all_elements)

        # Step 3: Populate vector store (using enriched FunctionNode objects)
        if self.vector_store:
            print("\n💾 Step 3: Populating vector store...")
            self._populate_vector_store()
        else:
            print("   Vector store is disabled, skipping population.")

        # Step 4: Generate summaries if enabled
        if self.summary_processor:
            print("\n🤖 Step 4: Generating LLM summaries...")
            self._generate_summaries()
            print("\n📊 Step 5: Generating analysis report...")
        else:
            print("\n📊 Step 4: Generating analysis report...")
        
        report = self._generate_report(classes, functions)
        self._export_metrics(report)

        print("\n✅ Analysis complete!")
        return report

    def _populate_vector_store(self) -> None:
        """
        Populate the vector store with FunctionNode objects and edges from the graph.
        This method uses the CORRECT data type (FunctionNode) after the graph is built.
        """
        if not self.vector_store:
            print("   Vector store is not initialized. Skipping query.", file=sys.stderr)
            return
        if not self.graph_builder.functions:
            print("   No functions found in graph builder, skipping vector store population.")
            return

        graph_functions = list(self.graph_builder.functions.values())
        print(f"   Storing {len(graph_functions)} functions and {len(self.graph_builder.edges)} edges...")

        # Read all source files first
        sources = {}
        for node in graph_functions:
            if node.file_path not in sources:
                try:
                    with open(node.file_path, 'r', encoding='utf-8') as f:
                        sources[node.file_path] = f.read()
                except FileNotFoundError:
                    print(f"   Warning: Source file for node {node.fully_qualified_name} not found at {node.file_path}. Skipping.", file=sys.stderr)
                    sources[node.file_path] = ""
                except Exception as e:
                    print(f"   Warning: Could not read source file {node.file_path}: {e}", file=sys.stderr)
                    sources[node.file_path] = ""

        if self.vector_store and hasattr(self.vector_store, "persist_directory"):
            Path(self.vector_store.persist_directory).mkdir(parents=True, exist_ok=True)

        # Batch store functions
        try:
            self.vector_store.add_function_nodes_batch(graph_functions, sources, batch_size=100)
        except Exception as e:
            print(f"   Warning: Batch function storage failed, falling back to individual: {e}", file=sys.stderr)
            for node in tqdm(graph_functions, desc="Storing functions individually"):
                try:
                    source_code = sources.get(node.file_path, "")
                    self.vector_store.add_function_node(node, source_code)
                except Exception as e2:
                    print(f"   Warning: Could not process/store node {node.fully_qualified_name}: {e2}", file=sys.stderr)

        # Batch store edges
        try:
            self.vector_store.add_edges_batch(self.graph_builder.edges, batch_size=100)
        except Exception as e:
            print(f"   Warning: Batch edge storage failed, falling back to individual: {e}", file=sys.stderr)
            for edge in tqdm(self.graph_builder.edges, desc="Storing edges individually"):
                try:
                    self.vector_store.add_edge(edge)
                except Exception as e2:
                    print(f"   Warning: Could not add edge {edge.caller} -> {edge.callee}: {e2}", file=sys.stderr)

        stats = self.vector_store.get_stats()
        print(f"   Vector store populated. Total documents: {stats.get('total_documents', 'N/A')}")
    
    def _generate_summaries(self) -> None:
        """Generate summaries for functions using LLM."""
        if not self.summary_processor:
            return
        
        # Run the async summary generation
        import asyncio
        asyncio.run(self._generate_summaries_async())
    
    async def _generate_summaries_async(self) -> None:
        """Async implementation of summary generation."""
        # Start the processor
        if not self.summary_processor or not self.vector_store:
            return
        await self.summary_processor.start_async()
        
        # Find nodes missing summaries
        all_missing_fqns = self.vector_store.get_nodes_missing_summary()
        
        # Filter to only include FQNs that exist in the builder
        # This avoids warnings for stale/filtered functions
        missing_fqns = [fqn for fqn in all_missing_fqns if fqn in self.summary_processor.builder.functions]
        
        skipped_count = len(all_missing_fqns) - len(missing_fqns)
        if skipped_count > 0:
            print(f"   Skipped {skipped_count} functions not in current call graph (stale/filtered)")
        
        if not missing_fqns:
            print("   All functions already have summaries.")
            await self.summary_processor.stop()
            return
            
        print(f"   Found {len(missing_fqns)} functions needing summaries")
        print("   Generating summaries in background...")
        
        # Enqueue all missing
        for fqn in missing_fqns:
            self.summary_processor.enqueue(fqn)
        
        # Wait for queue to complete with progress bar
        import time
        with tqdm(total=len(missing_fqns), desc="Generating summaries") as pbar:
            last_count = 0
            while not self.summary_processor.queue.empty() or any(w and not w.done() for w in self.summary_processor.workers):
                await asyncio.sleep(0.5)
                # Update progress based on queue size
                current_remaining = self.summary_processor.queue.qsize()
                completed = len(missing_fqns) - current_remaining
                pbar.update(completed - last_count)
                last_count = completed
            
            # Final update
            pbar.update(len(missing_fqns) - last_count)
        
        # Stop the processor
        await self.summary_processor.stop()
        
        print("   ✓ Summary generation complete")

    def _generate_report(self, classes: List[ClassElement], functions: List[FunctionElement]) -> Dict[str, Any]:
        """Generate a comprehensive analysis report using graph data."""
        if not self.graph_builder.functions:
            return {
                "summary": {"total_functions": 0, "total_edges": 0},
                "entry_points": [],
                "classes_summary": {"total": 0},
                "call_graph": {"functions": {}, "edges": {}}
            }

        graph_entry_points = self.graph_builder.get_entry_points_scored()

        graph_export = self.graph_builder.export_graph()
        if not isinstance(graph_export, dict):
            graph_export = {"summary": {}, "functions": {}, "edges": []}
        report = {
            "summary": graph_export.get("summary", {}),
            "entry_points": [
                {
                    "name": item["function"].name,
                    "fully_qualified_name": item["function"].fully_qualified_name,
                    "file_path": item["function"].file_path,
                    "line_number": item["function"].line_start,
                    "line_end": item["function"].line_end,
                    "is_method": item["function"].is_method,
                    "class_name": item["function"].class_name,
                    "is_async": item["function"].is_async,
                    "has_docstring": bool(item["function"].docstring),
                    "incoming_connections": len(item["function"].incoming_edges),
                    "outgoing_connections": len(item["function"].outgoing_edges),
                    "parameters": item["function"].parameters,
                    "complexity": item["function"].complexity,
                    "nloc": item["function"].nloc,
                    "external_dependencies": item["function"].external_dependencies,
                    "decorators": item["function"].decorators,
                    "catches_exceptions": item["function"].catches_exceptions,
                    "local_variables_declared": item["function"].local_variables_declared,
                    "hash_body": item["function"].hash_body,
                    "entry_point_score": item["meta"]["score"],
                    "entry_point_category": item["meta"]["category"],
                    "entry_point_priority": item["meta"]["priority"],
                    "entry_point_signals": item["meta"]["signals"],
                }
                for item in graph_entry_points
            ],
            "classes_summary": {
                "total": len(classes),
                "with_methods": sum(1 for c in classes if c.methods),
                "with_inheritance": sum(1 for c in classes if c.extends),
            },
            "functions_summary": {
                "total": len(functions),
                "with_parameters": sum(1 for f in functions if f.parameters),
                "with_return_type": sum(1 for f in functions if f.return_type),
            },
            "call_graph": graph_export,
        }
        if self.vector_store is not None:
            report["vector_store_enabled"] = True
        else:
            report["vector_store_enabled"] = False

        return report

    def _export_metrics(self, report: Dict[str, Any]) -> None:
        config = load_config()
        if self.root_directory:
            config.project_root = str(self.root_directory)
        config.reports_dir().mkdir(parents=True, exist_ok=True)
        metrics_path = config.reports_dir() / "call_graph_metrics.json"
        markdown_path = config.reports_dir() / "call_graph_metrics.md"

        topology = TopologyAnalyzer(project_root=str(self.root_directory))
        module_graph = topology._build_module_graph(
            list(self.graph_builder.functions.values()),
            self.graph_builder.edges,
        )
        cycles = topology._detect_cycles(module_graph)

        metrics = self.graph_builder.compute_metrics()
        metrics["cycles"] = cycles
        metrics["cycle_count"] = len(cycles)

        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        edges_per_module = metrics.get("edges_per_module", {})
        md_lines = [
            "# Call Graph Metrics",
            "",
            f"- Total edges: {metrics.get('total_edges', 0)}",
            f"- Low-confidence edges: {metrics.get('low_confidence_edges', 0)}",
            f"- Cycle count: {metrics.get('cycle_count', 0)}",
            "",
            "## Edges per module",
            "| Module | Edges |",
            "| --- | --- |",
        ]
        for module, count in sorted(edges_per_module.items()):
            md_lines.append(f"| {module} | {count} |")

        markdown_path.write_text("\n".join(md_lines), encoding="utf-8")

    def query(
        self,
        question: str,
        n_results: int = 5,
        generate_mermaid: bool = False,
        llm_optimized_mermaid: bool = False,
        min_similarity: float = 0.0,
    ) -> None:
        """
        Query the vector store for insights and print the results.
        Args:
            question: The semantic query string.
            n_results: Number of results to return.
            generate_mermaid: If True, also generates a Mermaid graph highlighting query results.
            llm_optimized_mermaid: If True, generates Mermaid graph optimized for LLM token count.
        """
        if not self.vector_store:
            print("Vector store is not available. Cannot perform query.", file=sys.stderr)
            return

        print(f"\n🔍 Running semantic search for: '{question}'")
        results = self.vector_store.query_codebase(
            question,
            n_results,
            min_similarity=min_similarity,
        )

        if not results:
            print("   No relevant functions found. Try a different query.")
            return

        print(f"\n# Top {len(results)} results:")
        highlight_fqns = []
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            fqn = meta.get('fully_qualified_name')
            if fqn:
                highlight_fqns.append(fqn)

            similarity = result.get('similarity', 1 - result['distance'])
            print(f"{i}. {meta.get('name', 'unknown')} (Similarity: {similarity:.3f})")
            print(f"   FQN: {fqn}")
            print(f"   Location: {meta.get('file_path', 'unknown')}:L{meta.get('line_start', '?')}-{meta.get('line_end', '?')}")
            print(f"   Type: {'Method' if meta.get('is_method') else 'Function'}{' (async)' if meta.get('is_async') else ''}")
            
            # Show summary if available
            summary = meta.get('summary')
            if summary:
                print(f"   Summary: {summary}")
            
            # Show connections if available
            connections_in = meta.get('incoming_degree')
            connections_out = meta.get('outgoing_degree')
            if connections_in is not None and connections_in > 0:
                print(f"   Connections in: {connections_in}")
            if connections_out is not None and connections_out > 0:
                print(f"   Connections out: {connections_out}")
            if meta.get('complexity') is not None: print(f"   Complexity: {meta['complexity']}")
            if meta.get('nloc') is not None: print(f"   NLOC: {meta['nloc']}")
            
            # Helper to safely parse list fields
            def _parse_list_field(value: Any) -> List[str]:
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                if isinstance(value, list):
                    return value
                return []

            # Handle external dependencies
            ext_deps = _parse_list_field(meta.get('external_dependencies'))
            if ext_deps:
                print(f"   External Deps: {', '.join(ext_deps)}")

            # Deserialize JSON string metadata fields for display
            decorators = _parse_list_field(meta.get('decorators'))
            if decorators:
                # Decorators might be dicts or strings
                if decorators and isinstance(decorators[0], dict):
                     decorator_names = [d.get('name', str(d)) for d in decorators if isinstance(d, dict)]
                     print(f"   Decorators: {', '.join(decorator_names)}")
                else:
                     print(f"   Decorators: {', '.join([str(d) for d in decorators])}")

            catches = _parse_list_field(meta.get('catches_exceptions'))
            if catches:
                print(f"   Catches: {', '.join(catches)}")

            local_vars = _parse_list_field(meta.get('local_variables_declared'))
            if local_vars:
                print(f"   Local Vars: {', '.join(local_vars)}")
        print("-" * 20)

        if generate_mermaid:
            if not self.graph_builder.functions and not self.graph_builder.edges:
                print("\n⚠️  Cannot generate Mermaid graph without full analysis data. Please run analysis first or ensure a report was generated.", file=sys.stderr)
                return

            print(f"\n# Mermaid Call Graph for relevant functions:")
            print("```mermaid")
            print(self.graph_builder.export_mermaid_graph(highlight_fqns=highlight_fqns, llm_optimized=llm_optimized_mermaid)) # Pass new flag
            print("```")
            if not llm_optimized_mermaid:
                print("Use a Mermaid viewer (e.g., VS Code, Mermaid.live) to render this graph.")
            else:
                print("This Mermaid graph is optimized for LLM token count, visual styling is minimal/removed.")


    def export_report(self, output_path: str) -> None:
        """Analyze and export the report to a file."""
        report = self.analyze()
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 Report exported to {output_path}")
        except Exception as e:
            print(f"❌ Error writing report to {output_path}: {e}", file=sys.stderr)
        try:
            self._export_metrics(report)
        except Exception as e:
            print(f"⚠️  Skipping metrics export: {e}", file=sys.stderr)

def _memory_command(config, args) -> int:
    from code_flow.core.cortex_memory import CortexMemoryStore

    if not config.memory_enabled:
        print("❌ Cortex memory is disabled in config.", file=sys.stderr)
        return 1

    store = CortexMemoryStore(
        persist_directory=config.chromadb_path,
        embedding_model_name=resolve_embedding_model(config.embedding_model),
        collection_name=config.memory_collection_name,
    )

    if args.memory_action == "add":
        record = store.add_memory(
            content=args.content,
            memory_type=args.type.upper(),
            tags=args.tags or [],
            scope=args.scope,
            file_paths=args.file_paths or [],
            source=args.source,
            base_confidence=args.base_confidence,
            decay_half_life_days=config.memory_half_life_days.get(args.type.upper(), 30.0),
            decay_floor=config.memory_decay_floor.get(args.type.upper(), 0.05),
            metadata={},
        )
        print(f"✅ Added memory {record.knowledge_id}")
        return 0

    if args.memory_action == "reinforce":
        record = store.reinforce_memory(args.knowledge_id)
        if not record:
            print("❌ Memory not found.", file=sys.stderr)
            return 1
        print(f"✅ Reinforced memory {record.knowledge_id}")
        return 0

    if args.memory_action == "forget":
        success = store.forget_memory(args.knowledge_id)
        if not success:
            print("❌ Failed to delete memory.", file=sys.stderr)
            return 1
        print(f"✅ Deleted memory {args.knowledge_id}")
        return 0

    if args.memory_action == "query":
        results = store.query_memory(
            query=args.query,
            n_results=args.limit,
            filters={"memory_type": args.type.upper()} if args.type else {},
            similarity_weight=config.memory_similarity_weight,
            memory_score_weight=config.memory_score_weight,
        )
        print(json.dumps(results, indent=2))
        return 0

    if args.memory_action == "list":
        results = store.list_memory(
            filters={"memory_type": args.type.upper()} if args.type else {},
            limit=args.limit,
            offset=args.offset,
        )
        print(json.dumps(results, indent=2))
        return 0

    print("❌ Unknown memory action.", file=sys.stderr)
    return 1


def _add_directory_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "directory",
        nargs='?',
        default=None,
        help="Path to codebase directory. If not provided, uses 'watch_directories' from config or defaults to current directory.",
    )


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to configuration YAML file (default: codeflow.config.yaml)")
    parser.add_argument("--language", choices=["python", "typescript", "rust"],
                        help="Programming language of the codebase (overrides config)")
    parser.add_argument("--embedding-model",
                        help="Embedding model to use. Shortcuts: 'fast' (all-MiniLM-L6-v2, 384-dim), "
                             "'medium' (all-MiniLM-L12-v2, 384-dim), or 'accurate' (all-mpnet-base-v2, 768-dim). "
                             "Default: 'fast'.")
    parser.add_argument("--max-tokens", type=int,
                        help="Maximum tokens per chunk for embedding model (default: 256).")


def _resolve_config_and_root(args: argparse.Namespace):
    cli_overrides = {}
    explicit_directory = getattr(args, "directory", None)
    if getattr(args, "directory", None):
        cli_overrides['project_root'] = str(Path(args.directory).resolve())
        cli_overrides['watch_directories'] = [str(Path(args.directory).resolve())]
    if getattr(args, "language", None):
        cli_overrides['language'] = args.language
    if getattr(args, "embedding_model", None):
        cli_overrides['embedding_model'] = args.embedding_model
    if getattr(args, "max_tokens", None):
        cli_overrides['max_tokens'] = args.max_tokens

    from code_flow.core.config import load_config
    config = load_config(config_path=getattr(args, "config", None), cli_args=cli_overrides)

    root_dir = config.require_project_root()
    if explicit_directory:
        root_dir = Path(explicit_directory).resolve()
        config.project_root = str(root_dir)

    if not config.project_root:
        raise ValueError("project_root is required; set it in codeflow.config.yaml")
    return config, root_dir


def _init_analyzer(root_dir: Path, config, enable_summaries: bool) -> CodeGraphAnalyzer:
    embedding_model = resolve_embedding_model(config.embedding_model)
    analyzer = CodeGraphAnalyzer(
        root_directory=root_dir,
        language=config.language,
        embedding_model=embedding_model,
        max_tokens=config.max_tokens,
        enable_summaries=enable_summaries,
        llm_config=config.llm_config if hasattr(config, 'llm_config') else {},
        chromadb_path=str(config.chroma_dir()),
    )
    analyzer.graph_builder.confidence_threshold = float(getattr(config, "call_graph_confidence_threshold", 0.8))
    return analyzer


def _run_analyze(args: argparse.Namespace) -> int:
    config, root_dir = _resolve_config_and_root(args)
    if not root_dir.is_dir():
        print(f"❌ Error: {root_dir} is not a valid directory for analysis.", file=sys.stderr)
        return 1

    config.chroma_dir().mkdir(parents=True, exist_ok=True)
    config.reports_dir().mkdir(parents=True, exist_ok=True)

    enable_summaries = args.summaries or getattr(config, "summary_generation_enabled", False)
    analyzer = _init_analyzer(root_dir, config, enable_summaries=enable_summaries)

    output_path = args.output
    if not Path(args.output).is_absolute():
        if args.output == "code_analysis_report.json":
            output_path = str(root_dir / args.output)
        else:
            output_path = str(config.reports_dir() / args.output)

    analyzer.export_report(output_path)

    drift_enabled = bool(args.drift)
    if drift_enabled:
        drift_report = DriftAnalyzer(
            project_root=str(root_dir),
            config=config.model_dump(),
        ).analyze(
            functions=list(analyzer.graph_builder.functions.values()),
            edges=analyzer.graph_builder.get_drift_edges(),
        )
        drift_output_path = str(Path(output_path).with_suffix(".drift.json"))
        with open(drift_output_path, 'w', encoding='utf-8') as f:
            json.dump(drift_report, f, indent=2, ensure_ascii=False)
        print(f"📄 Drift report exported to {drift_output_path}")

    return 0


def _run_query(args: argparse.Namespace) -> int:
    if args.llm_optimized:
        args.mermaid = True

    config, root_dir = _resolve_config_and_root(args)
    if not root_dir.is_dir() and not args.no_analyze:
        print(f"❌ Error: {root_dir} is not a valid directory for analysis.", file=sys.stderr)
        return 1

    config.chroma_dir().mkdir(parents=True, exist_ok=True)
    analyzer = _init_analyzer(root_dir, config, enable_summaries=False)
    min_similarity = args.min_similarity if args.min_similarity is not None else getattr(config, "min_similarity", 0.0)

    if args.no_analyze:
        print(f"⏩ Skipping code analysis. Attempting to query existing vector store in '{root_dir / 'code_vectors_chroma'}'.")
        return _safe_query(
            analyzer,
            args.query,
            n_results=args.limit,
            generate_mermaid=args.mermaid,
            llm_optimized_mermaid=args.llm_optimized,
            min_similarity=min_similarity,
        )

    analyzer.analyze()
    return _safe_query(
        analyzer,
        args.query,
        n_results=args.limit,
        generate_mermaid=args.mermaid,
        llm_optimized_mermaid=args.llm_optimized,
        min_similarity=min_similarity,
    )


def _run_graphs(args: argparse.Namespace) -> int:
    if args.llm_optimized and args.format != "mermaid":
        print("❌ Error: --llm-optimized is only valid with --format mermaid.", file=sys.stderr)
        return 1

    config, root_dir = _resolve_config_and_root(args)
    if not root_dir.is_dir():
        print(f"❌ Error: {root_dir} is not a valid directory for analysis.", file=sys.stderr)
        return 1

    config.chroma_dir().mkdir(parents=True, exist_ok=True)
    analyzer = _init_analyzer(root_dir, config, enable_summaries=False)
    analyzer.analyze()

    if args.format == "mermaid":
        output = analyzer.graph_builder.export_graph(
            format="mermaid",
            fqns=args.fqns or None,
            depth=args.depth,
            llm_optimized=args.llm_optimized,
        )
    else:
        output = analyzer.graph_builder.export_graph(
            format="json",
            fqns=args.fqns or None,
            depth=args.depth,
        )

    if args.output:
        output_path = Path(args.output)
        if isinstance(output, dict):
            output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            output_path.write_text(str(output), encoding="utf-8")
        print(f"📄 Graph exported to {output_path}")
    else:
        if isinstance(output, dict):
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(output)

    return 0


def _run_drift(args: argparse.Namespace) -> int:
    config, root_dir = _resolve_config_and_root(args)
    if not root_dir.is_dir():
        print(f"❌ Error: {root_dir} is not a valid directory for analysis.", file=sys.stderr)
        return 1

    config.chroma_dir().mkdir(parents=True, exist_ok=True)
    config.reports_dir().mkdir(parents=True, exist_ok=True)
    analyzer = _init_analyzer(root_dir, config, enable_summaries=False)
    analyzer.analyze()

    drift_report = DriftAnalyzer(
        project_root=str(root_dir),
        config=config.model_dump(),
    ).analyze(
        functions=list(analyzer.graph_builder.functions.values()),
        edges=analyzer.graph_builder.get_drift_edges(),
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (config.reports_dir() / "code_analysis_report.json").with_suffix(".drift.json")

    output_path.write_text(json.dumps(drift_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"📄 Drift report exported to {output_path}")
    return 0


def _run_memory(args: argparse.Namespace) -> int:
    from code_flow.core.config import load_config
    config = load_config(config_path=args.config, cli_args={})
    return _memory_command(config, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CodeFlow CLI: analysis, search, graphs, drift detection, and cortex memory."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a codebase and generate a report")
    _add_directory_arg(analyze)
    _add_common_flags(analyze)
    analyze.add_argument("--output", default="code_analysis_report.json",
                         help="Output file for the analysis report.")
    analyze.add_argument("--summaries", action="store_true",
                         help="Enable LLM-driven summary generation (requires OPENAI_API_KEY).")
    analyze.add_argument("--drift", action="store_true",
                         help="Generate a drift report alongside the analysis report.")
    analyze.set_defaults(func=_run_analyze)

    query = subparsers.add_parser("query", help="Run a semantic query against a codebase")
    _add_directory_arg(query)
    _add_common_flags(query)
    query.add_argument("--query", required=True,
                       help="Run a semantic query against the analyzed codebase or a previously stored vector store.")
    query.add_argument("--no-analyze", action="store_true",
                       help="Do not perform analysis. Assume the vector store is already populated.")
    query.add_argument("--limit", type=int, default=5, help="Number of results to return.")
    query.add_argument("--min-similarity", type=float, default=None,
                       help="Minimum similarity threshold (0-1). Uses config min_similarity if omitted.")
    query.add_argument("--mermaid", action="store_true",
                       help="Generate a Mermaid graph for query results.")
    query.add_argument("--llm-optimized", action="store_true",
                       help="Generate Mermaid graph optimized for LLM token count (removes styling). Implies --mermaid.")
    query.set_defaults(func=_run_query)

    graphs = subparsers.add_parser("graphs", help="Export call graph data")
    _add_directory_arg(graphs)
    _add_common_flags(graphs)
    graphs.add_argument("--format", choices=["json", "mermaid"], default="json",
                        help="Graph output format.")
    graphs.add_argument("--fqns", nargs='*', default=None,
                        help="Optional list of fully qualified names to highlight (mermaid only).")
    graphs.add_argument("--depth", type=int, default=1,
                        help="Depth of the call graph to export when --fqns is provided (default: 1).")
    graphs.add_argument("--llm-optimized", action="store_true",
                        help="Generate Mermaid graph optimized for LLM token count (mermaid only).")
    graphs.add_argument("--output", help="Write graph output to a file instead of stdout.")
    graphs.set_defaults(func=_run_graphs)

    drift = subparsers.add_parser("drift", help="Generate a drift report")
    _add_directory_arg(drift)
    _add_common_flags(drift)
    drift.add_argument("--output", help="Output path for drift report (default: code_analysis_report.drift.json)")
    drift.set_defaults(func=_run_drift)

    memory = subparsers.add_parser("memory", help="Manage Cortex memory entries")
    memory.add_argument("--config", help="Path to configuration YAML file (default: codeflow.config.yaml)")
    memory_sub = memory.add_subparsers(dest="memory_action", required=True)

    memory_add = memory_sub.add_parser("add", help="Add a memory")
    memory_add.add_argument("--type", default="FACT", help="Memory type TRIBAL|EPISODIC|FACT")
    memory_add.add_argument("--content", required=True, help="Memory content")
    memory_add.add_argument("--tags", nargs='*', default=[], help="Tags")
    memory_add.add_argument("--scope", default="repo", help="Scope: repo|project|file")
    memory_add.add_argument("--file-paths", nargs='*', default=[], help="Related file paths")
    memory_add.add_argument("--source", default="user", help="Source: user|system|tool")
    memory_add.add_argument("--base-confidence", type=float, default=1.0, help="Base confidence")

    memory_query = memory_sub.add_parser("query", help="Query memory")
    memory_query.add_argument("--query", required=True, help="Query string")
    memory_query.add_argument("--type", default=None, help="Filter by memory type")
    memory_query.add_argument("--limit", type=int, default=5, help="Number of results")

    memory_list = memory_sub.add_parser("list", help="List memory")
    memory_list.add_argument("--type", default=None, help="Filter by memory type")
    memory_list.add_argument("--limit", type=int, default=50, help="Number of results")
    memory_list.add_argument("--offset", type=int, default=0, help="Offset")

    memory_reinforce = memory_sub.add_parser("reinforce", help="Reinforce memory")
    memory_reinforce.add_argument("--knowledge-id", required=True, help="Memory ID")

    memory_forget = memory_sub.add_parser("forget", help="Delete memory")
    memory_forget.add_argument("--knowledge-id", required=True, help="Memory ID")

    memory.set_defaults(func=_run_memory)

    return parser


def main():
    """Main entry point for the analyzer."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        exit_code = args.func(args)
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
