"""
Main orchestrator for code graph analysis with corrected pipeline logic.
"""

from tqdm import tqdm
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import argparse

# Ensure local modules can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.python_extractor import PythonASTExtractor
from core.typescript_extractor import TypeScriptASTExtractor
from core.models import FunctionElement, ClassElement, CodeElement
from core.call_graph_builder import CallGraphBuilder, FunctionNode # Import FunctionNode here
from core.vector_store import CodeVectorStore

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


class CodeGraphAnalyzer:
    """Main analyzer that orchestrates the entire pipeline."""

    def __init__(self, root_directory: Path, language: str = "python", embedding_model: str = 'all-MiniLM-L6-v2', max_tokens: int = 256):
        """
        Initialize the analyzer.

        Args:
            root_directory: Root of the codebase. This is also used to derive the
                            persistence directory for the vector store.
            language: 'python' or 'typescript'.
        """
        self.root_directory = root_directory
        self.language = language.lower()
        self.extractor = self._get_extractor()
        self.graph_builder = CallGraphBuilder()
        # Pass the root_directory to the graph_builder for consistent FQN generation
        self.graph_builder.project_root = root_directory

        # Vector store path derived from root_directory for project-specific persistence
        vector_store_path = self.root_directory / "code_vectors_chroma"
        try:
            self.vector_store = CodeVectorStore(persist_directory=str(vector_store_path), embedding_model_name=embedding_model, max_tokens=max_tokens)
        except Exception as e:
            print(f"Could not start CodeVectorStore at {vector_store_path}. Vector-based features will be disabled. Error: {e}", file=sys.stderr)
            self.vector_store = None

        self.all_elements: List[CodeElement] = []
        self.classes: Dict[str, ClassElement] = {}

    def _get_extractor(self):
        """Get the appropriate AST extractor for the language."""
        if self.language == "python":
            return PythonASTExtractor()
        elif self.language == "typescript":
            return TypeScriptASTExtractor()
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
        functions = [e for e in self.all_elements if hasattr(e, 'kind') and e.kind == 'function']
        classes = [e for e in self.all_elements if hasattr(e, 'kind') and e.kind == 'class']
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

        # Step 4: Generate report
        print("\n📊 Step 4: Generating analysis report...")
        report = self._generate_report(classes, functions)

        print("\n✅ Analysis complete!")
        return report

    def _populate_vector_store(self) -> None:
        """
        Populate the vector store with FunctionNode objects and edges from the graph.
        This method uses the CORRECT data type (FunctionNode) after the graph is built.
        """
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")
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

    def _generate_report(self, classes: List[ClassElement], functions: List[FunctionElement]) -> Dict[str, Any]:
        """Generate a comprehensive analysis report using graph data."""
        if not self.graph_builder.functions:
            return {
                "summary": {"total_functions": 0, "total_edges": 0},
                "entry_points": [],
                "classes_summary": {"total": 0},
                "call_graph": {"functions": {}, "edges": {}}
            }

        graph_entry_points = self.graph_builder.get_entry_points()

        report = {
            "summary": self.graph_builder.export_graph()['summary'],
            "entry_points": [
                {
                    "name": ep.name,
                    "fully_qualified_name": ep.fully_qualified_name,
                    "file_path": ep.file_path,
                    "line_number": ep.line_start,
                    "line_end": ep.line_end,
                    "is_method": ep.is_method,
                    "class_name": ep.class_name,
                    "is_async": ep.is_async,
                    "has_docstring": bool(ep.docstring),
                    "incoming_connections": len(ep.incoming_edges),
                    "outgoing_connections": len(ep.outgoing_edges),
                    "parameters": ep.parameters,
                    "complexity": ep.complexity,
                    "nloc": ep.nloc,
                    "external_dependencies": ep.external_dependencies,
                    "decorators": ep.decorators,
                    "catches_exceptions": ep.catches_exceptions,
                    "local_variables_declared": ep.local_variables_declared,
                    "hash_body": ep.hash_body,
                }
                for ep in graph_entry_points
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
            "call_graph": self.graph_builder.export_graph(),
        }
        return report

    def query(self, question: str, n_results: int = 5, generate_mermaid: bool = False, llm_optimized_mermaid: bool = False) -> None: # NEW: llm_optimized_mermaid flag
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
        results = self.vector_store.query_functions(question, n_results)

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

            print(f"{i}. {meta.get('name')} (Similarity: {1 - result['distance']:.3f})")
            print(f"   FQN: {fqn}")
            print(f"   Location: {meta.get('file_path')}:L{meta.get('line_start')}-{meta.get('line_end')}")
            print(f"   Type: {'Method' if meta.get('is_method') else 'Function'}{' (async)' if meta.get('is_async') else ''}")
            if meta.get('incoming_degree', 0) > 0:
                print(f"   Connections in: {meta.get('incoming_degree')}")
            if meta.get('outgoing_degree', 0) > 0:
                print(f"   Connections out: {meta.get('outgoing_degree')}")
            if meta.get('complexity') is not None: print(f"   Complexity: {meta['complexity']}")
            if meta.get('nloc') is not None: print(f"   NLOC: {meta['nloc']}")
            if meta.get('external_dependencies'): print(f"   External Deps: {', '.join(meta['external_dependencies'])}")
            # Deserialize JSON string metadata fields for display
            decorators = meta.get('decorators')
            if decorators:
                try:
                    if isinstance(decorators, str):
                        decorators = json.loads(decorators)
                    decorator_names = [d.get('name', 'unknown') if isinstance(d, dict) else str(d) for d in decorators]
                    print(f"   Decorators: {', '.join(decorator_names)}")
                except (json.JSONDecodeError, AttributeError):
                    print(f"   Decorators: {decorators}")

            catches = meta.get('catches_exceptions')
            if catches:
                try:
                    if isinstance(catches, str):
                        catches = json.loads(catches)
                    print(f"   Catches: {', '.join(catches) if isinstance(catches, list) else catches}")
                except (json.JSONDecodeError, AttributeError):
                    print(f"   Catches: {catches}")

            local_vars = meta.get('local_variables_declared')
            if local_vars:
                try:
                    if isinstance(local_vars, str):
                        local_vars = json.loads(local_vars)
                    print(f"   Local Vars: {', '.join(local_vars) if isinstance(local_vars, list) else local_vars}")
                except (json.JSONDecodeError, AttributeError):
                    print(f"   Local Vars: {local_vars}")

            # Handle external dependencies
            ext_deps = meta.get('external_dependencies')
            if ext_deps:
                try:
                    if isinstance(ext_deps, str):
                        ext_deps = json.loads(ext_deps)
                    print(f"   External Deps: {', '.join(ext_deps) if isinstance(ext_deps, list) else ext_deps}")
                except (json.JSONDecodeError, AttributeError):
                    print(f"   External Deps: {ext_deps}")
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

def main():
    """Main entry point for the analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze a codebase to build a call graph and identify entry points. "
                    "Optionally query an existing analysis."
    )
    parser.add_argument("directory", nargs='?', default='.',
                        help="Path to codebase directory (default: current directory). "
                             "This directory is also used to locate the vector store "
                             "(e.g., <directory>/code_vectors_chroma/).")
    parser.add_argument("--language", choices=["python", "typescript"], default="python",
                        help="Programming language of the codebase")
    parser.add_argument("--output", default="code_analysis_report.json",
                        help="Output file for the analysis report (only used when performing analysis).")
    parser.add_argument("--query",
                        help="Run a semantic query against the analyzed codebase or a previously stored vector store.")
    parser.add_argument("--no-analyze", action="store_true",
                        help="Do not perform analysis. Assume the vector store is already populated "
                             "from a previous run in the specified directory. This flag must be "
                             "used with --query.")
    parser.add_argument("--mermaid", action="store_true",
                        help="Generate a Mermaid graph for query results (requires --query).")
    parser.add_argument("--llm-optimized", action="store_true",
                        help="Generate Mermaid graph optimized for LLM token count (removes styling). Implies --mermaid.")
    parser.add_argument("--embedding-model", default="fast",
                        help="Embedding model to use. Shortcuts: 'fast' (all-MiniLM-L6-v2, 384-dim), "
                             "'medium' (all-MiniLM-L12-v2, 384-dim), or 'accurate' (all-mpnet-base-v2, 768-dim). "
                             "You can also specify any SentenceTransformer model name directly. Default: 'fast'. "
                             "See https://www.sbert.net/docs/pretrained_models.html for more options.")
    parser.add_argument("--max-tokens", type=int, default=256,
                      help="Maximum tokens per chunk for embedding model (default: 256). Adjust based on model.")

    args = parser.parse_args()

    root_dir = Path(args.directory).resolve()

    if not args.no_analyze and not root_dir.is_dir():
        print(f"❌ Error: {root_dir} is not a valid directory for analysis.", file=sys.stderr)
        sys.exit(1)

    if args.no_analyze and not args.query:
        print("❌ Error: The --no-analyze flag must be used with --query.", file=sys.stderr)
        sys.exit(1)

    # If --llm-optimized is set, implicitly set --mermaid
    if args.llm_optimized:
        args.mermaid = True

    if args.mermaid and not args.query:
        print("❌ Error: The --mermaid flag must be used with --query.", file=sys.stderr)
        sys.exit(1)

    try:
        # Resolve embedding model shorthand to actual model name
        embedding_model = resolve_embedding_model(args.embedding_model)
        analyzer = CodeGraphAnalyzer(root_dir, args.language, embedding_model, args.max_tokens)

        if args.no_analyze:
            print(f"⏩ Skipping code analysis. Attempting to query existing vector store in '{root_dir / 'code_vectors_chroma'}'.")
            analyzer.query(args.query, generate_mermaid=args.mermaid, llm_optimized_mermaid=args.llm_optimized)
        elif args.query:
            # Analyze first to populate/update the store, then query
            analyzer.analyze()
            analyzer.query(args.query, generate_mermaid=args.mermaid, llm_optimized_mermaid=args.llm_optimized)
        else:
            # Just generate the report (implies analysis)
            # Ensure output file is created in the analyzed directory if no explicit output path given
            output_path = args.output
            if not Path(args.output).is_absolute():
                output_path = str(root_dir / args.output)

            analyzer.export_report(output_path)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
