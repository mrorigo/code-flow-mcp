"""
Main orchestrator for code graph analysis with corrected pipeline logic.
"""

from tqdm import tqdm
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import argparse

# Ensure local modules can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ast_extractor import PythonASTExtractor, TypeScriptASTExtractor, FunctionElement, ClassElement
from core.call_graph_builder import CallGraphBuilder
from core.vector_store import CodeVectorStore
from core.ast_extractor import CodeElement # Import CodeElement for typing hints

class CodeGraphAnalyzer:
    """Main analyzer that orchestrates the entire pipeline."""

    def __init__(self, root_directory: Path, language: str = "python"):
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

        # Vector store path derived from root_directory for project-specific persistence
        vector_store_path = self.root_directory / "code_vectors_chroma"
        try:
            self.vector_store = CodeVectorStore(persist_directory=str(vector_store_path))
        except Exception as e:
            print(f"Could not start CodeVectorStore at {vector_store_path}. Vector-based features will be disabled. Error: {e}")
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

        # Step 4: Generate report
        print("\n📊 Step 4: Generating analysis report...")
        report = self._generate_report(classes)

        print("\n✅ Analysis complete!")
        return report

    def _populate_vector_store(self) -> None:
        """
        Populate the vector store with FunctionNode objects and edges from the graph.
        This method uses the CORRECT data type (FunctionNode) after the graph is built.
        """
        if not self.vector_store:
            # Should not be reached if self.vector_store is checked before call
            raise ValueError("Vector store is not initialized.")
        if not self.graph_builder.functions:
            print("   No functions found in graph builder, skipping vector store population.")
            return

        graph_functions = list(self.graph_builder.functions.values())
        print(f"   Storing {len(graph_functions)} functions and {len(self.graph_builder.edges)} edges...")

        for node in tqdm(graph_functions, desc="Storing functions"):
            try:
                # Read source code from the file path associated with the node
                with open(node.file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                self.vector_store.add_function_node(node, source_code)
            except FileNotFoundError:
                print(f"   Warning: Source file for node {node.fully_qualified_name} not found at {node.file_path}. Skipping.")
            except Exception as e:
                print(f"   Warning: Could not process/store node {node.fully_qualified_name}: {e}")

        for edge in tqdm(self.graph_builder.edges, desc="Storing edges"):
            try:
                self.vector_store.add_edge(edge)
            except Exception as e:
                print(f"   Warning: Could not add edge {edge.caller} -> {edge.callee}: {e}")

        stats = self.vector_store.get_stats()
        print(f"   Vector store populated. Total documents: {stats.get('total_documents', 'N/A')}")

    def _generate_report(self, classes: List[ClassElement]) -> Dict[str, Any]:
        """Generate a comprehensive analysis report using graph data."""
        # Ensure graph_builder has data, especially if analysis was skipped for some reason
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
                    "is_method": ep.is_method,
                    "class_name": ep.class_name,
                    "is_async": ep.is_async,
                    "has_docstring": bool(ep.docstring),
                    "incoming_connections": len(ep.incoming_edges),
                    "outgoing_connections": len(ep.outgoing_edges),
                    "parameters": ep.parameters,
                }
                for ep in graph_entry_points
            ],
            "classes_summary": {
                "total": len(classes),
                "with_methods": sum(1 for c in classes if c.methods),
                "with_inheritance": sum(1 for c in classes if c.extends),
            },
            "call_graph": self.graph_builder.export_graph(),
        }
        return report

    def query(self, question: str, n_results: int = 5) -> None:
        """Query the vector store for insights and print the results."""
        if not self.vector_store:
            print("Vector store is not available. Cannot perform query.")
            return

        print(f"\n🔍 Running semantic search for: '{question}'")
        results = self.vector_store.query_functions(question, n_results)

        if not results:
            print("   No relevant functions found. Try a different query.")
            return

        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            print("-" * 20)
            print(f"{i}. {meta.get('name')} (Similarity: {1 - result['distance']:.3f})")
            print(f"   FQN: {meta.get('fully_qualified_name')}")
            print(f"   Location: {meta.get('file_path')}:{meta.get('line_start')}")
            print(f"   Type: {'Method' if meta.get('is_method') else 'Function'}{' (async)' if meta.get('is_async') else ''}")
            print(f"   Connections: {meta.get('incoming_degree')} in, {meta.get('outgoing_degree')} out")
        print("-" * 20)

    def export_report(self, output_path: str) -> None:
        """Analyze and export the report to a file."""
        report = self.analyze()
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 Report exported to {output_path}")
        except Exception as e:
            print(f"❌ Error writing report to {output_path}: {e}")

def main():
    """Main entry point for the analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze a codebase to build a call graph and identify entry points. "
                    "Optionally query an existing analysis."
    )
    # Directory is now a positional argument, not an optional flag
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

    args = parser.parse_args()

    root_dir = Path(args.directory).resolve()

    # Validate directory only if analysis is being performed
    if not args.no_analyze and not root_dir.is_dir():
        print(f"❌ Error: {root_dir} is not a valid directory for analysis.")
        sys.exit(1)

    # Enforce --no-analyze with --query
    if args.no_analyze and not args.query:
        print("❌ Error: The --no-analyze flag must be used with --query.")
        sys.exit(1)

    try:
        analyzer = CodeGraphAnalyzer(root_dir, args.language)

        if args.no_analyze:
            print(f"⏩ Skipping code analysis. Attempting to query existing vector store in '{root_dir / 'code_vectors_chroma'}'.")
            analyzer.query(args.query)
        elif args.query:
            # Analyze first to populate/update the store, then query
            analyzer.analyze()
            analyzer.query(args.query)
        else:
            # Just generate the report (implies analysis)
            analyzer.export_report(args.output)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
