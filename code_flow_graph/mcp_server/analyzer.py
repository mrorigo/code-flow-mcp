from pathlib import Path
from typing import Optional
import asyncio
import logging

import watchdog.observers
from watchdog.events import FileSystemEventHandler


from code_flow_graph.core.ast_extractor import PythonASTExtractor
from code_flow_graph.core.call_graph_builder import CallGraphBuilder
from code_flow_graph.core.vector_store import CodeVectorStore

class WatcherHandler(FileSystemEventHandler):
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            logging.info(f"File modified: {event.src_path}")
            asyncio.run_coroutine_threadsafe(self.analyzer._incremental_update(event.src_path), self.analyzer.loop)


class MCPAnalyzer:
    """Analyzer for MCP server that extracts, builds, and stores code graph data."""

    def __init__(self, config: dict):
        """Initialize the analyzer with configuration.

        Args:
            config: Configuration dictionary containing watch_directories and chromadb_path
        """
        self.config = config
        root = Path(config['watch_directories'][0])
        logging.info(f"Initializing MCPAnalyzer with root: {root}")
        self.extractor = PythonASTExtractor()
        self.extractor.project_root = root
        self.builder = CallGraphBuilder()
        self.builder.project_root = root
        self.vector_store: Optional[CodeVectorStore] = None
        self.observer = None

        # Initialize vector store if path exists
        if Path(config['chromadb_path']).exists():
            self.vector_store = CodeVectorStore(persist_directory=config['chromadb_path'], embedding_model_name=config['embedding_model'] if config['embedding_model'] else 'all-MiniLM-L6-v2')
        else:
            logging.warning(f"ChromaDB path {config['chromadb_path']} does not exist, skipping vector store initialization")

    async def analyze(self) -> None:
        """Analyze the codebase by extracting elements, building graph, and populating vector store."""
        # Extract code elements
        elements = await asyncio.to_thread(self.extractor.extract_from_directory, Path(self.config['watch_directories'][0]))

        # Build call graph
        self.builder.build_from_elements(elements)

        # Populate vector store if available
        if self.vector_store:
            await asyncio.to_thread(self._populate_vector_store)
        else:
            logging.info("Vector store not available, skipping population")

        # Start file watcher
        self.loop = asyncio.get_running_loop()
        observer = watchdog.observers.Observer()
        observer.schedule(WatcherHandler(analyzer=self), self.config['watch_directories'][0], recursive=True)
        observer.start()
        self.observer = observer

    def _populate_vector_store(self) -> None:
        """Populate the vector store with functions and edges from the builder."""
        graph_functions = list(self.builder.functions.values())

        # Read all source files first
        sources = {}
        for node in graph_functions:
            if node.file_path not in sources:
                try:
                    with open(node.file_path, 'r', encoding='utf-8') as f:
                        sources[node.file_path] = f.read()
                except Exception as e:
                    logging.warning(f"Could not read source file {node.file_path}: {e}")
                    sources[node.file_path] = ""

        # Batch store functions
        try:
            self.vector_store.add_function_nodes_batch(graph_functions, sources, batch_size=512)
        except Exception as e:
            logging.warning(f"Batch function storage failed, falling back to individual: {e}")
            for node in graph_functions:
                try:
                    source = sources.get(node.file_path, "")
                    self.vector_store.add_function_node(node, source)
                except Exception as e2:
                    logging.warning(f"Could not process/store node {node.fully_qualified_name}: {e2}")

        # Batch store edges
        try:
            self.vector_store.add_edges_batch(self.builder.edges, batch_size=512)
        except Exception as e:
            logging.warning(f"Batch edge storage failed, falling back to individual: {e}")
            for edge in self.builder.edges:
                try:
                    self.vector_store.add_edge(edge)
                except Exception as e2:
                    logging.warning(f"Could not add edge {edge.caller} -> {edge.callee}: {e2}")

    async def _incremental_update(self, file_path: str):
        logging.info(f"Starting incremental update for {file_path}")
        await asyncio.sleep(1)  # Debounce stub
        elements = await asyncio.to_thread(self.extractor.extract_from_file, Path(file_path))
        logging.info(f"Extracted {len(elements)} elements from {file_path}")
        # Update builder/store incrementally (add new, skip same hash); if delete, remove by fqn.
        # For simplicity, re-analyze the file and update
        # Assuming elements have hash or something, but for now, just add/update
        for element in elements:
            if element.fqn not in self.builder.functions:
                self.builder.functions[element.fqn] = element
            # For vector store, if available, add if not present
            if self.vector_store and element.fqn not in [n.fqn for n in self.vector_store.get_all_nodes()]:
                with open(element.file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                self.vector_store.add_function_node(element, source)