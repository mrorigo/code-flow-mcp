from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
import logging
import threading
import time
from enum import Enum

import watchdog.observers
from watchdog.events import FileSystemEventHandler


from code_flow_graph.core.python_extractor import PythonASTExtractor
from code_flow_graph.core.typescript_extractor import TypeScriptASTExtractor
from code_flow_graph.core.call_graph_builder import CallGraphBuilder
from code_flow_graph.core.vector_store import CodeVectorStore

class AnalysisState(Enum):
    """Enum representing the current state of code analysis."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class WatcherHandler(FileSystemEventHandler):
    def __init__(self, analyzer):
        self.analyzer = analyzer
        # Get supported file extensions based on language
        language = self.analyzer.config.get('language', 'python').lower()
        if language == 'typescript':
            self.supported_extensions = ['.py', '.ts', '.tsx', '.js', '.jsx']  # Support both for mixed projects
        else:
            self.supported_extensions = ['.py']

    def on_modified(self, event):
        if any(event.src_path.endswith(ext) for ext in self.supported_extensions):
            logging.info(f"File modified: {event.src_path}")
            asyncio.run_coroutine_threadsafe(self.analyzer._incremental_update(event.src_path), self.analyzer.loop)


class MCPAnalyzer:
    """Analyzer for MCP server that extracts, builds, and stores code graph data."""

    def __init__(self, config: dict):
        """Initialize the analyzer with configuration.

        Args:
            config: Configuration dictionary containing watch_directories, chromadb_path, and optional language
        """
        self.config = config
        root = Path(config['watch_directories'][0]).resolve()
        language = config.get('language', 'python').lower()
        logging.info(f"Initializing MCPAnalyzer with root: {root}, language: {language}")
        
        # Initialize appropriate extractor based on language
        if language == 'typescript':
            self.extractor = TypeScriptASTExtractor()
        else:
            self.extractor = PythonASTExtractor()
        
        self.extractor.project_root = root
        self.builder = CallGraphBuilder()
        self.builder.project_root = root
        self.vector_store: Optional[CodeVectorStore] = None
        self.observer = None

        # Analysis state tracking
        self.analysis_state = AnalysisState.NOT_STARTED
        self.analysis_task: Optional[asyncio.Task] = None
        self.analysis_error: Optional[Exception] = None

        # Initialize vector store if path exists
        if Path(config['chromadb_path']).exists():
            self.vector_store = CodeVectorStore(
                persist_directory=config['chromadb_path'],
                embedding_model_name=config.get('embedding_model', 'all-MiniLM-L6-v2'),
                max_tokens=config.get('max_tokens', 256)
            )
        else:
            logging.warning(f"ChromaDB path {config['chromadb_path']} does not exist, skipping vector store initialization")

        # Background cleanup configuration
        self.cleanup_interval = config.get('cleanup_interval_minutes', 30)  # Default: 30 minutes
        self.cleanup_task = None
        self.cleanup_shutdown_event = threading.Event()

    def start_background_cleanup(self) -> None:
        """
        Start the background cleanup task that periodically removes stale file references.
        This runs in a separate thread to avoid blocking the main event loop.
        """
        if not self.vector_store:
            logging.info("Vector store not available, skipping background cleanup")
            return

        def cleanup_worker():
            """Background worker function that runs the cleanup periodically."""
            while not self.cleanup_shutdown_event.is_set():
                try:
                    # Run cleanup if vector store is available
                    if self.vector_store:
                        logging.info("Running background cleanup of stale file references")
                        cleanup_stats = self.vector_store.cleanup_stale_references()
                        if cleanup_stats['removed_documents'] > 0:
                            logging.info(f"Cleanup removed {cleanup_stats['removed_documents']} stale references")

                    # Wait for next cleanup interval or shutdown event
                    self.cleanup_shutdown_event.wait(timeout=self.cleanup_interval * 60)

                except Exception as e:
                    logging.error(f"Error in background cleanup: {e}")
                    # Wait a bit before retrying on error
                    time.sleep(60)

        # Start cleanup thread
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        self.cleanup_task = cleanup_thread
        logging.info(f"Background cleanup started (interval: {self.cleanup_interval} minutes)")

    def stop_background_cleanup(self) -> None:
        """Stop the background cleanup task."""
        if self.cleanup_task and self.cleanup_task.is_alive():
            self.cleanup_shutdown_event.set()
            self.cleanup_task.join(timeout=10)  # Wait up to 10 seconds
            logging.info("Background cleanup stopped")

    def is_ready(self) -> bool:
        """Check if analysis is complete and ready for queries.
        
        Returns:
            True if analysis is completed, False otherwise
        """
        return self.analysis_state == AnalysisState.COMPLETED

    async def wait_for_analysis(self, timeout: Optional[float] = None) -> bool:
        """Wait for analysis to complete.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
            
        Returns:
            True if analysis completed successfully, False if timeout or failed
        """
        if self.analysis_state == AnalysisState.COMPLETED:
            return True
        
        if self.analysis_task is None:
            return False
            
        try:
            await asyncio.wait_for(self.analysis_task, timeout=timeout)
            return self.analysis_state == AnalysisState.COMPLETED
        except asyncio.TimeoutError:
            return False

    async def start_analysis(self) -> None:
        """Start code analysis in the background.
        
        This method starts the analysis process as a background task,
        allowing the server to become available immediately.
        """
        if self.analysis_state != AnalysisState.NOT_STARTED:
            logging.warning(f"Analysis already started (state: {self.analysis_state.value})")
            return
        
        async def run_analysis():
            """Background task that runs the analysis."""
            try:
                self.analysis_state = AnalysisState.IN_PROGRESS
                logging.info("Background analysis started")
                await self.analyze()
                self.analysis_state = AnalysisState.COMPLETED
                logging.info(f"Background analysis completed: indexed {len(self.builder.functions)} functions")
            except Exception as e:
                self.analysis_state = AnalysisState.FAILED
                self.analysis_error = e
                logging.error(f"Background analysis failed: {e}", exc_info=True)
        
        # Create the background task
        self.analysis_task = asyncio.create_task(run_analysis())
        logging.info("Analysis task started in background")

    async def analyze(self) -> None:
        """Analyze the codebase by extracting elements, building graph, and populating vector store."""
        # Extract code elements from all watch directories
        all_elements = []
        
        for watch_dir in self.config['watch_directories']:
            logging.info(f"Extracting elements from directory: {watch_dir}")
            elements = await asyncio.to_thread(self.extractor.extract_from_directory, Path(watch_dir))
            logging.info(f"Found {len(elements)} elements in {watch_dir}")
            all_elements.extend(elements)

        logging.info(f"Total elements extracted from {len(self.config['watch_directories'])} directories: {len(all_elements)}")

        # Build call graph with all elements
        self.builder.build_from_elements(all_elements)

        # Populate vector store if available
        if self.vector_store:
            await asyncio.to_thread(self._populate_vector_store)
        else:
            logging.info("Vector store not available, skipping population")

        # Start file watchers for all directories
        self.loop = asyncio.get_running_loop()
        observer = watchdog.observers.Observer()
        
        for watch_dir in self.config['watch_directories']:
            observer.schedule(WatcherHandler(analyzer=self), watch_dir, recursive=True)
            logging.info(f"Started watching directory: {watch_dir}")
        
        observer.start()
        self.observer = observer

        # Start background cleanup task
        self.start_background_cleanup()

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

    async def cleanup_stale_references(self) -> Dict[str, Any]:
        """
        Manually trigger cleanup of stale file references.
        Useful for immediate cleanup without waiting for the background task.

        Returns:
            Dict with cleanup statistics
        """
        if not self.vector_store:
            return {'removed_documents': 0, 'errors': 0, 'message': 'Vector store not available'}

        return self.vector_store.cleanup_stale_references()

    def shutdown(self) -> None:
        """Shutdown the analyzer and cleanup resources."""
        # Stop background cleanup
        self.stop_background_cleanup()

        # Stop file watcher
        if self.observer:
            self.observer.stop()
            self.observer.join()