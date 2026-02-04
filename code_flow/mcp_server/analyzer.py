from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import asyncio
import logging
import threading
import time
from enum import Enum

import watchdog.observers
from watchdog.events import FileSystemEventHandler

from code_flow.core.utils import get_gitignore_patterns, match_file_against_pattern


from code_flow.core.treesitter.python_extractor import TreeSitterPythonExtractor
from code_flow.core.treesitter.typescript_extractor import TreeSitterTypeScriptExtractor
from code_flow.core.treesitter.rust_extractor import TreeSitterRustExtractor
from code_flow.core.treesitter.rust_extractor import TreeSitterRustExtractor
from code_flow.core.structured_extractor import StructuredDataExtractor
from code_flow.core.call_graph_builder import CallGraphBuilder
from code_flow.core.vector_store import CodeVectorStore
from code_flow.core.cortex_memory import CortexMemoryStore
from code_flow.core.llm_summary import SummaryGenerator, SummaryProcessor

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
            self.supported_extensions = ['.py', '.ts', '.tsx', '.js', '.jsx', '.json', '.yaml', '.yml']  # Support both for mixed projects
        elif language == 'rust':
            self.supported_extensions = ['.rs', '.json', '.yaml', '.yml']
        else:
            self.supported_extensions = ['.py', '.json', '.yaml', '.yml']

    def on_modified(self, event):
        if any(event.src_path.endswith(ext) for ext in self.supported_extensions):
            file_path = Path(event.src_path)
            if not file_path.is_absolute():
                file_path = (self.analyzer.project_root / file_path).resolve()
            if self.analyzer._is_ignored_path(file_path):
                logging.info("Ignoring gitignored path: %s", file_path)
                return

            logging.info(f"File modified: {event.src_path}")
            loop = getattr(self.analyzer, "loop", None)
            if not loop or loop.is_closed() or not loop.is_running():
                logging.debug("Skipping incremental update: event loop is not available")
                return
            asyncio.run_coroutine_threadsafe(
                self.analyzer._incremental_update(event.src_path),
                loop,
            )


class MCPAnalyzer:
    """Analyzer for MCP server that extracts, builds, and stores code graph data."""

    def __init__(self, config: dict):
        """Initialize the analyzer with configuration.

        Args:
            config: Configuration dictionary containing watch_directories, chroma_dir, and optional language
        """
        self.config = config
        root = Path(config.get('project_root', '.')).resolve()
        self.project_root = root
        language = config.get('language', 'python').lower()
        logging.info(f"Initializing MCPAnalyzer with root: {root}, language: {language}")
        
        # Initialize appropriate extractor based on language
        if language == 'typescript':
            self.extractor = TreeSitterTypeScriptExtractor()
        elif language == 'rust':
            self.extractor = TreeSitterRustExtractor()
        else:
            self.extractor = TreeSitterPythonExtractor()
        
        self.extractor.project_root = root
        
        # Initialize structured data extractor
        ignored_filenames = set(config.get('ignored_filenames', []))
        self.structured_extractor = StructuredDataExtractor(ignored_filenames=ignored_filenames)
        
        self.builder = CallGraphBuilder()
        self.builder.project_root = root
        self.builder.confidence_threshold = float(config.get("call_graph_confidence_threshold", 0.8))
        self.vector_store: Optional[CodeVectorStore] = None
        self.memory_store: Optional[CortexMemoryStore] = None
        self.observer = None

        # Analysis state tracking
        self.analysis_state = AnalysisState.NOT_STARTED
        self.analysis_task: Optional[asyncio.Task] = None
        self.analysis_error: Optional[Exception] = None

        # Track changed files since last analysis (for impact analysis)
        self.changed_files_since_analysis: Dict[str, float] = {}

        # Initialize vector store if path exists
        chroma_dir = config.get('chroma_dir')
        if chroma_dir:
            chroma_path = Path(chroma_dir)
            chroma_path.mkdir(parents=True, exist_ok=True)
            self.vector_store = CodeVectorStore(
                persist_directory=str(chroma_path),
                embedding_model_name=config.get('embedding_model', 'all-MiniLM-L6-v2'),
                max_tokens=config.get('max_tokens', 256)
            )

        # Initialize memory store if enabled
        if config.get('memory_enabled', True):
            memory_dir = config.get('memory_dir')
            if memory_dir:
                memory_path = Path(memory_dir)
                memory_path.mkdir(parents=True, exist_ok=True)
                self.memory_store = CortexMemoryStore(
                    persist_directory=str(memory_path),
                    embedding_model_name=config.get('embedding_model', 'all-MiniLM-L6-v2'),
                    collection_name=config.get('memory_collection_name', 'cortex_memory_v1'),
                )
        elif not config.get('memory_enabled', True):
            logging.info("Cortex memory disabled")

        # Background cleanup configuration
        self.cleanup_interval = config.get('cleanup_interval_minutes', 30)  # Default: 30 minutes
        self.memory_cleanup_interval_seconds = config.get('memory_cleanup_interval_seconds', 3600)
        self.memory_grace_seconds = config.get('memory_grace_seconds', 86400)
        self.memory_min_score = config.get('memory_min_score', 0.1)
        self.cleanup_task = None
        self.memory_cleanup_task = None
        self.cleanup_shutdown_event = threading.Event()
        self.memory_cleanup_shutdown_event = threading.Event()

        # Initialize Summary Processor if enabled
        self.summary_processor = None
        if config.get('summary_generation_enabled', False):
            if not self.vector_store:
                logging.warning("Summary generation enabled but vector store is not initialized, disabling summary generation")
            else:
                llm_config = config.get('llm_config', {})
                self.summary_generator = SummaryGenerator(config)
                self.summary_processor = SummaryProcessor(
                    generator=self.summary_generator,
                    builder=self.builder,
                    vector_store=self.vector_store,
                    concurrency=llm_config.get('concurrency', 5),
                    prioritize_entry_points=llm_config.get('prioritize_entry_points', False)
                )
        else:
            logging.info("Summary generation disabled")

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

        if self.memory_store:
            def memory_cleanup_worker():
                """Background worker function that prunes stale cortex memory."""
                while not self.memory_cleanup_shutdown_event.is_set() and self.memory_store:
                    try:
                        logging.info("Running background cleanup of cortex memory")
                        stats = self.memory_store.cleanup_stale_memory(
                            min_score=self.memory_min_score,
                            grace_seconds=self.memory_grace_seconds,
                        )
                        if stats.get("removed", 0) > 0:
                            logging.info(f"Memory cleanup removed {stats['removed']} entries")
                        self.memory_cleanup_shutdown_event.wait(timeout=self.memory_cleanup_interval_seconds)
                    except Exception as e:
                        logging.error(f"Error in memory cleanup: {e}")
                        time.sleep(60)

            memory_cleanup_thread = threading.Thread(target=memory_cleanup_worker, daemon=True)
            memory_cleanup_thread.start()
            self.memory_cleanup_task = memory_cleanup_thread
            logging.info(
                f"Memory cleanup started (interval: {self.memory_cleanup_interval_seconds} seconds)"
            )

    def stop_background_cleanup(self) -> None:
        """Stop the background cleanup task."""
        if self.cleanup_task and self.cleanup_task.is_alive():
            self.cleanup_shutdown_event.set()
            self.cleanup_task.join(timeout=10)  # Wait up to 10 seconds
            logging.info("Background cleanup stopped")

        if self.memory_cleanup_task and self.memory_cleanup_task.is_alive():
            self.memory_cleanup_shutdown_event.set()
            self.memory_cleanup_task.join(timeout=10)
            logging.info("Memory cleanup stopped")

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
        
        # Start summary processor if enabled
        if self.summary_processor:
            self.summary_processor.start()
            
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

            # Extract structured data
            logging.info(f"Extracting structured data from directory: {watch_dir}")
            structured_elements = await asyncio.to_thread(self.structured_extractor.extract_from_directory, Path(watch_dir))
            logging.info(f"Found {len(structured_elements)} structured elements in {watch_dir}")
            
            # Add structured elements to vector store immediately (they don't need graph building)
            if self.vector_store and structured_elements:
                await asyncio.to_thread(self.vector_store.add_structured_elements_batch, structured_elements)

        logging.info(f"Total code elements extracted from {len(self.config['watch_directories'])} directories: {len(all_elements)}")

        # Build call graph with all elements
        self.builder.build_from_elements(all_elements)

        # Reset change tracking after a full analysis run
        self.changed_files_since_analysis.clear()

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

        # Enqueue missing summaries if processor is active
        if self.summary_processor and self.vector_store:
            logging.info("Checking for nodes missing summaries...")
            missing_fqns = await asyncio.to_thread(self.vector_store.get_nodes_missing_summary)
            if missing_fqns:
                logging.info(f"Found {len(missing_fqns)} nodes missing summaries, enqueueing...")
                for fqn in missing_fqns:
                    self.summary_processor.enqueue(fqn)
            
            # Also enqueue all newly added functions from this analysis run
            # (Optimization: get_nodes_missing_summary might cover this if we did it after population)
            # But since we just populated, they are likely missing summaries unless they were already there.
            # Actually, get_nodes_missing_summary is the source of truth.
            pass

    def _populate_vector_store(self) -> None:
        """Populate the vector store with functions and edges from the builder."""
        graph_functions = list(self.builder.functions.values())

        if not self.vector_store:
            logging.warning("Vector store not initialized, cannot populate")
            return

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
        resolved_path = Path(file_path)
        if not resolved_path.is_absolute():
            resolved_path = (self.project_root / resolved_path).resolve()
        if self._is_ignored_path(resolved_path):
            logging.info("Skipping incremental update for gitignored path: %s", resolved_path)
            return

        logging.info(f"Starting incremental update for {file_path}")
        self._record_changed_file(resolved_path)
        await asyncio.sleep(1)  # Debounce stub
        
        # Handle structured data files
        if any(file_path.endswith(ext) for ext in ['.json', '.yaml', '.yml']):
            elements = await asyncio.to_thread(self.structured_extractor.extract_from_file, Path(file_path))
            logging.info(f"Extracted {len(elements)} structured elements from {file_path}")
            if self.vector_store:
                await asyncio.to_thread(self.vector_store.add_structured_elements_batch, elements)
            return

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
                
                # Enqueue for summarization if enabled
                if self.summary_processor:
                    self.summary_processor.enqueue(element.fqn)

    def _is_ignored_path(self, file_path: Path) -> bool:
        patterns_with_dirs = get_gitignore_patterns(file_path.parent)
        return any(
            match_file_against_pattern(file_path, pattern, gitignore_dir, self.project_root)
            for pattern, gitignore_dir in patterns_with_dirs
        )

    def _record_changed_file(self, file_path: Path) -> None:
        normalized = file_path.resolve().as_posix()
        self.changed_files_since_analysis[normalized] = datetime.now(timezone.utc).timestamp()

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
            
        # Stop summary processor
        if self.summary_processor:
            # We need to run async stop in a sync context or fire and forget
            # Since shutdown is sync, we can try to run it if loop is running
            try:
                if self.loop and self.loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.summary_processor.stop(), self.loop)
            except Exception as e:
                logging.error(f"Error stopping summary processor: {e}")
