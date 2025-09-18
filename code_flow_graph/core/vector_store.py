"""
Vector store integration for code graph elements using ChromaDB.
This version is type-safe and expects enriched FunctionNode objects.
"""

from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

# Import the specific, enriched data types from the call graph builder
from core.call_graph_builder import FunctionNode, CallEdge

class CodeVectorStore:
    """Vector store for code elements with explicit indexing strategy using ChromaDB."""

    def __init__(self, persist_directory: str):
        """
        Initialize the ChromaDB vector store.

        Args:
            persist_directory: Where to persist the vector database on disk.
        """
        print(f"Initializing ChromaDB client at: {persist_directory}")
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="code_graph_v2",
                metadata={"hnsw:space": "cosine"}
            )
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"✅ ChromaDB collection '{self.collection.name}' and Sentence Transformers loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize ChromaDB or SentenceTransformer at {persist_directory}: {e}")
            print("   Please ensure you have run 'pip install chromadb sentence-transformers'")
            raise

    def add_function_node(self, node: FunctionNode, source_code: str) -> str:
        """
        Add a fully-formed FunctionNode to the vector store.

        Args:
            node: A FunctionNode object from the call graph.
            source_code: The full source code of the file containing the function.

        Returns:
            The unique ID of the stored document.
        """
        # Create a descriptive document for this function
        document = self._create_function_document(node, source_code)

        # Generate embedding
        embedding = self.embedding_model.encode([document]).tolist()[0]

        # Create rich metadata from the FunctionNode
        metadata = {
            "type": "function",
            "fully_qualified_name": node.fully_qualified_name,
            "name": node.name,
            "file_path": node.file_path,
            "line_start": node.line_start,
            "is_entry_point": node.is_entry_point,
            "is_method": node.is_method,
            "class_name": node.class_name or "N/A",
            "parameter_count": len(node.parameters),
            "is_async": node.is_async,
            "incoming_degree": len(node.incoming_edges),
            "outgoing_degree": len(node.outgoing_edges),
            "has_docstring": bool(node.docstring),
            "access_modifier": node.access_modifier or "public"
        }

        # Use a deterministic ID for idempotency
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, node.fully_qualified_name))

        # Upsert ensures that if we run the analysis again, we update existing entries
        self.collection.upsert(
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )

        return doc_id

    def _create_function_document(self, node: FunctionNode, source_code: str) -> str:
        """Create a descriptive document for vector search from a FunctionNode."""
        # Extract relevant source code snippet
        try:
            lines = source_code.split('\n')
            start_line_idx = max(0, node.line_start - 1)
            # Determine end line by looking for next function/class or end of file
            end_line_idx = node.line_end # Use the AST extractor's end_line for precision
            func_snippet = '\n'.join(lines[start_line_idx:end_line_idx]).strip()
        except Exception:
            func_snippet = "Source snippet unavailable."

        # Create comprehensive document
        document_parts = [
            f"FUNCTION: {node.fully_qualified_name}",
            f"Name: {node.name}",
            f"Parameters: {', '.join(node.parameters) if node.parameters else 'None'}",
            f"Location: {node.file_path}:{node.line_start}",
            f"Type: {'Method' if node.is_method else 'Function'}{' (async)' if node.is_async else ''}",
            f"Class: {node.class_name if node.class_name else 'N/A'}",
            f"Entry Point: {node.is_entry_point}",
            f"Connections: {len(node.incoming_edges)} in, {len(node.outgoing_edges)} out",
            f"Docstring: {node.docstring[:150] + '...' if node.docstring and len(node.docstring) > 150 else (node.docstring or 'None')}",
            f"Code Snippet: {func_snippet[:400]}..."
        ]

        return " | ".join([part for part in document_parts if part])

    def add_edge(self, edge: CallEdge) -> str:
        """Add a call edge to the vector store."""
        document = f"CALL_EDGE: From {edge.caller} to {edge.callee} at line {edge.line_number}"
        embedding = self.embedding_model.encode([document]).tolist()[0]

        metadata = {
            "type": "call_edge",
            "caller": edge.caller,
            "callee": edge.callee,
            "file_path": edge.file_path,
            "line_number": edge.line_number,
            "confidence": edge.confidence
        }

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{edge.caller}->{edge.callee}@{edge.line_number}"))

        self.collection.upsert(
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id

    def query_functions(self, query: str, n_results: int = 10, where_filter: Dict = None) -> List[Dict]:
        """
        Query for functions using semantic search.

        Args:
            query: Search query (e.g., "functions that handle user authentication").
            n_results: Number of results to return.
            where_filter: Optional ChromaDB filter (e.g., {"is_entry_point": True}).

        Returns:
            List of matching documents with metadata and distance.
        """
        if where_filter is None:
            where_filter = {"type": "function"}
        else:
            # Ensure the type filter is always applied, or combined
            if "type" not in where_filter:
                where_filter["type"] = "function"
            elif where_filter["type"] != "function":
                print("Warning: Overwriting 'type' filter to 'function' for query_functions.")
                where_filter["type"] = "function"


        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )

        formatted_results = []
        if not results or not results.get('documents') or not results['documents'][0]:
            return []

        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "document": results['documents'][0][i],
                "distance": results['distances'][0][i],
                "metadata": results['metadatas'][0][i],
                "id": results['ids'][0][i]
            })

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        count = self.collection.count()
        return {
            "total_documents": count
            # More detailed stats can be added by querying with metadata filters,
            # but this is slower. `count` is efficient.
        }
