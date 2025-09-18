"""
Vector store integration for code graph elements using ChromaDB.
This version is type-safe and expects enriched FunctionNode objects.
"""

from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
import json # For serializing complex metadata

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
                metadata={"hnsw:space": "cosine"} # Using cosine distance for semantic similarity
            )
            # Consider using a smaller, faster model if performance is critical for embedding
            # e.g., 'all-MiniLM-L6-v2', 'all-distilroberta-v1'
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"✅ ChromaDB collection '{self.collection.name}' and Sentence Transformers loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize ChromaDB or SentenceTransformer at {persist_directory}: {e}")
            print("   Please ensure you have run 'pip install chromadb sentence-transformers'")
            raise

    def add_function_node(self, node: FunctionNode, full_file_source: str) -> str:
        """
        Add a fully-formed FunctionNode to the vector store.
        Checks for `hash_body` to potentially skip re-embedding unchanged functions.

        Args:
            node: A FunctionNode object from the call graph.
            full_file_source: The full source code of the file containing the function.

        Returns:
            The unique ID of the stored document.
        """
        # Use a deterministic ID for idempotency based on FQN
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, node.fully_qualified_name))

        # Check if an existing document with this FQN has the same hash_body
        if node.hash_body:
            existing_doc = self.collection.get(ids=[doc_id], include=['metadatas'])
            if existing_doc and existing_doc['metadatas'] and existing_doc['metadatas'][0].get('hash_body') == node.hash_body:
                # print(f"   Skipping unchanged function: {node.fully_qualified_name}")
                return doc_id # No need to re-embed or upsert if body hash is the same

        # Create a descriptive document for this function
        document = self._create_function_document(node, full_file_source)

        # Generate embedding
        embedding = self.embedding_model.encode([document]).tolist()[0]

        # Create rich metadata from the FunctionNode
        metadata = {
            "type": "function",
            "fully_qualified_name": node.fully_qualified_name,
            "name": node.name,
            "file_path": node.file_path,
            "line_start": node.line_start,
            "line_end": node.line_end, # NEW: Include end line
            "is_entry_point": node.is_entry_point,
            "is_method": node.is_method,
            "class_name": node.class_name or "N/A",
            "parameter_count": len(node.parameters),
            "is_async": node.is_async,
            "incoming_degree": len(node.incoming_edges),
            "outgoing_degree": len(node.outgoing_edges),
            "has_docstring": bool(node.docstring),
            "access_modifier": node.access_modifier or "public",
            # --- NEW METADATA ATTRIBUTES ---
            "complexity": node.complexity,
            "nloc": node.nloc,
            "external_dependencies": json.dumps(node.external_dependencies), # Store list as JSON string for ChromaDB filtering
            "decorators": json.dumps(node.decorators), # Store list of dicts as JSON string
            "catches_exceptions": json.dumps(node.catches_exceptions), # Store list as JSON string
            "local_variables_declared": json.dumps(node.local_variables_declared), # Store list as JSON string
            "hash_body": node.hash_body, # Store hash for future comparisons
        }

        # Upsert ensures that if we run the analysis again, we update existing entries
        self.collection.upsert(
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id

    def _create_function_document(self, node: FunctionNode, full_file_source: str) -> str:
        """Create a descriptive document for vector search from a FunctionNode, including new attributes."""
        # Extract relevant source code snippet
        try:
            lines = full_file_source.split('\n')
            start_line_idx = max(0, node.line_start - 1)
            end_line_idx = node.line_end # Use the AST extractor's end_line for precision
            func_snippet = '\n'.join(lines[start_line_idx:end_line_idx]).strip()
        except Exception:
            func_snippet = "Source snippet unavailable."

        # Prepare new attributes for the document string
        decorators_str = ', '.join([d.get('name', 'unknown') for d in node.decorators]) if node.decorators else 'None'
        ext_deps_str = ', '.join(node.external_dependencies) if node.external_dependencies else 'None'
        catches_str = ', '.join(node.catches_exceptions) if node.catches_exceptions else 'None'
        local_vars_str = ', '.join(node.local_variables_declared) if node.local_variables_declared else 'None'

        # Create comprehensive document including NEW ATTRIBUTES
        document_parts = [
            f"FUNCTION: {node.fully_qualified_name}",
            f"Name: {node.name}",
            f"Parameters: {', '.join(node.parameters) if node.parameters else 'None'}",
            f"Returns: {node.return_type or 'unknown'}",
            f"Location: {node.file_path}:{node.line_start}-{node.line_end}",
            f"Type: {'Method' if node.is_method else 'Function'}{' (async)' if node.is_async else ''}{' (static)' if node.is_static else ''}",
            f"Class: {node.class_name if node.class_name else 'N/A'}",
            f"Entry Point: {node.is_entry_point}",
            f"Connections: {len(node.incoming_edges)} in, {len(node.outgoing_edges)} out",
            f"Complexity: {node.complexity or 'N/A'}", # NEW
            f"NLOC: {node.nloc or 'N/A'}", # NEW
            f"External Dependencies: {ext_deps_str}", # NEW
            f"Decorators: {decorators_str}", # NEW
            f"Catches Exceptions: {catches_str}", # NEW
            f"Local Variables: {local_vars_str}", # NEW
            f"Docstring: {node.docstring[:150] + '...' if node.docstring and len(node.docstring) > 150 else (node.docstring or 'None')}",
            f"Code Snippet: {func_snippet[:400]}..."
        ]

        return " | ".join([part for part in document_parts if part])

    def add_edge(self, edge: CallEdge) -> str:
        """Add a call edge to the vector store."""
        # Edges currently don't have new attributes, so this remains mostly unchanged
        document = f"CALL_EDGE: From {edge.caller} to {edge.callee} at line {edge.line_number}. Call type: {edge.call_type} with parameters {', '.join(edge.parameters)}."
        embedding = self.embedding_model.encode([document]).tolist()[0]

        metadata = {
            "type": "call_edge",
            "caller": edge.caller,
            "callee": edge.callee,
            "file_path": edge.file_path,
            "line_number": edge.line_number,
            "call_type": edge.call_type,
            "parameters": json.dumps(edge.parameters), # Store list as JSON string
            "confidence": edge.confidence
        }

        # Use a deterministic ID for idempotency for edges
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{edge.caller}->{edge.callee}@{edge.file_path}@{edge.line_number}"))

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
                          Note: For list/dict metadata stored as JSON strings,
                          you'll need to use `$contains` with a substring of the JSON.

        Returns:
            List of matching documents with metadata and distance.
        """
        effective_filter = {"type": "function"}
        if where_filter:
            effective_filter.update(where_filter)

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=effective_filter
        )

        formatted_results = []
        if not results or not results.get('documents') or not results['documents'][0]:
            return []

        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            # Deserialize JSON string attributes back into Python objects
            if 'external_dependencies' in meta: meta['external_dependencies'] = json.loads(meta['external_dependencies'])
            if 'decorators' in meta: meta['decorators'] = json.loads(meta['decorators'])
            if 'catches_exceptions' in meta: meta['catches_exceptions'] = json.loads(meta['catches_exceptions'])
            if 'local_variables_declared' in meta: meta['local_variables_declared'] = json.loads(meta['local_variables_declared'])
            if 'parameters' in meta: meta['parameters'] = json.loads(meta['parameters']) # for edges

            formatted_results.append({
                "document": results['documents'][0][i],
                "distance": results['distances'][0][i],
                "metadata": meta,
                "id": results['ids'][0][i]
            })

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        count = self.collection.count()
        return {
            "total_documents": count
        }
