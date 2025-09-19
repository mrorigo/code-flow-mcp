"""
Vector store integration for code graph elements using ChromaDB.
This version is type-safe and expects enriched FunctionNode objects.
"""

from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import json # For serializing complex metadata

# Import the specific, enriched data types from the call graph builder
from code_flow_graph.core.call_graph_builder import FunctionNode, CallEdge

class CodeVectorStore:
    """Vector store for code elements with explicit indexing strategy using ChromaDB."""

    def __init__(self, persist_directory: str, embedding_model_name: str = 'all-mpnet-base-v2', max_tokens: int = 256):
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
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.tokenizer = self.embedding_model.tokenizer
            self.max_tokens = max_tokens

            print(f"✅ ChromaDB collection '{self.collection.name}' and Sentence Transformers loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize ChromaDB or SentenceTransformer at {persist_directory}: {e}")
            print("   Please ensure you have run 'pip install chromadb sentence-transformers'")
            raise

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text using the model's tokenizer."""
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)

    def _split_document(self, document: str, max_tokens: int = 256) -> List[str]:
        """Split document into chunks that fit within max_tokens."""
        if self._count_tokens(document) <= max_tokens:
            return [document]

        chunks = []
        sentences = document.split(' | ')  # Split on your document separators for natural breaks
        current_chunk = ""
        
        for sentence in sentences:
            potential_chunk = current_chunk + " | " + sentence if current_chunk else sentence
            if self._count_tokens(potential_chunk) <= max_tokens:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def add_function_node(self, node: FunctionNode, full_file_source: str) -> List[str]:
        """
        Add a fully-formed FunctionNode to the vector store.
        Checks for `hash_body` to potentially skip re-embedding unchanged functions.

        Args:
            node: A FunctionNode object from the call graph.
            full_file_source: The full source code of the file containing the function.

        Returns:
            List of unique IDs of the stored documents (one per chunk).
        """
        # Check if an existing document with this FQN has the same hash_body
        if node.hash_body:
            # Query for any documents with same FQN and hash_body
            existing_docs = self.collection.get(
                where={"$and": [
                    {"fully_qualified_name": node.fully_qualified_name},
                    {"hash_body": node.hash_body}
                ]},
                include=['metadatas']
            )
            if existing_docs and existing_docs['metadatas']:
                # Skip if any chunk exists with same hash_body
                return [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{node.fully_qualified_name}_chunk_{i}")) for i in range(len(existing_docs['metadatas']))]

        # Create a descriptive document for this function
        document = self._create_function_document(node, full_file_source)

        # Split into chunks
        chunks = self._split_document(document, max_tokens=self.max_tokens)

        batch_documents = []
        batch_embeddings = []
        batch_metadatas = []
        batch_ids = []

        for i, chunk in enumerate(chunks):
            # Skip empty chunks
            if not chunk.strip():
                print(f"WARNING: Skipping empty chunk {i} for {node.fully_qualified_name}")
                continue

            # Generate embedding for each chunk
            embedding = self.embedding_model.encode([chunk]).tolist()[0]

            # Create metadata (add chunk index)
            metadata = {
                "type": "function",
                "fully_qualified_name": node.fully_qualified_name,
                "name": node.name,
                "file_path": node.file_path,
                "line_start": node.line_start,
                "line_end": node.line_end,
                "is_entry_point": node.is_entry_point,
                "is_method": node.is_method,
                "class_name": node.class_name or "N/A",
                "parameter_count": len(node.parameters),
                "is_async": node.is_async,
                "incoming_degree": len(node.incoming_edges),
                "outgoing_degree": len(node.outgoing_edges),
                "has_docstring": bool(node.docstring),
                "access_modifier": node.access_modifier or "public",
                "complexity": node.complexity,
                "nloc": node.nloc,
                "external_dependencies": json.dumps(node.external_dependencies),
                "decorators": json.dumps(node.decorators),
                "catches_exceptions": json.dumps(node.catches_exceptions),
                "local_variables_declared": json.dumps(node.local_variables_declared),
                "hash_body": node.hash_body,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{node.fully_qualified_name}_chunk_{i}"))
            batch_documents.append(chunk)
            batch_embeddings.append(embedding)
            batch_metadatas.append(metadata)
            batch_ids.append(doc_id)

        # Upsert all chunks at once
        if batch_documents:
            self.collection.upsert(
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

        return batch_ids

    def add_function_nodes_batch(self, nodes: List[FunctionNode], sources: Dict[str, str], batch_size: int = 100) -> List[str]:
        """
        Add multiple FunctionNodes to the vector store in batches for improved performance.

        Args:
            nodes: List of FunctionNode objects from the call graph.
            sources: Dict mapping file_path to full source code content.
            batch_size: Number of nodes to process in each batch.

        Returns:
            List of unique IDs of the stored documents (one per chunk).
        """
        print(f"Adding {len(nodes)} function nodes to vector store in batches of {batch_size}...")
        all_doc_ids = []
        doc_sizes = []
        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i:i + batch_size]
            batch_documents = []
            batch_embeddings = []
            batch_metadatas = []
            batch_ids = []
            seen_fqns = set()

            for node in batch_nodes:
                # Skip if we've already seen this FQN in the batch
                if node.fully_qualified_name in seen_fqns:
                    continue
                seen_fqns.add(node.fully_qualified_name)

                # Check if an existing document with this FQN has the same hash_body
                skip_node = False
                if node.hash_body:
                    existing_docs = self.collection.get(
                        where={"$and": [
                            {"fully_qualified_name": node.fully_qualified_name},
                            {"hash_body": node.hash_body}
                        ]},
                        include=['metadatas']
                    )
                    if existing_docs and existing_docs['metadatas']:
                        skip_node = True

                if skip_node:
                    # Add existing chunk IDs to the list
                    existing_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{node.fully_qualified_name}_chunk_{j}")) for j in range(len(existing_docs['metadatas']))]
                    all_doc_ids.extend(existing_ids)
                    continue

                # Get source code for this node
                full_file_source = sources.get(node.file_path, "")

                # Debug: Check source availability
                if not full_file_source or not full_file_source.strip():
                    print(f"WARNING: No source code available for {node.file_path}")

                # Create a descriptive document for this function
                document = self._create_function_document(node, full_file_source)

                # Debug: Check document content
                if not document or not document.strip():
                    print(f"WARNING: Empty document created for {node.fully_qualified_name}")
                    continue

                # Split into chunks
                chunks = self._split_document(document, max_tokens=256)

                # Debug: Check chunks
                if not chunks:
                    print(f"WARNING: No chunks created for {node.fully_qualified_name}")
                    continue
                # print(f"Chunks for {node.fully_qualified_name}: {len(chunks)}")

                for j, chunk in enumerate(chunks):
                    # Skip empty chunks
                    if not chunk.strip():
                        print(f"WARNING: Skipping empty chunk {j} for {node.fully_qualified_name}")
                        continue

                    # Track actual chunk size
                    doc_sizes.append(len(chunk))

                    # Create metadata for each chunk
                    metadata = {
                        "type": "function",
                        "fully_qualified_name": node.fully_qualified_name,
                        "name": node.name,
                        "file_path": node.file_path,
                        "line_start": node.line_start,
                        "line_end": node.line_end,
                        "is_entry_point": node.is_entry_point,
                        "is_method": node.is_method,
                        "class_name": node.class_name or "N/A",
                        "parameter_count": len(node.parameters),
                        "is_async": node.is_async,
                        "incoming_degree": len(node.incoming_edges),
                        "outgoing_degree": len(node.outgoing_edges),
                        "has_docstring": bool(node.docstring),
                        "access_modifier": node.access_modifier or "public",
                        "complexity": node.complexity,
                        "nloc": node.nloc,
                        "external_dependencies": json.dumps(node.external_dependencies),
                        "decorators": json.dumps(node.decorators),
                        "catches_exceptions": json.dumps(node.catches_exceptions),
                        "local_variables_declared": json.dumps(node.local_variables_declared),
                        "hash_body": node.hash_body,
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                    }
                    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{node.fully_qualified_name}_chunk_{j}"))
                    batch_documents.append(chunk)
                    batch_metadatas.append(metadata)
                    batch_ids.append(doc_id)
                    all_doc_ids.append(doc_id)

            # Generate embeddings for the batch
            if batch_documents:
                embeddings = self.embedding_model.encode(batch_documents).tolist()
                batch_embeddings.extend(embeddings)

                # Upsert the batch
                self.collection.upsert(
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )

        print(f"   Batch added {len(all_doc_ids)} function chunks with average chunk size {sum(doc_sizes)//len(doc_sizes) if doc_sizes else 0} chars")

        return all_doc_ids

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
            f"Complexity: {node.complexity or 'N/A'}",
            f"NLOC: {node.nloc or 'N/A'}",
            f"External Deps: {ext_deps_str}",
            f"Decorators: {decorators_str}",
            f"Catches: {catches_str}",
            f"Locals: {local_vars_str}",
            f"Doc: {node.docstring[:150] + '...' if node.docstring and len(node.docstring) > 150 else (node.docstring or 'None')}",
            f"Code: {func_snippet[:400]}..."
        ]

        # Filter out empty/whitespace-only parts
        filtered_parts = [part for part in document_parts if part and part.strip() and part.strip() != "None"]
        result = " | ".join(filtered_parts)
        
        return result

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

    def add_edges_batch(self, edges: List[CallEdge], batch_size: int = 100) -> List[str]:
        """
        Add multiple CallEdges to the vector store in batches for improved performance.

        Args:
            edges: List of CallEdge objects from the call graph.
            batch_size: Number of edges to process in each batch.

        Returns:
            List of unique IDs of the stored documents.
        """
        doc_ids = []
        for i in range(0, len(edges), batch_size):
            batch_edges = edges[i:i + batch_size]
            batch_documents = []
            batch_embeddings = []
            batch_metadatas = []
            batch_ids = []
            seen_ids = set()

            for edge in batch_edges:
                # Use a deterministic ID for idempotency for edges
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{edge.caller}->{edge.callee}@{edge.file_path}@{edge.line_number}"))

                # Skip if we've already seen this ID in the batch (deduplicate)
                if doc_id in seen_ids:
                    continue

                seen_ids.add(doc_id)
                doc_ids.append(doc_id)

                document = f"CALL_EDGE: From {edge.caller} to {edge.callee} at line {edge.line_number}. Call type: {edge.call_type} with parameters {', '.join(edge.parameters)}."
                batch_documents.append(document)

                metadata = {
                    "type": "call_edge",
                    "caller": edge.caller,
                    "callee": edge.callee,
                    "file_path": edge.file_path,
                    "line_number": edge.line_number,
                    "call_type": edge.call_type,
                    "parameters": json.dumps(edge.parameters),
                    "confidence": edge.confidence
                }
                batch_metadatas.append(metadata)
                batch_ids.append(doc_id)

            # Generate embeddings for the batch
            if batch_documents:
                embeddings = self.embedding_model.encode(batch_documents).tolist()
                batch_embeddings.extend(embeddings)

                # Upsert the batch
                self.collection.upsert(
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )

        return doc_ids

    def query_functions(self, query: str, n_results: int = 10, where_filter: Dict|None = None) -> List[Dict]:
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
