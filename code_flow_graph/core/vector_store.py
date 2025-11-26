"""
Vector store integration for code graph elements using ChromaDB.
This version is type-safe and expects enriched FunctionNode objects.
"""

from typing import List, Dict, Any, Set
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import json # For serializing complex metadata
import os
import asyncio
from pathlib import Path
import logging

# Import the specific, enriched data types from the call graph builder
from code_flow_graph.core.call_graph_builder import FunctionNode, CallEdge
from code_flow_graph.core.models import StructuredDataElement

class CodeVectorStore:
    """Vector store for code elements with explicit indexing strategy using ChromaDB."""

    def __init__(self, persist_directory: str, embedding_model_name: str = 'all-MiniLM-L6-v2', max_tokens: int = 256):
        """
        Initialize the ChromaDB vector store with consistent embedding dimensions.

        Args:
            persist_directory: Where to persist the vector database on disk.
            embedding_model_name: Model to use for embeddings (defaults to 384-dim for consistency)
        """
        logging.info(f"Initializing ChromaDB client at: {persist_directory}")
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)

            # Check if collection exists and get its current embedding dimension
            existing_collections = self.client.list_collections()
            collection_exists = any(col.name == "code_graph_v2" for col in existing_collections)

            if collection_exists:
                try:
                    existing_collection = self.client.get_collection("code_graph_v2")
                    # Check if collection has documents to determine embedding dimension
                    if existing_collection.count() > 0:
                        # Get a sample to check dimensions
                        sample = existing_collection.peek(1)
                        if sample['embeddings'] is not None and len(sample['embeddings']) > 0:
                            existing_dim = len(sample['embeddings'][0])
                            logging.info(f"   Info: Existing collection has {existing_dim}D embeddings")

                            # Use the same dimension as existing collection for consistency
                            if existing_dim == 768:
                                embedding_model_name = 'all-mpnet-base-v2'
                                logging.info(f"   Info: Using all-mpnet-base-v2 to match existing 768D collection")
                            elif existing_dim == 384:
                                embedding_model_name = 'all-MiniLM-L6-v2'
                                logging.info(f"   Info: Using all-MiniLM-L6-v2 to match existing 384D collection")
                            else:
                                logging.warning(f"   Warning: Unknown embedding dimension {existing_dim}, using default model")
                except Exception as e:
                    logging.warning(f"   Warning: Could not determine existing collection dimensions: {e}")

            self.collection = self.client.get_or_create_collection(
                name="code_graph_v2",
                metadata={"hnsw:space": "cosine"} # Using cosine distance for semantic similarity
            )

            # Use consistent embedding model (384 dimensions by default)
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.tokenizer = self.embedding_model.tokenizer
            self.max_tokens = max_tokens

            logging.info(f"✅ ChromaDB collection '{self.collection.name}' and Sentence Transformers loaded successfully.")
            logging.info(f"   Using embedding model: {embedding_model_name} ({self.embedding_model.get_sentence_embedding_dimension()} dimensions)")
        except Exception as e:
            logging.error(f"❌ Failed to initialize ChromaDB or SentenceTransformer at {persist_directory}: {e}")
            logging.error("   Please ensure you have run 'pip install chromadb sentence-transformers'")
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
                logging.warning(f"WARNING: Skipping empty chunk {i} for {node.fully_qualified_name}")
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
                "summary": node.summary,
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
        logging.info(f"Adding {len(nodes)} function nodes to vector store in batches of {batch_size}...")
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
                    logging.warning(f"WARNING: No source code available for {node.file_path}")

                # Create a descriptive document for this function
                document = self._create_function_document(node, full_file_source)

                # Debug: Check document content
                if not document or not document.strip():
                    logging.warning(f"WARNING: Empty document created for {node.fully_qualified_name}")
                    continue

                # Split into chunks
                chunks = self._split_document(document, max_tokens=256)

                # Debug: Check chunks
                if not chunks:
                    logging.warning(f"WARNING: No chunks created for {node.fully_qualified_name}")
                    continue
                # logging.info(f"Chunks for {node.fully_qualified_name}: {len(chunks)}")

                for j, chunk in enumerate(chunks):
                    # Skip empty chunks
                    if not chunk.strip():
                        logging.warning(f"WARNING: Skipping empty chunk {j} for {node.fully_qualified_name}")
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
                        "summary": node.summary,
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

        logging.info(f"   Batch added {len(all_doc_ids)} function chunks with average chunk size {sum(doc_sizes)//len(doc_sizes) if doc_sizes else 0} chars")

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
            f"Locals: {local_vars_str}",
            f"Doc: {node.docstring[:150] + '...' if node.docstring and len(node.docstring) > 150 else (node.docstring or 'None')}",
            f"Summary: {node.summary}" if node.summary else "",
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

    def add_structured_element(self, element: StructuredDataElement) -> str:
        """Add a structured data element to the vector store."""
        document = element.content
        embedding = self.embedding_model.encode([document]).tolist()[0]
        
        metadata = {
            "type": "structured_data",
            "fully_qualified_name": element.json_path,
            "name": element.key_name,
            "file_path": element.file_path,
            "line_start": element.line_start,
            "line_end": element.line_end,
            "json_path": element.json_path,
            "value_type": element.value_type,
            "key_name": element.key_name,
            "file_type": element.metadata.get('file_type', 'unknown'),
            "chunk_index": 0,
            "total_chunks": 1
        }
        
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{element.file_path}:{element.json_path}"))
        
        self.collection.upsert(
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id

    def add_structured_elements_batch(self, elements: List[StructuredDataElement], batch_size: int = 100) -> List[str]:
        """Add multiple structured data elements in batches."""
        doc_ids = []
        for i in range(0, len(elements), batch_size):
            batch_elements = elements[i:i + batch_size]
            batch_documents = []
            batch_embeddings = []
            batch_metadatas = []
            batch_ids = []
            
            for element in batch_elements:
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{element.file_path}:{element.json_path}"))
                doc_ids.append(doc_id)
                
                batch_documents.append(element.content)
                
                metadata = {
                    "type": "structured_data",
                    "fully_qualified_name": element.json_path,
                    "name": element.key_name,
                    "file_path": element.file_path,
                    "line_start": element.line_start,
                    "line_end": element.line_end,
                    "json_path": element.json_path,
                    "value_type": element.value_type,
                    "key_name": element.key_name,
                    "file_type": element.metadata.get('file_type', 'unknown'),
                    "chunk_index": 0,
                    "total_chunks": 1
                }
                batch_metadatas.append(metadata)
                batch_ids.append(doc_id)
            
            if batch_documents:
                embeddings = self.embedding_model.encode(batch_documents).tolist()
                batch_embeddings.extend(embeddings)
                
                self.collection.upsert(
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
        return doc_ids

    def query_codebase(self, query: str, n_results: int = 10, where_filter: Dict|None = None) -> List[Dict]:
        """
        Query the codebase (functions and structured data) using semantic search.
        
        Args:
            query: Search query.
            n_results: Number of complete documents to return.
            where_filter: Optional ChromaDB filter.

        Returns:
            List of documents with metadata and distance.
        """
        # Only include where filter if it's not None and not empty
        query_kwargs = {
            "query_texts": [query],
            "n_results": n_results * 4,  # Request more to account for grouping
            "include": ["metadatas", "documents", "distances"]
        }
        
        if where_filter:
            query_kwargs["where"] = where_filter
            
        raw_results = self.collection.query(**query_kwargs)

        return self._group_chunks_by_document(raw_results, n_results)

    def _group_chunks_by_document(self, results: Dict, max_results: int) -> List[Dict]:
        """
        Group chunks by fully_qualified_name and reconstruct complete documents.
        
        Args:
            results: Raw ChromaDB query results
            max_results: Maximum number of complete documents to return
        
        Returns:
            List of complete documents with all chunks consolidated
        """
        if not results or not results.get('documents') or not results['documents'][0]:
            return []

        # Group chunks by fully_qualified_name (document reference ID)
        document_groups = {}
        
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i]
            fqn = meta.get('fully_qualified_name')
            
            if not fqn:
                continue
                
            if fqn not in document_groups:
                document_groups[fqn] = {
                    'chunks': [],
                    'best_distance': float('inf'),
                    'metadata': meta.copy()
                }
            
            document_groups[fqn]['chunks'].append({
                'document': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'metadata': meta,
                'id': results['ids'][0][i]
            })
            
            # Track the best (lowest) distance for this document
            if results['distances'][0][i] < document_groups[fqn]['best_distance']:
                document_groups[fqn]['best_distance'] = results['distances'][0][i]

        # Sort documents by best distance and take top max_results
        sorted_documents = sorted(document_groups.values(), key=lambda x: x['best_distance'])
        
        formatted_results = []
        for doc_group in sorted_documents[:max_results]:
            # Sort chunks by chunk_index to reconstruct in proper order
            sorted_chunks = sorted(doc_group['chunks'], key=lambda x: x['metadata'].get('chunk_index', 0))
            
            # Reconstruct complete document from chunks
            complete_content = self._reconstruct_document_from_chunks(sorted_chunks)
            
            # Use metadata from chunk 0 (first chunk) as the primary metadata
            primary_meta = doc_group['metadata'].copy()
            primary_meta['chunk_count'] = len(sorted_chunks)
            primary_meta['document_reconstructed'] = True
            primary_meta['chunk_indices'] = [chunk['metadata'].get('chunk_index', 0) for chunk in sorted_chunks]
            
            formatted_results.append({
                'document': complete_content,
                'distance': doc_group['best_distance'],
                'metadata': primary_meta,
                'id': results['ids'][0][sorted_chunks[0]['metadata'].get('chunk_index', 0)],
                'chunk_info': {
                    'total_chunks': len(sorted_chunks),
                    'reconstructed': True
                }
            })
        
        return formatted_results

    def _reconstruct_document_from_chunks(self, sorted_chunks: List[Dict]) -> str:
        """
        Reconstruct a complete document from its sorted chunks.
        
        Args:
            sorted_chunks: List of chunk dictionaries sorted by chunk_index
        
        Returns:
            Reconstructed complete document
        """
        if not sorted_chunks:
            return ""
        
        # Join chunks with the same separator used in _split_document
        # This preserves the original document structure
        chunk_texts = [chunk['document'] for chunk in sorted_chunks]
        return " | ".join(chunk_texts)

    def get_all_file_paths(self) -> Set[str]:
        """
        Get all unique file paths referenced in the vector store.
        This is used for cleanup operations to identify potentially stale references.

        Returns:
            Set of unique file paths from all documents in the collection
        """
        try:
            # Get all documents with file_path metadata
            all_docs = self.collection.get(include=['metadatas'])

            file_paths = set()
            if all_docs and all_docs.get('metadatas'):
                for metadata in all_docs['metadatas']:
                    file_path = metadata.get('file_path')
                    if file_path:
                        file_paths.add(file_path)

            return file_paths
        except Exception as e:
            logging.warning(f"Error retrieving file paths from vector store: {e}")
            return set()

    def cleanup_stale_references(self, valid_file_paths: Set[str] = None) -> Dict[str, Any]:
        """
        Remove documents that reference files that no longer exist on the filesystem.

        Args:
            valid_file_paths: Optional set of known valid file paths to check against.
                            If None, will check all file paths in the store.

        Returns:
            Dict with cleanup statistics: {'removed_documents': int, 'errors': int}
        """
        try:
            removed_count = 0
            errors = 0

            # Get all file paths to check if not provided
            if valid_file_paths is None:
                file_paths_to_check = self.get_all_file_paths()
            else:
                file_paths_to_check = valid_file_paths

            # Find stale file paths (files that don't exist)
            stale_paths = set()
            for file_path in file_paths_to_check:
                if not os.path.exists(file_path):
                    stale_paths.add(file_path)

            if not stale_paths:
                logging.info("No stale file references found in vector store")
                return {'removed_documents': 0, 'errors': 0, 'stale_paths': 0}

            logging.info(f"Found {len(stale_paths)} stale file paths to clean up")

            # Remove documents for each stale file path in batches
            batch_size = 100
            for stale_path in stale_paths:
                try:
                    # Find all documents with this file path
                    docs_to_delete = self.collection.get(
                        where={"file_path": stale_path}
                    )

                    if docs_to_delete and docs_to_delete.get('ids'):
                        # Delete in batches to avoid memory issues
                        ids_to_delete = docs_to_delete['ids']
                        for i in range(0, len(ids_to_delete), batch_size):
                            batch_ids = ids_to_delete[i:i + batch_size]
                            self.collection.delete(ids=batch_ids)
                            removed_count += len(batch_ids)

                        logging.info(f"Removed {len(ids_to_delete)} documents for stale file: {stale_path}")

                except Exception as e:
                    errors += 1
                    logging.warning(f"Error removing documents for stale file {stale_path}: {e}")

            logging.info(f"Cleanup completed: removed {removed_count} documents, {errors} errors")
            return {
                'removed_documents': removed_count,
                'errors': errors,
                'stale_paths': len(stale_paths)
            }

        except Exception as e:
            logging.error(f"Error during stale reference cleanup: {e}")
            return {'removed_documents': 0, 'errors': 1, 'stale_paths': 0}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        count = self.collection.count()
        return {
            "total_documents": count
        }

    def update_function_node(self, node: FunctionNode, full_file_source: str) -> List[str]:
        """
        Update an existing function node in the vector store, typically to add a summary.
        This effectively overwrites the existing documents for this function.
        
        Args:
            node: The FunctionNode with updated data (e.g., summary).
            full_file_source: The full source code of the file.
            
        Returns:
            List of new document IDs.
        """
        # First, find existing documents to delete them (to avoid duplicates if chunking changes, though unlikely for just metadata)
        # Actually, upsert handles replacement by ID. But since IDs are deterministic based on FQN and chunk index,
        # and adding a summary might change the document content (if we include summary in content) or just metadata.
        # If we include summary in document content, chunking might change.
        # Safer to delete old chunks first if we can identify them, but upsert is usually fine if chunk count is same.
        # However, if summary is added to content, document length changes, so chunk count might change.
        
        # Strategy: Delete all chunks for this FQN first, then add fresh.
        try:
            self.collection.delete(
                where={"fully_qualified_name": node.fully_qualified_name}
            )
        except Exception as e:
            logging.warning(f"Error deleting existing chunks for {node.fully_qualified_name} during update: {e}")
            
        # Now add as new
        return self.add_function_node(node, full_file_source)

    def get_nodes_missing_summary(self) -> List[str]:
        """
        Retrieve FQNs of function nodes that are indexed but lack a summary.
        
        Returns:
            List of fully qualified names.
        """
        try:
            # Query for documents where type is function and summary is missing (or null/empty)
            # ChromaDB filtering limitations might apply. 
            # We can query for all functions and check metadata in python if needed, 
            # or use where={"$and": [{"type": "function"}, {"summary": {"$eq": None}}]} if supported.
            # ChromaDB might not support explicit NULL checks easily in all versions.
            # A safer approach for now: get all functions, check metadata. 
            # For large codebases this is heavy. 
            # Optimization: Can we filter by existence? 
            # Let's try to get all function metadatas and filter in python.
            
            results = self.collection.get(
                where={"type": "function"},
                include=["metadatas"]
            )
            
            missing_summary_fqns = set()
            if results and results['metadatas']:
                for meta in results['metadatas']:
                    if not meta.get('summary'):
                        fqn = meta.get('fully_qualified_name')
                        if fqn:
                            missing_summary_fqns.add(fqn)
                            
            return list(missing_summary_fqns)
            
        except Exception as e:
            logging.error(f"Error identifying nodes missing summary: {e}")
            return []
