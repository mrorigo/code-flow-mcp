"""
Cortex memory store for long-lived (tribal) and short-lived (episodic) knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import json
import logging
import math
import uuid

import chromadb
from sentence_transformers import SentenceTransformer


@dataclass
class CortexMemoryRecord:
    knowledge_id: str
    memory_type: str
    content: str
    created_at: int
    last_reinforced: int
    reinforcement_count: int
    base_confidence: float
    tags: List[str]
    scope: str
    file_paths: List[str]
    source: str
    decay_half_life_days: float
    decay_floor: float
    metadata: Dict[str, Any]


class CortexMemoryStore:
    """Vector-backed memory store with decay-aware ranking."""

    def __init__(
        self,
        persist_directory: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "cortex_memory_v1",
    ) -> None:
        logging.info(f"Initializing Cortex memory store at: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_model = SentenceTransformer(embedding_model_name)

    @staticmethod
    def _now_epoch() -> int:
        return int(datetime.now(timezone.utc).timestamp())

    @staticmethod
    def _serialize_list(value: Optional[List[str]]) -> str:
        return json.dumps(value or [])

    @staticmethod
    def _deserialize_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                return []
        return []

    @staticmethod
    def _compute_decay(
        last_reinforced: int,
        half_life_days: float,
        decay_floor: float,
        now_epoch: Optional[int] = None,
    ) -> float:
        now_epoch = now_epoch or CortexMemoryStore._now_epoch()
        age_seconds = max(0, now_epoch - last_reinforced)
        half_life_seconds = max(1.0, half_life_days * 86400.0)
        decay = 0.5 ** (age_seconds / half_life_seconds)
        return max(decay_floor, decay)

    @staticmethod
    def _compute_memory_score(
        base_confidence: float,
        reinforcement_count: int,
        decay: float,
    ) -> float:
        return base_confidence * decay * (1.0 + math.log1p(max(0, reinforcement_count)))

    def add_memory(
        self,
        content: str,
        memory_type: str,
        tags: Optional[List[str]] = None,
        scope: str = "repo",
        file_paths: Optional[List[str]] = None,
        source: str = "user",
        base_confidence: float = 1.0,
        decay_half_life_days: float = 30.0,
        decay_floor: float = 0.05,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CortexMemoryRecord:
        knowledge_id = str(uuid.uuid4())
        now_epoch = self._now_epoch()
        record = CortexMemoryRecord(
            knowledge_id=knowledge_id,
            memory_type=memory_type,
            content=content,
            created_at=now_epoch,
            last_reinforced=now_epoch,
            reinforcement_count=0,
            base_confidence=base_confidence,
            tags=tags or [],
            scope=scope,
            file_paths=file_paths or [],
            source=source,
            decay_half_life_days=decay_half_life_days,
            decay_floor=decay_floor,
            metadata=metadata or {},
        )

        embedding = self.embedding_model.encode([content]).tolist()[0]
        meta = self.record_to_metadata(record)
        self.collection.upsert(
            documents=[content],
            embeddings=[embedding],
            metadatas=[meta],
            ids=[knowledge_id],
        )
        return record

    def reinforce_memory(
        self,
        knowledge_id: str,
        increment: int = 1,
        boost_confidence: Optional[float] = None,
    ) -> Optional[CortexMemoryRecord]:
        result = self.collection.get(ids=[knowledge_id], include=["documents", "metadatas"])
        if not result or not result.get("ids"):
            return None
        metadata = result["metadatas"][0]
        content = result["documents"][0]
        record = self._metadata_to_record(metadata, content)
        record.reinforcement_count += max(0, increment)
        record.last_reinforced = self._now_epoch()
        if boost_confidence is not None:
            record.base_confidence = boost_confidence

        embedding = self.embedding_model.encode([record.content]).tolist()[0]
        self.collection.upsert(
            documents=[record.content],
            embeddings=[embedding],
            metadatas=[self.record_to_metadata(record)],
            ids=[record.knowledge_id],
        )
        return record

    def query_memory(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        similarity_weight: float = 0.7,
        memory_score_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        where_filter = self._build_where_filter(filters or {})
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, n_results * 4),
            include=["documents", "metadatas", "distances"],
            where=where_filter or None,
        )

        if not results or not results.get("documents"):
            return []

        formatted = []
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            content = results["documents"][0][i]
            distance = results["distances"][0][i]
            record = self._metadata_to_record(metadata, content)

            decay = self._compute_decay(
                record.last_reinforced,
                record.decay_half_life_days,
                record.decay_floor,
            )
            memory_score = self._compute_memory_score(
                record.base_confidence,
                record.reinforcement_count,
                decay,
            )
            similarity = 1.0 - distance
            final_score = similarity_weight * similarity + memory_score_weight * memory_score

            if filters and filters.get("tags"):
                tags_filter = set(filters.get("tags", []))
                if not tags_filter.issubset(set(record.tags)):
                    continue

            formatted.append(
                {
                    "id": metadata.get("knowledge_id", ""),
                    "document": content,
                    "distance": distance,
                    "similarity": similarity,
                    "memory_score": memory_score,
                    "final_score": final_score,
                    "metadata": self.record_to_metadata(record),
                }
            )

        formatted.sort(key=lambda item: item["final_score"], reverse=True)
        return formatted[:n_results]

    def list_memory(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        where_filter = self._build_where_filter(filters or {})
        results = self.collection.get(include=["documents", "metadatas"], where=where_filter or None)
        if not results or not results.get("ids"):
            return []

        items = []
        for idx, knowledge_id in enumerate(results["ids"]):
            metadata = results["metadatas"][idx]
            content = results["documents"][idx]
            record = self._metadata_to_record(metadata, content)
            items.append(
                {
                    "id": knowledge_id,
                    "document": content,
                    "metadata": self.record_to_metadata(record),
                }
            )

        sliced = items[offset: offset + limit]
        return sliced

    def forget_memory(self, knowledge_id: str) -> bool:
        try:
            self.collection.delete(ids=[knowledge_id])
            return True
        except Exception as exc:
            logging.warning(f"Failed to delete memory {knowledge_id}: {exc}")
            return False

    def cleanup_stale_memory(self, min_score: float, grace_seconds: int) -> Dict[str, Any]:
        results = self.collection.get(include=["documents", "metadatas"])
        if not results or not results.get("ids"):
            return {"removed": 0, "checked": 0}

        now_epoch = self._now_epoch()
        to_delete = []
        for idx, knowledge_id in enumerate(results["ids"]):
            metadata = results["metadatas"][idx]
            content = results["documents"][idx]
            record = self._metadata_to_record(metadata, content)
            if now_epoch - record.last_reinforced < grace_seconds:
                continue

            decay = self._compute_decay(
                record.last_reinforced,
                record.decay_half_life_days,
                record.decay_floor,
                now_epoch=now_epoch,
            )
            memory_score = self._compute_memory_score(
                record.base_confidence,
                record.reinforcement_count,
                decay,
            )
            if memory_score < min_score:
                to_delete.append(knowledge_id)

        if to_delete:
            self.collection.delete(ids=to_delete)

        return {"removed": len(to_delete), "checked": len(results["ids"])}

    def record_to_metadata(self, record: CortexMemoryRecord) -> Dict[str, Any]:
        return {
            "knowledge_id": record.knowledge_id,
            "memory_type": record.memory_type,
            "content": record.content,
            "created_at": record.created_at,
            "last_reinforced": record.last_reinforced,
            "reinforcement_count": record.reinforcement_count,
            "base_confidence": record.base_confidence,
            "tags": self._serialize_list(record.tags),
            "scope": record.scope,
            "file_paths": self._serialize_list(record.file_paths),
            "source": record.source,
            "decay_half_life_days": record.decay_half_life_days,
            "decay_floor": record.decay_floor,
            "metadata": json.dumps(record.metadata),
            "type": "cortex_memory",
        }

    def _metadata_to_record(self, metadata: Dict[str, Any], content: str) -> CortexMemoryRecord:
        return CortexMemoryRecord(
            knowledge_id=metadata.get("knowledge_id") or metadata.get("id") or "",
            memory_type=metadata.get("memory_type", "FACT"),
            content=content,
            created_at=int(metadata.get("created_at", 0)),
            last_reinforced=int(metadata.get("last_reinforced", 0)),
            reinforcement_count=int(metadata.get("reinforcement_count", 0)),
            base_confidence=float(metadata.get("base_confidence", 1.0)),
            tags=self._deserialize_list(metadata.get("tags")),
            scope=metadata.get("scope", "repo"),
            file_paths=self._deserialize_list(metadata.get("file_paths")),
            source=metadata.get("source", "user"),
            decay_half_life_days=float(metadata.get("decay_half_life_days", 30.0)),
            decay_floor=float(metadata.get("decay_floor", 0.05)),
            metadata=self._deserialize_metadata(metadata.get("metadata")),
        )

    @staticmethod
    def _deserialize_metadata(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _build_where_filter(filters: Dict[str, Any]) -> Dict[str, Any]:
        where_filter: Dict[str, Any] = {}
        for key in ("memory_type", "scope", "source"):
            if filters.get(key):
                where_filter[key] = filters[key]
        return where_filter
