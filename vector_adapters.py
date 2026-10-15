"""Small local vector-index seam with real Chroma and LanceDB adapters.

Used by tests/test_vector_adapters.py with caller-supplied vectors.
Both providers use cosine; these checks do not measure semantic retrieval quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class VectorHit:
    record_id: str
    similarity: float
    metadata: dict


class VectorIndex(Protocol):
    def upsert(self, records: list[dict]) -> None: ...
    def search(self, vector: list[float], *, k: int,
               where: dict[str, str] | None = None) -> list[VectorHit]: ...
    def delete(self, record_ids: list[str]) -> None: ...
    def count(self) -> int: ...


class ChromaIndex:
    """Chroma ephemeral collection using cosine distance."""
    def __init__(self, name: str = "noisy-rag-local"):
        import chromadb
        from chromadb.config import Settings
        self.client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.create_collection(
            name=name, metadata={"hnsw:space": "cosine"})

    def upsert(self, records: list[dict]) -> None:
        if not records:
            return
        self.collection.upsert(
            ids=[str(r["id"]) for r in records],
            embeddings=[r["vector"] for r in records],
            metadatas=[r["metadata"] for r in records],
        )

    def search(self, vector: list[float], *, k: int,
               where: dict[str, str] | None = None) -> list[VectorHit]:
        result = self.collection.query(query_embeddings=[vector], n_results=k,
                                       where=where, include=["metadatas", "distances"])
        return [VectorHit(record_id, 1.0 - float(distance), metadata or {})
                for record_id, distance, metadata in zip(
                    result["ids"][0], result["distances"][0], result["metadatas"][0])]

    def delete(self, record_ids: list[str]) -> None:
        if record_ids:
            self.collection.delete(ids=record_ids)

    def count(self) -> int:
        return self.collection.count()


class LanceDBIndex:
    """LanceDB local table using its native cosine vector search.

    Metadata uses flat columns; storage names and backticks in keys are unsupported.
    """
    def __init__(self, uri: str | Path, name: str = "noisy_rag_local"):
        import lancedb
        self.db = lancedb.connect(str(uri))
        self.name = name
        self.table = None

    def upsert(self, records: list[dict]) -> None:
        if not records:
            return
        if any({str(key) for key in record["metadata"]} & {"id", "vector", "_distance"}
               for record in records):
            raise ValueError("LanceDB metadata contains a reserved storage field")
        if any("`" in str(key) for record in records for key in record["metadata"]):
            raise ValueError("LanceDB metadata field names cannot contain backticks")
        rows = [{"id": str(r["id"]), "vector": r["vector"],
                 **{str(k): str(v) for k, v in r["metadata"].items()}}
                for r in records]
        if self.name not in self.db.list_tables().tables:
            self.table = self.db.create_table(self.name, data=rows)
        else:
            self.table = self.db.open_table(self.name)
            self.table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)

    def search(self, vector: list[float], *, k: int,
               where: dict[str, str] | None = None) -> list[VectorHit]:
        if self.table is None:
            self.table = self.db.open_table(self.name)
        query = self.table.search(vector, vector_column_name="vector").distance_type("cosine")
        if where:
            clauses = []
            for key, value in sorted(where.items()):
                column = str(key).replace("`", "``")
                literal = str(value).replace("'", "''")
                # Lance SQL uses backticks for identifiers; double quotes are strings.
                clauses.append(f"`{column}` = '{literal}'")
            query = query.where(" AND ".join(clauses), prefilter=True)
        rows = query.limit(k).to_list()
        hits = []
        for row in rows:
            metadata = {key: value for key, value in row.items()
                        if key not in {"id", "vector", "_distance"}}
            hits.append(VectorHit(str(row["id"]), 1.0 - float(row["_distance"]), metadata))
        return hits

    def delete(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        if self.table is None:
            self.table = self.db.open_table(self.name)
        literals = ", ".join("'" + value.replace("'", "''") + "'" for value in record_ids)
        self.table.delete(f"id IN ({literals})")

    def count(self) -> int:
        if self.table is None:
            if self.name not in self.db.list_tables().tables:
                return 0
            self.table = self.db.open_table(self.name)
        return self.table.count_rows()
