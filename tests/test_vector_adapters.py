"""Exercise the real local vector SDKs through vector_adapters.py.

Provider parity includes metric values and scale-independent cosine ordering.
Each fixture owns its Chroma collection or temporary LanceDB directory.
"""
import pytest

from vector_adapters import ChromaIndex, LanceDBIndex


@pytest.fixture(params=["chroma", "lancedb"])
def index(request, tmp_path):
    if request.param == "chroma":
        index = ChromaIndex(name="vector-adapter-test")
        try:
            yield index
        finally:
            index.client.delete_collection("vector-adapter-test")
    else:
        yield LanceDBIndex(tmp_path / "lance")


def records():
    return [
        {"id": "a", "vector": [1.0, 0.0], "metadata": {"source": "doc-a", "page": "1"}},
        {"id": "b", "vector": [0.0, 1.0], "metadata": {"source": "doc-b", "page": "2"}},
        {"id": "c", "vector": [-1.0, 0.0], "metadata": {"source": "doc-a", "page": "3"}},
    ]


def test_indexes_preserve_cosine_order_metadata_filter_upsert_and_delete(index):
    index.upsert(records())
    assert index.count() == 3
    hits = index.search([1.0, 0.0], k=3)
    assert hits[0].record_id == "a"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert hits[0].metadata["source"] == "doc-a"
    filtered = index.search([1.0, 0.0], k=3, where={"source": "doc-b"})
    assert [hit.record_id for hit in filtered] == ["b"]
    index.upsert([{"id": "a", "vector": [0.0, 1.0],
                   "metadata": {"source": "doc-c", "page": "4"}}])
    assert index.count() == 3
    refreshed = index.search([0.0, 1.0], k=3, where={"source": "doc-c"})
    assert [hit.record_id for hit in refreshed] == ["a"]
    index.delete(["a"])
    assert index.count() == 2
    assert "a" not in {hit.record_id for hit in index.search([0.0, 1.0], k=3)}


def test_cosine_metric_ignores_magnitude_and_preserves_similarity_values(index):
    index.upsert([
        {"id": "scaled", "vector": [10.0, 0.0], "metadata": {"source": "doc"}},
        {"id": "nearby", "vector": [0.9, 0.1], "metadata": {"source": "doc"}},
        {"id": "orthogonal", "vector": [0.0, 1.0], "metadata": {"source": "doc"}},
        {"id": "opposite", "vector": [-1.0, 0.0], "metadata": {"source": "doc"}},
    ])
    hits = index.search([1.0, 0.0], k=4)
    assert [hit.record_id for hit in hits] == ["scaled", "nearby", "orthogonal", "opposite"]
    similarities = {hit.record_id: hit.similarity for hit in hits}
    assert similarities["scaled"] == pytest.approx(1.0, abs=1e-5)
    assert similarities["orthogonal"] == pytest.approx(0.0, abs=1e-5)
    assert similarities["opposite"] == pytest.approx(-1.0, abs=1e-5)


def test_lancedb_reopen_count_reads_existing_table(tmp_path):
    location = tmp_path / "persisted"
    original = LanceDBIndex(location)
    original.upsert(records())
    reopened = LanceDBIndex(location)
    assert reopened.count() == 3


def test_lancedb_metadata_cannot_replace_storage_fields(tmp_path):
    index = LanceDBIndex(tmp_path / "reserved")
    with pytest.raises(ValueError, match="reserved"):
        index.upsert([{ "id": "intended", "vector": [1.0, 0.0],
                       "metadata": {"id": "replacement"}}])
    assert index.count() == 0


def test_lancedb_filter_quotes_metadata_column_names(tmp_path):
    index = LanceDBIndex(tmp_path / "quoted")
    key = 'source-"name'
    index.upsert([{ "id": "a", "vector": [1.0, 0.0], "metadata": {key: "owner's doc"}}])
    assert [hit.record_id for hit in index.search([1.0, 0.0], k=1, where={key: "owner's doc"})] == ["a"]


def test_lancedb_rejects_unrepresentable_backtick_column_before_write(tmp_path):
    index = LanceDBIndex(tmp_path / "backtick")
    with pytest.raises(ValueError, match="backticks"):
        index.upsert([{ "id": "a", "vector": [1.0, 0.0],
                       "metadata": {"source`name": "doc"}}])
    assert index.count() == 0
