import shutil
import uuid

import pytest

from pylate import indexes, models, retrieve

faiss = pytest.importorskip("faiss")


def test_faiss_index():
    random_hash = uuid.uuid4().hex
    index = indexes.Faiss(
        index_folder=f"test_indexes_{random_hash}",
        index_name=f"colbert_{random_hash}",
        override=True,
        embedding_size=128,
        nlist=1,
        store_on_disk=True,
    )

    model = models.ColBERT(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    documents_embeddings = model.encode(
        ["fruits are healthy.", "fruits are good for health."],
        is_query=False,
    )

    index.add_documents(
        documents_ids=["1", "2"], documents_embeddings=documents_embeddings
    )

    queries_embeddings = model.encode(["fruits are healthy."], is_query=True)

    results = index(queries_embeddings, k=2)
    assert isinstance(results, dict)
    assert len(results["documents_ids"]) == 1
    assert results["distances"].shape[0] == 1

    retriever = retrieve.ColBERT(index=index)
    retrieved = retriever.retrieve(
        queries_embeddings=queries_embeddings,
        k=2,
    )
    assert isinstance(retrieved, list)
    assert len(retrieved) == 1

    # Test loading existing index
    index = indexes.Faiss(
        index_folder=f"test_indexes_{random_hash}",
        index_name=f"colbert_{random_hash}",
        override=False,
        embedding_size=128,
        nlist=1,
        store_on_disk=True,
    )
    results = index(queries_embeddings, k=2)
    assert isinstance(results, dict)
    assert len(results["documents_ids"]) == 1

    # Test remove and re-add
    index.remove_documents(["1"])
    results = index(queries_embeddings, k=2)
    assert len(results["documents_ids"][0]) == len(results["distances"][0])

    index.add_documents(
        documents_ids=["1"], documents_embeddings=documents_embeddings[0]
    )
    results = index(queries_embeddings, k=2)
    assert len(results["documents_ids"][0]) == 2

    shutil.rmtree(f"test_indexes_{random_hash}")
