# Faiss

Faiss based index for approximate nearest neighbour search using an IVF-PQ index with an OPQ rotation. The interface mirrors the one of the Voyager index so that it can be used interchangeably for candidate generation and reranking. When ``use_gpu=True`` the index is moved to GPU (if available) for faster search. Set ``store_on_disk=True`` to memory-map the index instead of keeping it fully in RAM (ignored when ``use_gpu=True``).

```python
>>> from pylate import indexes, models

>>> index = indexes.Faiss(
...     index_folder="test_indexes",
...     index_name="colbert",
...     override=True,
...     embedding_size=128,
...     use_gpu=True,
...     store_on_disk=False,
... )

>>> model = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
... )

>>> docs_embs = model.encode([
...     "fruits are healthy.",
...     "fruits are good for health.",
... ], is_query=False)

>>> index.add_documents(documents_ids=["1", "2"], documents_embeddings=docs_embs)

>>> query_embs = model.encode("fruits are healthy.", is_query=True)
>>> index(query_embs, k=5)
{'documents_ids': [[['1'], ['2']]], 'distances': array([[[1.]], [[0.5]]])}
```
