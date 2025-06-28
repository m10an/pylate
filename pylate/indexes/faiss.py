from __future__ import annotations

import itertools
import os
from typing import List

import numpy as np
import torch
from sqlitedict import SqliteDict

try:
    import faiss
except Exception:  # pragma: no cover - we handle optional dependency
    faiss = None

from ..utils import iter_batch
from .base import Base


def reshape_embeddings(
    embeddings: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Reshape embeddings to expected shape (bs, n_tokens, dim)."""
    if isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 2:
            return np.expand_dims(a=embeddings, axis=0)

    if isinstance(embeddings, torch.Tensor):
        return reshape_embeddings(embeddings=embeddings.cpu().detach().numpy())

    if isinstance(embeddings, list) and isinstance(embeddings[0], torch.Tensor):
        return [embedding.cpu().detach().numpy() for embedding in embeddings]

    return embeddings


class Faiss(Base):
    """Faiss based index for multi-vector search using HNSW.

    Parameters
    ----------
    use_gpu
        Whether to move the index to GPU. Requires faiss compiled with GPU support.
    gpu_id
        The GPU id to use when ``use_gpu`` is ``True``.
    store_on_disk
        Memory-map the index from disk instead of keeping it fully in RAM.
        This option has no effect when ``use_gpu=True``.
    """

    def __init__(
        self,
        index_folder: str = "indexes",
        index_name: str = "colbert",
        override: bool = False,
        embedding_size: int = 128,
        M: int = 64,
        ef_construction: int = 200,
        ef_search: int = 200,
        use_gpu: bool = False,
        gpu_id: int = 0,
        store_on_disk: bool = False,
    ) -> None:
        if faiss is None:
            raise ImportError("faiss library is required to use the Faiss index")

        self.ef_search = ef_search
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.store_on_disk = store_on_disk
        self.gpu_resources = None
        if not os.path.exists(index_folder):
            os.makedirs(index_folder)
        if not os.path.exists(os.path.join(index_folder, index_name)):
            os.makedirs(os.path.join(index_folder, index_name))

        self.index_path = os.path.join(index_folder, index_name, "index.faiss")
        self.documents_ids_to_embeddings_path = os.path.join(
            index_folder, index_name, "document_ids_to_embeddings.sqlite"
        )
        self.embeddings_to_documents_ids_path = os.path.join(
            index_folder, index_name, "embeddings_to_documents_ids.sqlite"
        )

        self.index_cpu = self._create_collection(
            index_path=self.index_path,
            embedding_size=embedding_size,
            M=M,
            ef_construction=ef_construction,
            override=override,
        )
        if self.store_on_disk and not self.use_gpu:
            faiss.write_index(self.index_cpu, self.index_path)
            self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
            self.index_cpu = None
        else:
            self.index = self.index_cpu
            if self.use_gpu:
                self._move_to_gpu()

    def _load_documents_ids_to_embeddings(self) -> SqliteDict:
        return SqliteDict(self.documents_ids_to_embeddings_path, outer_stack=False)

    def _load_embeddings_to_documents_ids(self) -> SqliteDict:
        return SqliteDict(self.embeddings_to_documents_ids_path, outer_stack=False)

    def _create_collection(
        self,
        index_path: str,
        embedding_size: int,
        M: int,
        ef_construction: int,
        override: bool,
    ):
        if os.path.exists(index_path) and not override:
            if self.store_on_disk and not self.use_gpu:
                return faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
            return faiss.read_index(index_path)
        if os.path.exists(index_path):
            os.remove(index_path)

        index = faiss.IndexHNSWFlat(embedding_size, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = self.ef_search

        faiss.write_index(index, index_path)

        if override and os.path.exists(self.documents_ids_to_embeddings_path):
            os.remove(self.documents_ids_to_embeddings_path)
        if override and os.path.exists(self.embeddings_to_documents_ids_path):
            os.remove(self.embeddings_to_documents_ids_path)

        documents_ids_to_embeddings = self._load_documents_ids_to_embeddings()
        documents_ids_to_embeddings.close()

        embeddings_to_documents_ids = self._load_embeddings_to_documents_ids()
        embeddings_to_documents_ids.close()

        return index

    def _move_to_gpu(self) -> None:
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError("faiss was not compiled with GPU support")
        self.gpu_resources = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(
            self.gpu_resources, self.gpu_id, self.index_cpu
        )

    def add_documents(
        self,
        documents_ids: str | List[str],
        documents_embeddings: List[np.ndarray | torch.Tensor],
        batch_size: int = 2000,
    ) -> "Faiss":
        if isinstance(documents_ids, str):
            documents_ids = [documents_ids]

        if self.store_on_disk and self.index_cpu is None:
            self.index_cpu = faiss.read_index(self.index_path)

        documents_embeddings = reshape_embeddings(documents_embeddings)

        documents_ids_to_embeddings = self._load_documents_ids_to_embeddings()
        embeddings_to_documents_ids = self._load_embeddings_to_documents_ids()

        for doc_emb_batch, doc_ids_batch in zip(
            iter_batch(
                documents_embeddings,
                batch_size,
                desc=f"Adding documents to the index (bs={batch_size})",
            ),
            iter_batch(documents_ids, batch_size, tqdm_bar=False),
        ):
            flat_embeddings = np.vstack(doc_emb_batch).astype("float32")
            faiss.normalize_L2(flat_embeddings)
            start_id = self.index_cpu.ntotal
            self.index_cpu.add(flat_embeddings)
            embedding_ids = np.arange(start_id, start_id + len(flat_embeddings))

            total = 0
            for doc_id, embeddings in zip(doc_ids_batch, doc_emb_batch):
                emb_ids = embedding_ids[total : total + len(embeddings)]
                documents_ids_to_embeddings[doc_id] = emb_ids.tolist()
                embeddings_to_documents_ids.update(
                    dict.fromkeys(emb_ids.tolist(), doc_id)
                )
                total += len(embeddings)

        documents_ids_to_embeddings.commit()
        embeddings_to_documents_ids.commit()
        documents_ids_to_embeddings.close()
        embeddings_to_documents_ids.close()
        if self.use_gpu:
            faiss.write_index(self.index_cpu, self.index_path)
            self.index = faiss.index_cpu_to_gpu(
                self.gpu_resources, self.gpu_id, self.index_cpu
            )
        elif self.store_on_disk:
            faiss.write_index(self.index_cpu, self.index_path)
            self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
            self.index_cpu = None
        else:
            self.index = self.index_cpu
            faiss.write_index(self.index_cpu, self.index_path)
        return self

    def remove_documents(self, documents_ids: List[str]) -> "Faiss":
        if self.store_on_disk and self.index_cpu is None:
            self.index_cpu = faiss.read_index(self.index_path)

        documents_ids_to_embeddings = self._load_documents_ids_to_embeddings()
        embeddings_to_documents_ids = self._load_embeddings_to_documents_ids()
        for doc_id in documents_ids:
            emb_ids = documents_ids_to_embeddings[doc_id]
            for emb_id in emb_ids:
                del embeddings_to_documents_ids[str(emb_id)]
                self.index_cpu.remove_ids(np.array([emb_id], dtype="int64"))
            del documents_ids_to_embeddings[doc_id]
        documents_ids_to_embeddings.commit()
        embeddings_to_documents_ids.commit()
        documents_ids_to_embeddings.close()
        embeddings_to_documents_ids.close()
        if self.use_gpu:
            faiss.write_index(self.index_cpu, self.index_path)
            self.index = faiss.index_cpu_to_gpu(
                self.gpu_resources, self.gpu_id, self.index_cpu
            )
        elif self.store_on_disk:
            faiss.write_index(self.index_cpu, self.index_path)
            self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
            self.index_cpu = None
        else:
            self.index = self.index_cpu
            faiss.write_index(self.index_cpu, self.index_path)
        return self

    def __call__(self, queries_embeddings: np.ndarray | torch.Tensor, k: int = 10):
        embeddings_to_documents_ids = self._load_embeddings_to_documents_ids()
        k = min(k, len(embeddings_to_documents_ids))

        queries_embeddings = reshape_embeddings(queries_embeddings)
        n_queries = len(queries_embeddings)
        flat_queries = np.vstack(queries_embeddings).astype("float32")
        faiss.normalize_L2(flat_queries)

        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = self.ef_search
        else:
            try:
                faiss.ParameterSpace().set_index_parameter(
                    self.index, "efSearch", self.ef_search
                )
            except Exception:
                pass
        distances, indices = self.index.search(flat_queries, k)

        documents = [
            [
                [embeddings_to_documents_ids[str(eid)] for eid in token_ids]
                for token_ids in doc_indices
            ]
            for doc_indices in indices.reshape(n_queries, -1, k)
        ]

        embeddings_to_documents_ids.close()

        return {
            "documents_ids": documents,
            "distances": distances.reshape(n_queries, -1, k),
        }

    def get_documents_embeddings(
        self, document_ids: List[List[str]]
    ) -> List[List[List[float]]]:
        if self.store_on_disk and self.index_cpu is None:
            self.index_cpu = faiss.read_index(self.index_path)

        documents_ids_to_embeddings = self._load_documents_ids_to_embeddings()
        embedding_ids_structure = [
            [documents_ids_to_embeddings[doc_id] for doc_id in doc_group]
            for doc_group in document_ids
        ]
        documents_ids_to_embeddings.close()

        flat_ids = list(
            itertools.chain.from_iterable(
                itertools.chain.from_iterable(embedding_ids_structure)
            )
        )
        embeddings = [self.index_cpu.reconstruct(int(eid)) for eid in flat_ids]

        reconstructed = []
        pos = 0
        for group_ids in embedding_ids_structure:
            group_embeddings = []
            for doc_emb_ids in group_ids:
                num = len(doc_emb_ids)
                group_embeddings.append(embeddings[pos : pos + num])
                pos += num
            reconstructed.append(group_embeddings)
        return reconstructed
