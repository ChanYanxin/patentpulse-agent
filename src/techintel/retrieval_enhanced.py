"""OPTIONAL real dense+sparse hybrid retrieval with reciprocal rank fusion (RRF).

Local pretrained models or paid embeddings MUST be initialized by the caller.
Neither online dense-model retrieval quality nor BGE reranking quality is measured
by the repo's offline lexical baseline. The unit tests inject fake encoders.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Callable
import numpy as np

from .bm25 import BM25Index, rrf_fuse
from .retrieval import KnowledgeIndex, Hit
from .vector import VectorIndex, openai_embed


class DenseSparseRRF:
    """BM25 + true dense vector embeddings + optional pairwise cross-encoder ranking."""

    def __init__(self, index: KnowledgeIndex, *,
                 embed: Callable[[list[str]], np.ndarray] = openai_embed,
                 rerank: Callable[[str, list[str]], list[float]] | None = None):
        self.index = index
        self.sparse = BM25Index(index)
        self.dense = VectorIndex(index, embed=embed)
        self.rerank = rerank

    def search(self, query: str, k: int = 3, pool_size: int = 12,
               min_dense_score: float = 0.28) -> list[Hit]:
        if not query.strip():
            return []
        if not (1 <= k <= pool_size <= 100):
            raise ValueError("need 1 <= k <= pool_size <= 100")
        candidates = rrf_fuse(
            self.sparse.search(query, k=pool_size),
            self.dense.search(query, k=pool_size, min_score=min_dense_score),
            k=pool_size,
        )
        if self.rerank is None or not candidates:
            return candidates[:k]
        scores = self.rerank(query, [h.snippet for h in candidates])
        if len(scores) != len(candidates):
            raise ValueError("Reranker must return one numeric score per candidate")
        if not all(np.isfinite(float(s)) for s in scores):
            raise ValueError("Reranker scores must be finite")
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
        return [Hit(candidates[i].id, candidates[i].doc_id, candidates[i].title,
                    candidates[i].snippet, round(float(scores[i]), 5), candidates[i].path)
                for i in order[:k]]


def local_bge_embedding(model_name: str = "BAAI/bge-m3"):
    """Requires optional `sentence-transformers`; downloads substantial model weights."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("pip install -e '.[local-dense]' for local pretrained models") from exc
    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)
    return encode


def local_cross_encoder_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """Optional BGE cross-encoder; validated only on the caller's installed environment."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("pip install -e '.[local-dense]' for local pretrained models") from exc
    model = CrossEncoder(model_name)

    def rerank(query: str, docs: list[str]) -> list[float]:
        return [float(v) for v in model.predict([(query, doc) for doc in docs])]
    return rerank


def main():
    parser = argparse.ArgumentParser(description="True dense+sparse RRF retrieval (requires embeddings)")
    parser.add_argument("query")
    parser.add_argument("--embedding", choices=["openai", "bge-m3"], default="openai")
    parser.add_argument("--rerank", action="store_true", help="load optional BGE local cross-encoder")
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2]
    index = KnowledgeIndex([base / "corpus" / "patents_source_notes"])
    encoder = openai_embed if args.embedding == "openai" else local_bge_embedding()
    reranker = local_cross_encoder_reranker() if args.rerank else None
    results = DenseSparseRRF(index, embed=encoder, rerank=reranker).search(args.query)
    print(json.dumps([h.to_dict() for h in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
