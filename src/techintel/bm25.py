"""Lightweight BM25 lexical retrieval for the same chunk corpus.

This is an independent sparse-retrieval baseline; no dense embeddings are used.
"""
from __future__ import annotations
from collections import Counter
import math
import re

from .retrieval import KnowledgeIndex, Hit

_EN = re.compile(r"[a-z][a-z0-9-]*", flags=re.I)
_CJK = re.compile(r"[\u3400-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """English words + overlapping Chinese bigrams; deliberately simple tokenizer."""
    out = [m.group(0).lower() for m in _EN.finditer(text)]
    for match in _CJK.finditer(text):
        seq = match.group(0)
        if len(seq) == 1:
            out.append(seq)
        else:
            out.extend(seq[i:i+2] for i in range(len(seq)-1))
    return out


class BM25Index:
    def __init__(self, source: KnowledgeIndex, k1: float = 1.5, b: float = .75):
        self.source = source
        self.k1 = k1
        self.b = b
        self.freqs = [Counter(tokenize(c.text)) for c in source.chunks]
        self.lengths = [sum(freq.values()) for freq in self.freqs]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter()
        for freq in self.freqs:
            self.df.update(freq.keys())
        self.n = len(self.freqs)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        if not query.strip():
            return []
        if k < 1:
            raise ValueError("k must be >=1")
        terms = set(tokenize(query))
        hits = []
        for i, freq in enumerate(self.freqs):
            score = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                df = self.df[term]
                idf = math.log(1 + (self.n - df + .5) / (df + .5))
                normalizer = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                score += idf * tf * (self.k1 + 1) / normalizer
            if score > 0:
                hits.append((score, i))
        hits.sort(key=lambda x: (-x[0], x[1]))
        out = []
        for score, i in hits[:k]:
            chunk = self.source.chunks[i]
            out.append(Hit(chunk.id, chunk.doc_id, chunk.title, chunk.text[:750],
                           round(score, 5), chunk.path))
        return out


def rrf_fuse(*ranked_hit_lists: list[Hit], k: int = 3, smoothing: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion for BM25 and optional dense search hit lists."""
    scores: dict[str, float] = {}
    canonical: dict[str, Hit] = {}
    for ranked in ranked_hit_lists:
        for rank, h in enumerate(ranked, 1):
            canonical[h.id] = h
            scores[h.id] = scores.get(h.id, 0) + 1 / (smoothing + rank)
    order = sorted(scores, key=lambda h: (-scores[h], h))[:k]
    return [Hit(canonical[x].id, canonical[x].doc_id, canonical[x].title,
                canonical[x].snippet, round(scores[x], 6), canonical[x].path) for x in order]
