"""Four retrievers over the same chunks, plus the metrics.

bm25        lexical, pure NumPy, k1=1.2 b=0.75
bm25-canon  the same, over an OCR-canonicalized index and query
dense       MiniLM-L6-v2 cosine, pure NumPy
hybrid      reciprocal rank fusion of bm25-canon and dense, k=60
"""
import math
import re
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

# The 11-line canonicalizer an on-prem RAG box should run before indexing scans.
_CANON = str.maketrans({
    "O": "0", "o": "0", "l": "1", "I": "1", "i": "1", "|": "1",
    "S": "5", "s": "5", "B": "8", "G": "6", "Z": "2", "z": "2",
})
_CANON_PAIRS = [("-\n", ""), ("rn", "m"), ("cl", "d"), ("vv", "w"), ("nn", "m")]


def canon(text):
    t = text.replace("-\n", "").replace("\n", " ").lower()
    for a, b in _CANON_PAIRS:
        t = t.replace(a, b)
    t = t.translate(_CANON)
    return re.sub(r"\s+", " ", t)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.dl = np.array([len(t) for t in self.toks], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) or 1.0
        vocab = {}
        for t in self.toks:
            for w in set(t):
                vocab.setdefault(w, len(vocab))
        self.vocab = vocab
        N, V = len(docs), len(vocab)
        self.tf = np.zeros((N, V), dtype=np.float32)
        for i, t in enumerate(self.toks):
            for w, c in Counter(t).items():
                self.tf[i, vocab[w]] = c
        df = (self.tf > 0).sum(0)
        self.idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        denom_norm = k1 * (1 - b + b * self.dl / self.avgdl)
        self.num = self.tf * (k1 + 1.0)
        self.den = self.tf + denom_norm[:, None]
        self.w = np.divide(self.num, self.den, out=np.zeros_like(self.num),
                           where=self.den > 0) * self.idf[None, :]

    def scores(self, query):
        cols = [self.vocab[w] for w in tokenize(query) if w in self.vocab]
        if not cols:
            return np.zeros(self.w.shape[0], dtype=np.float32)
        return self.w[:, cols].sum(1)

    def nbytes(self):
        return self.w.nbytes


class Dense:
    def __init__(self, emb):
        self.emb = emb

    def scores(self, qvec):
        return self.emb @ qvec

    def nbytes(self):
        return self.emb.nbytes


def rrf(rank_lists, k=60):
    """Reciprocal rank fusion. rank_lists: list of arrays of doc ids, best first."""
    n = max(len(r) for r in rank_lists)
    s = np.zeros(n, dtype=np.float32)
    for r in rank_lists:
        s[r] += 1.0 / (k + 1.0 + np.arange(len(r)))
    return s


# ----------------------------------------------------------------- metrics --
def recall_at_k(ranked, gold, k):
    return float(gold in ranked[:k])


def rr_at_k(ranked, gold, k):
    hits = np.nonzero(ranked[:k] == gold)[0]
    return float(1.0 / (hits[0] + 1)) if len(hits) else 0.0


def answer_present(answer, texts, canonicalize=False):
    """Is the gold answer literally recoverable from the retrieved passages?"""
    blob = " ".join(texts)
    if canonicalize:
        return float(canon(answer) in canon(blob))
    return float(answer in blob)


def eval_run(rankings, questions, chunk_texts, k=5, mrr_k=10):
    """rankings: (n_queries, n_chunks) int array of chunk ids, best first."""
    out = dict(recall_at_5=0.0, mrr_at_10=0.0, ans_exact_at_5=0.0, ans_canon_at_5=0.0)
    for qi, q in enumerate(questions):
        r = rankings[qi]
        gold = q["gold_chunk_id"]
        out["recall_at_5"] += recall_at_k(r, gold, k)
        out["mrr_at_10"] += rr_at_k(r, gold, mrr_k)
        top = [chunk_texts[c] for c in r[:k]]
        out["ans_exact_at_5"] += answer_present(q["answer"], top)
        out["ans_canon_at_5"] += answer_present(q["answer"], top, canonicalize=True)
    n = len(questions)
    return {kk: v / n for kk, v in out.items()}
