"""Honest on-prem cost numbers: end-to-end single-query latency and peak RSS.

The per-query latency in run_noise_bench.py excludes query encoding, because
queries are encoded once in a batch there. That understates dense retrieval.
This probe measures the real thing: one query at a time, encode included.

    python probe.py bm25 | dense | hybrid | ingest
Each mode runs in its own process so ru_maxrss is that mode's own peak.
"""
import resource
import statistics
import sys
import time

import numpy as np

from corpus import build_corpus
from minilm import MiniLM
from noise import CONDITIONS, corrupt_chunks
from retrieval import BM25, Dense, canon, rrf

SPEC = dict(CONDITIONS)["scan-realistic"]


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def report(mode, lat, extra=""):
    lat = sorted(lat)
    print(f"{mode:8s} p50={statistics.median(lat)*1000:8.3f} ms  "
          f"p95={lat[int(0.95*len(lat))]*1000:8.3f} ms  "
          f"peak_rss={rss_mb():7.1f} MB  {extra}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    chunks, questions = build_corpus()
    noisy = corrupt_chunks(chunks, SPEC, seed=7)
    texts = [c["indexed"] for c in noisy]
    qs = [q["question"] for q in questions]

    if mode == "bm25":
        bm = BM25([canon(t) for t in texts])
        lat = []
        for q in qs:
            t0 = time.perf_counter()
            np.argsort(-bm.scores(canon(q)), kind="stable")[:5]
            lat.append(time.perf_counter() - t0)
        report("bm25", lat, "(index: dense tf matrix, this implementation)")
        return

    model = MiniLM()
    if mode == "ingest":
        t0 = time.perf_counter()
        emb = model.encode(texts, batch_size=8)
        dt = time.perf_counter() - t0
        chars = sum(len(t) for t in texts)
        print(f"ingest   {len(texts)} chunks in {dt*1000:.0f} ms = "
              f"{len(texts)/dt:.1f} chunks/s, {chars/dt/1000:.1f} kchar/s, "
              f"emb {emb.nbytes/1e6:.2f} MB, peak_rss={rss_mb():.1f} MB")
        return

    emb = model.encode(texts, batch_size=8)
    dn = Dense(emb)
    bm = BM25([canon(t) for t in texts])
    lat = []
    for q in qs:
        t0 = time.perf_counter()
        qv = model.encode([q])[0]
        ds = dn.scores(qv)
        if mode == "dense":
            np.argsort(-ds, kind="stable")[:5]
        else:
            bs = bm.scores(canon(q))
            np.argsort(-rrf([np.argsort(-bs, kind="stable"),
                             np.argsort(-ds, kind="stable")]), kind="stable")[:5]
        lat.append(time.perf_counter() - t0)
    report(mode, lat, "(query encode included)")


if __name__ == "__main__":
    main()
