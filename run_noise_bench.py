"""Degradation curve: on-prem RAG accuracy, latency and memory vs document noise.

    python run_noise_bench.py            # full run, writes results/noise_bench.json + .md
    python run_noise_bench.py --quick    # clean + scan-degraded only, for a smoke test
"""
import argparse
import json
import os
import resource
import statistics
import sys
import time

import numpy as np

from corpus import build_corpus
from minilm import MiniLM
from noise import CONDITIONS, char_error_rate, corrupt_chunks
from retrieval import BM25, Dense, canon, eval_run, rrf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
RETRIEVERS = ["bm25", "bm25-canon", "dense", "hybrid"]


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def rank_all(score_fn, queries, n):
    """Returns (rankings, per-query latencies in ms)."""
    ranks, lat = np.empty((len(queries), n), dtype=np.int32), []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        s = score_fn(i, q)
        r = np.argsort(-s, kind="stable")
        lat.append((time.perf_counter() - t0) * 1000.0)
        ranks[i] = r
    return ranks, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    clean_chunks, questions = build_corpus()
    qtexts = [q["question"] for q in questions]
    n = len(clean_chunks)

    conds = CONDITIONS
    if args.quick:
        conds = [c for c in CONDITIONS if c[0] in ("clean", "scan-degraded")]

    print(f"corpus: {n} chunks, {len(questions)} questions, "
          f"{len(conds)} conditions x {len(RETRIEVERS)} retrievers", flush=True)

    t_model = time.perf_counter()
    model = MiniLM()
    model_load_s = time.perf_counter() - t_model
    rss_after_model = peak_rss_mb()

    # Queries are clean: the lawyer types the question correctly. Only the archive
    # is damaged. Encode them once.
    qvecs = model.encode(qtexts)
    qvecs_canon = model.encode([canon(t) for t in qtexts])

    rows = []
    serving_rss = None
    for label, spec in conds:
        t_cond = time.perf_counter()
        noisy = corrupt_chunks(clean_chunks, spec, seed=7)
        cer = statistics.mean(char_error_rate(a["text"], b["text"])
                              for a, b in zip(clean_chunks, noisy))
        texts = [c["indexed"] for c in noisy]
        ctexts = [canon(t) for t in texts]

        t0 = time.perf_counter()
        bm = BM25(texts)
        t_bm = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        bmc = BM25(ctexts)
        t_bmc = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        emb = model.encode(texts)
        t_emb = (time.perf_counter() - t0) * 1000.0
        dn = Dense(emb)

        runs = {}
        runs["bm25"] = rank_all(lambda i, q: bm.scores(q), qtexts, n)
        runs["bm25-canon"] = rank_all(lambda i, q: bmc.scores(canon(q)), qtexts, n)
        runs["dense"] = rank_all(lambda i, q: dn.scores(qvecs[i]), qtexts, n)

        bmc_ranks = runs["bm25-canon"][0]
        dn_ranks = runs["dense"][0]
        hyb_ranks, hyb_lat = np.empty((len(qtexts), n), np.int32), []
        for i in range(len(qtexts)):
            t0 = time.perf_counter()
            s = rrf([bmc_ranks[i], dn_ranks[i]])
            hyb_ranks[i] = np.argsort(-s, kind="stable")
            # fusion cost sits on top of both parents
            hyb_lat.append((time.perf_counter() - t0) * 1000.0
                           + runs["bm25-canon"][1][i] + runs["dense"][1][i])
        runs["hybrid"] = (hyb_ranks, hyb_lat)
        if serving_rss is None:
            serving_rss = peak_rss_mb()   # model + all four indexes + queries served

        build_ms = {"bm25": t_bm, "bm25-canon": t_bmc, "dense": t_emb,
                    "hybrid": t_bmc + t_emb}
        idx_bytes = {"bm25": bm.nbytes(), "bm25-canon": bmc.nbytes(),
                     "dense": dn.nbytes(), "hybrid": bmc.nbytes() + dn.nbytes()}

        for name in RETRIEVERS:
            ranks, lat = runs[name]
            m = eval_run(ranks, questions, texts)
            rows.append(dict(
                condition=label, retriever=name, cer=round(cer, 4),
                recall_at_5=round(m["recall_at_5"], 4),
                mrr_at_10=round(m["mrr_at_10"], 4),
                ans_exact_at_5=round(m["ans_exact_at_5"], 4),
                ans_canon_at_5=round(m["ans_canon_at_5"], 4),
                p50_query_ms=round(statistics.median(lat), 3),
                p95_query_ms=round(sorted(lat)[int(0.95 * len(lat))], 3),
                index_build_ms=round(build_ms[name], 1),
                index_mb=round(idx_bytes[name] / 1e6, 2),
            ))
        print(f"  {label:22s} CER={cer:.3f}  "
              + "  ".join(f"{r['retriever']}:R@5={r['recall_at_5']:.2f}"
                          for r in rows[-4:])
              + f"   [{time.perf_counter()-t_cond:.1f}s]", flush=True)

    meta = dict(
        n_chunks=n, n_questions=len(questions),
        model="sentence-transformers/all-MiniLM-L6-v2 (pure NumPy forward pass)",
        model_load_s=round(model_load_s, 2),
        peak_rss_mb_after_model_load=round(rss_after_model, 1),
        peak_rss_mb_serving=round(serving_rss, 1),
        peak_rss_mb_total_incl_ingest=round(peak_rss_mb(), 1),
        encode_batch_size=8,
        numpy=np.__version__, python=sys.version.split()[0],
        platform=sys.platform,
        note="Queries are clean; only the indexed corpus is corrupted.",
    )
    with open(os.path.join(OUT, "noise_bench.json"), "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1)

    hdr = ("| condition | CER | retriever | R@5 | MRR@10 | ans exact@5 | ans canon@5 "
           "| p50 ms | p95 ms | index MB |")
    lines = [hdr, "|" + "---|" * 10]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['cer']:.3f} | {r['retriever']} | "
            f"{r['recall_at_5']:.3f} | {r['mrr_at_10']:.3f} | "
            f"{r['ans_exact_at_5']:.3f} | {r['ans_canon_at_5']:.3f} | "
            f"{r['p50_query_ms']:.2f} | {r['p95_query_ms']:.2f} | {r['index_mb']:.2f} |")
    with open(os.path.join(OUT, "noise_bench.md"), "w") as f:
        f.write("\n".join(lines) + "\n\n```json\n" + json.dumps(meta, indent=1) + "\n```\n")

    print(f"\npeak RSS total: {peak_rss_mb():.1f} MB "
          f"(after model load: {rss_after_model:.1f} MB)")
    print(f"wrote {OUT}/noise_bench.json and .md")


if __name__ == "__main__":
    main()
