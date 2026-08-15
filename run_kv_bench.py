"""KV-cache compression: memory, latency, and the quality delta that pays for them.

Setup mirrors the RAG case that motivates it: a long retrieved context (the prefill)
followed by a short answer (the decode). Bits are applied to the cache only; weights
stay fp32.

    python run_kv_bench.py                      # default 768-token prefill, 64 decode
    python run_kv_bench.py --prefill 512 --decode 32
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
from gpt2_kv import GPT2, KVCache, log_softmax

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
BITS = [32, 16, 8, 4]

# Ordinary prose, written for this benchmark, deliberately not in the corpus.
HELD_OUT = (
    "A deployment inside a customer perimeter is judged on a fixed hardware budget "
    "rather than on a leaderboard. The operator has a box with a known amount of "
    "memory and a known number of cores, and the question that decides whether the "
    "system ships is whether the working set fits and the answer arrives before the "
    "reviewer loses patience. Compression changes that arithmetic, but it is only "
    "worth taking if somebody has measured what it costs in quality and published "
    "the number alongside the saving instead of quietly leaving it out."
)


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", type=int, default=768)
    ap.add_argument("--decode", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    chunks, _ = build_corpus()
    doc = " ".join(c["doc_title"] + " " + c["section"] + " " + c["text"] for c in chunks)
    m = GPT2()
    ctx_ids = m.tok.encode(doc).ids
    if len(ctx_ids) < args.prefill:
        ctx_ids = ctx_ids * (args.prefill // len(ctx_ids) + 1)
    prefill = ctx_ids[:args.prefill]
    # The scored continuation is NOT drawn from the corpus. The corpus is templated
    # boilerplate, and scoring it lets induction heads copy near-perfectly: perplexity
    # comes out near 1.0 and the quality delta between bit-widths becomes invisible.
    # Scoring ordinary prose instead keeps perplexity in a normal range, so a
    # quantization error large enough to matter would actually show up.
    tail = m.tok.encode(HELD_OUT).ids
    if len(tail) < args.decode:
        tail = tail * (args.decode // len(tail) + 1)
    tail = tail[:args.decode]

    rows, ref_argmax, ref_nll = [], None, None
    for bits in BITS:
        cache = KVCache(bits)
        t0 = time.perf_counter()
        logits = m.forward(np.array(prefill), cache, 0)
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        lat, nll, argmax_ids = [], [], []
        cur = logits[-1]
        for step in range(args.decode):
            lp = log_softmax(cur)
            nll.append(-float(lp[tail[step]]))
            argmax_ids.append(int(cur.argmax()))
            # teacher forcing on the real next token: perplexity is then comparable
            # across bit-widths instead of measuring divergent generations
            t0 = time.perf_counter()
            cur = m.forward(np.array([tail[step]]), cache, args.prefill + step)[-1]
            lat.append((time.perf_counter() - t0) * 1000.0)

        kv_bytes = cache.nbytes()
        ppl = float(np.exp(np.mean(nll)))
        if bits == 32:
            ref_argmax, ref_nll = argmax_ids, ppl
        agree = float(np.mean([a == b for a, b in zip(argmax_ids, ref_argmax)]))
        rows.append(dict(
            kv_bits=bits,
            kv_cache_mb=round(kv_bytes / 1e6, 2),
            kv_bytes_per_token=round(kv_bytes / (args.prefill + args.decode), 1),
            mem_vs_fp32=round(kv_bytes / (rows[0]["kv_cache_mb"] * 1e6), 3) if rows else 1.0,
            prefill_ms=round(prefill_ms, 1),
            decode_p50_ms=round(statistics.median(lat), 2),
            decode_p95_ms=round(sorted(lat)[int(0.95 * len(lat))], 2),
            perplexity=round(ppl, 4),
            ppl_delta_pct=round(100.0 * (ppl - ref_nll) / ref_nll, 3),
            top1_agree_vs_fp32=round(agree, 4),
            peak_rss_mb=round(rss_mb(), 1),
        ))
        print(f"  {bits:2d}-bit KV: {rows[-1]['kv_cache_mb']:6.2f} MB  "
              f"decode p50 {rows[-1]['decode_p50_ms']:6.2f} ms  "
              f"ppl {ppl:8.4f} ({rows[-1]['ppl_delta_pct']:+.2f}%)  "
              f"top-1 agree {agree:.3f}", flush=True)

    meta = dict(model="openai-community/gpt2 (124M) pure NumPy forward pass, fp32 weights",
                prefill_tokens=args.prefill, decode_tokens=args.decode,
                quant="asymmetric min/max, per (layer, head, token) over 64 head dims; "
                      "int4 bit-packed two nibbles per byte",
                scale_overhead="2 fp32 values per (layer, head, token) per K and V, "
                               "counted in kv_cache_mb",
                note="Weights are fp32 throughout. Only the cache is compressed. "
                     "Perplexity is teacher-forced on the same token sequence for "
                     "every bit-width, so the numbers are directly comparable.",
                numpy=np.__version__, python=sys.version.split()[0], platform=sys.platform)
    with open(os.path.join(OUT, "kv_bench.json"), "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1)
    print(f"\nwrote {OUT}/kv_bench.json")


if __name__ == "__main__":
    main()
