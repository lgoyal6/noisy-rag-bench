# noisy-rag-bench

A degradation curve for on-prem RAG under realistic document noise, plus a KV-cache
compression table. CPU only, NumPy only, no PyTorch, no network at run time.

## Why

Retrieval benchmarks are run on clean text. Documents inside a bank or a law firm are
not clean: they are scans, they have running headers and Bates stamps, words are
hyphenated across line breaks, and spans have been redacted out. The interesting
question is not "does RAG work" but "how fast does it fall over as the input degrades,
and what does that cost in RAM and milliseconds on a fixed box."

The corpus and the questions are generated here, so this is shareable. No client data,
nothing scraped, nothing under licence.

## What it measures

Five noise families, at three to four severities each, plus two combined "scan" profiles:

| family | what it models |
|---|---|
| `ocr` | character confusions from a real OCR pass: `rn`/`m`, `0`/`O`, `l`/`1`, `cl`/`d`, `vv`/`w`, `S`/`5`, `B`/`8` |
| `whitespace` | dropped spaces (words fuse) and spurious spaces (words split) |
| `hyphenation` | words broken across a line end and never rejoined |
| `header` | running headers, footers, page numbers and Bates stamps injected into the body |
| `redaction` | blackouts that delete word spans outright, leaving no marker |

Four retrievers over the same chunks:

- `bm25` lexical, k1=1.2 b=0.75
- `bm25-canon` the same over an OCR-canonicalized index and query
- `dense` all-MiniLM-L6-v2 cosine, forward pass reimplemented in NumPy
- `hybrid` reciprocal rank fusion of the two, k=60

Metrics per condition per retriever: `recall@5`, `MRR@10`, whether the gold answer is
literally recoverable from the top-5 passages (`ans_exact@5`) and whether it is
recoverable after canonicalization (`ans_canon@5`), plus p50/p95 query latency, index
build time and index size. Peak RSS is measured per retriever in its own process.

## Files

```
corpus.py          132 chunks / 132 questions: credit agreements + engagement letters
noise.py           the five noise families, the condition ladder, and a CER meter
minilm.py          all-MiniLM-L6-v2 forward pass in pure NumPy over safetensors
retrieval.py       BM25, dense, RRF, the canonicalizer, and the metrics
run_noise_bench.py the degradation curve                  -> results/noise_bench.{json,md}
gpt2_kv.py         GPT-2 small in pure NumPy with a real bit-packed quantized KV cache
run_kv_bench.py    KV memory / latency / quality table    -> results/kv_bench.json
probe.py           honest per-retriever latency and peak RSS, one process each
```

## Run

```bash
uv venv --python 3.12 ../work/.venv
uv pip install --python ../work/.venv/bin/python numpy safetensors tokenizers psutil huggingface_hub
../work/.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('sentence-transformers/all-MiniLM-L6-v2')"
../work/.venv/bin/python -c "from huggingface_hub import hf_hub_download as d; [d('openai-community/gpt2', f) for f in ['config.json','tokenizer.json','model.safetensors']]"

../work/.venv/bin/python run_noise_bench.py                         # ~68 s
../work/.venv/bin/python run_kv_bench.py --prefill 768 --decode 64  # ~8 s
for m in bm25 dense hybrid ingest; do ../work/.venv/bin/python probe.py $m; done
```

`run_noise_bench.py --quick` runs clean and scan-degraded only, as a smoke test.

Only `run_kv_bench.py` needs GPT-2. Everything else runs on the 90 MB MiniLM checkpoint.

## Swapping in real documents

Replace `build_corpus()` in `corpus.py` with a loader that returns the same two lists:
chunks as `{id, doc_id, doc_title, section, text}` and questions as
`{qid, question, answer, gold_chunk_id}`. Nothing else changes. If the real documents are
already noisy, run with `CONDITIONS = [("as-is", {})]` to get the single operating point,
or keep the ladder to see how much further headroom there is.

## Findings

See `../RESULTS.md`. The short version: retrieval recall is far more noise-tolerant than
answer accuracy, so a recall-only eval will call a broken configuration healthy; an 11-line
OCR normalizer buys back more accuracy than changing retrievers does; character error rate
is a poor predictor of which noise actually hurts; and fp16 KV cache is free while int4
costs 11% perplexity and changes 16% of greedy decisions.
