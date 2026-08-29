#!/usr/bin/env python3
"""Generate the Hugging Face dataset card from the committed benchmark run.

Every number in the card is read out of `results/noise_bench.json`. Nothing is
typed by hand, because the whole point of this repository is that a number
quoted away from the run that produced it drifts from it.

    python scripts/dataset-card.py > /tmp/README.md
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# The retriever the top-level README reports. Named once so the card and the
# README cannot disagree about which system the headline belongs to.
HEADLINE = "hybrid"
BASELINE = "bm25-canon"


def main():
    bench = json.loads((RESULTS / "noise_bench.json").read_text(encoding="utf-8"))
    meta, rows = bench["meta"], bench["rows"]

    by = {}
    for r in rows:
        by.setdefault(r["condition"], {})[r["retriever"]] = r
    conds = list(dict.fromkeys(r["condition"] for r in rows))

    h = {c: by[c][HEADLINE] for c in conds}
    clean = h["clean"]
    ocr5 = h["ocr cer~5%"]
    hdr = h["header x3"]

    # The baseline's win count, on retrieval quality, which is the axis it wins on.
    beats = sum(1 for c in conds if by[c][BASELINE]["recall_at_5"] >= h[c]["recall_at_5"])

    out = []
    w = out.append

    w("---")
    w("license: mit")
    w("language:\n- en")
    w("task_categories:\n- question-answering\n- text-retrieval")
    w("tags:\n- rag\n- retrieval\n- ocr\n- document-noise\n- robustness\n- benchmark")
    w("size_categories:\n- n<1K")
    w("configs:")
    w("- config_name: questions\n  data_files: questions.jsonl\n  default: true")
    w("- config_name: corpus\n  data_files: corpus.jsonl")
    w("---")
    w("")
    w("# noisy-rag-bench")
    w("")
    w("A retrieval corpus and QA set for measuring what realistic document noise does to a")
    w("RAG pipeline, plus the benchmark run over 18 noise conditions.")
    w("")
    w("Retrieval benchmarks run on clean text. Documents inside a bank or a law firm are")
    w("scans: OCR confusions, running headers, hyphens broken across lines, redacted spans.")
    w("This is the corpus for measuring that, and the result it was built to expose.")
    w("")
    w("Code and full write-up: https://github.com/lgoyal6/noisy-rag-bench")
    w("")
    w("## The finding")
    w("")
    # Recall under this condition is in fact marginally higher than on clean text
    # (0.9848 against 0.9773). Saying "no lower than" rather than "the same as"
    # keeps that true at any rounding.
    held = ocr5["recall_at_5"] >= clean["recall_at_5"]
    w(f"**At {ocr5['cer'] * 100:.1f}% character error, recall@5 is {ocr5['recall_at_5']:.2f}, "
      + (f"no lower than the {clean['recall_at_5']:.2f} measured on clean documents"
         if held else f"down from {clean['recall_at_5']:.2f} on clean documents")
      + f", while answer recoverability has fallen from {clean['ans_exact_at_5']:.2f} to "
        f"{ocr5['ans_exact_at_5']:.2f}.** A dashboard")
    w("tracking retrieval quality reports that system as healthy. Retrieval still finds the")
    w("right page; the figure on that page is no longer legible.")
    w("")
    w(f"Canonicalising the text recovers {ocr5['ans_exact_at_5']:.2f} to "
      f"{ocr5['ans_canon_at_5']:.2f} at the same condition.")
    w("")
    w(f"**Character error rate is a poor predictor of damage.** `header x3` carries "
      f"{hdr['cer'] * 100:.1f}% character error, {hdr['cer'] / ocr5['cer']:.1f} times the OCR case "
      f"above, and costs nothing: recall@5 {hdr['recall_at_5']:.2f}, answers "
      f"{hdr['ans_exact_at_5']:.2f}. What kind of noise it is matters far more than how much.")
    w("")
    w("## Files")
    w("")
    w("| file | rows | what it is |")
    w("|---|---:|---|")
    w(f"| `questions.jsonl` | {meta['n_questions']} | Gold QA pairs. `qid`, `question`, `answer`, "
      "`gold_chunk_id`, `family`. |")
    w(f"| `corpus.jsonl` | {meta['n_chunks']} | The chunks to index. `id`, `doc_id`, `doc_title`, "
      "`section`, `text`. `gold_chunk_id` joins to `id`. |")
    w(f"| `noise_bench.json` | {len(rows)} | The run: {len(conds)} noise conditions across "
      f"{len({r['retriever'] for r in rows})} retrievers. |")
    w("")
    w("The corpus is **synthetic and generated**, seeded and reproducible from `corpus.py` in")
    w("the repository. No client data, nothing scraped. The shape is the realistic part:")
    w("near-duplicate boilerplate across many counterparties, so answering requires telling")
    w("entity A's threshold from entity B's, which is exactly what OCR noise destroys.")
    w("")
    w(f"## The curve ({HEADLINE} retriever)")
    w("")
    w("`recall@5` is whether the right chunk was retrieved. `answer` is whether the answer was")
    w("still extractable from what came back. `+canon` is after an 11-line canonicalizer.")
    w("")
    w("| condition | CER | recall@5 | answer | +canon |")
    w("|---|---:|---:|---:|---:|")
    for c in conds:
        r = h[c]
        w(f"| {c} | {r['cer'] * 100:.1f}% | {r['recall_at_5']:.2f} | "
          f"{r['ans_exact_at_5']:.2f} | {r['ans_canon_at_5']:.2f} |")
    w("")
    w("## The losing axis")
    w("")
    w(f"A canonicalised BM25 baseline matches or beats the {HEADLINE} retriever on recall@5 at")
    w(f"**{beats} of {len(conds)} conditions**. Both are in `noise_bench.json`, so the")
    w("comparison can be redone on any column rather than taken from this sentence.")
    w("")
    w("## Scope and limits")
    w("")
    w(f"- {meta['n_questions']} questions over {meta['n_chunks']} chunks. Small, and single-hop.")
    w(f"- Queries are clean; only the indexed corpus is corrupted.")
    w(f"- Dense retrieval is {meta['model']}, CPU only, "
      f"peak RSS {meta['peak_rss_mb_serving']:.0f} MB serving.")
    w("- The corpus is synthetic. It is built to have realistic *structure*, not to be a sample")
    w("  of real documents, and results on it transfer as a method rather than as numbers.")
    w("")
    w("## Licence")
    w("")
    w("MIT. Generated corpus, no third-party document data.")

    print("\n".join(out))


if __name__ == "__main__":
    main()
