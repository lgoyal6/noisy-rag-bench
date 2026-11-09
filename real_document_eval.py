"""Evaluate receipt evidence recovery on the independently annotated CORD test split.

The user selects a receipt before retrieval. Fixed spatial bands, constructed
without labels, are the retrievable passages. This is evidence recovery, not
answer generation, a CORD parsing leaderboard, or a template-heldout claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from heldout_eval import _answer_occurs
from retrieval import BM25, Dense, canon, rrf

REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
PARQUET_SHA256 = "51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7"
VALIDATION_SHA256 = "0d0f6dac11fdcc549de2746aa9f53136a3bc22a2a1aff2b0b847f7622ad60c15"
# Fixed before viewing test results. Queries never contain answer values.
QUESTIONS = {
    "total.total_price": "TOTAL GRAND TOTAL JUMLAH TOTAL BAYAR",
    "total.cashprice": "CASH TUNAI BAYAR",
    "total.changeprice": "CHANGE KEMBALI KEMBALIAN",
    "sub_total.subtotal_price": "SUBTOTAL SUB TOTAL",
    "sub_total.tax_price": "TAX PAJAK PPN PB1",
    "sub_total.service_price": "SERVICE SERVIS LAYANAN",
    "sub_total.discount_price": "DISCOUNT DISC DISKON POTONGAN",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def band_for(y: float, height: int, bands: int) -> int:
    if height <= 0 or bands < 2:
        raise ValueError("positive image height and at least two bands required")
    return max(0, min(bands - 1, int(y * bands / height)))


def word_center(word):
    quad = word["quad"]
    return (sum(quad[f"x{i}"] for i in range(1, 5)) / 4,
            sum(quad[f"y{i}"] for i in range(1, 5)) / 4)


def reference_receipt(ground_truth: dict, *, bands: int = 12):
    """Use published labels as the oracle, never as OCR text or query features."""
    height = ground_truth["meta"]["image_size"]["height"]
    passages = [[] for _ in range(bands)]
    by_category = {}
    for line in ground_truth["valid_line"]:
        words = line["words"]
        for word in words:
            x, y = word_center(word)
            passages[band_for(y, height, bands)].append((y, x, word["text"]))
        by_category.setdefault(line["category"], []).append(line)
    questions, excluded = [], Counter()
    for category, query in QUESTIONS.items():
        group, key = category.split(".")
        section = ground_truth["gt_parse"].get(group, {})
        answer = section.get(key) if isinstance(section, dict) else None
        lines = by_category.get(category, [])
        if not isinstance(answer, str) or not answer.strip() or len(lines) != 1:
            excluded["missing_or_repeated_field"] += 1
            continue
        words = lines[0]["words"]
        # A publisher parse value not supported by its own line is retained as
        # an explicit exclusion, not silently treated as an OCR failure.
        if not _answer_occurs(answer, " ".join(w["text"] for w in words)):
            excluded["label_not_in_annotated_line"] += 1
            continue
        gold_bands = sorted({band_for(word_center(w)[1], height, bands) for w in words})
        questions.append({"category": category, "query": query, "answer": answer,
                          "gold_bands": gold_bands})
    return [" ".join(w[2] for w in sorted(ws)) for ws in passages], questions, dict(excluded)


def tsv_passages(tsv: str, height: int, bands: int):
    passages = [[] for _ in range(bands)]
    confidences = []
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t")
    if not {"level", "left", "top", "height", "conf", "text"}.issubset(reader.fieldnames or []):
        raise ValueError("Tesseract did not produce the required word-level TSV")
    for row in reader:
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        y = float(row["top"]) + float(row["height"]) / 2
        passages[band_for(y, height, bands)].append((float(row["top"]), float(row["left"]), text))
        confidences.append(float(row["conf"]))
    return [" ".join(w[2] for w in sorted(ws)) for ws in passages], confidences


def score_ranking(ranking, question, texts, *, k: int = 3):
    if not 1 <= k <= len(texts):
        raise ValueError("k must fit the passage collection")
    ids = [int(i) for i in ranking[:k]]
    if len(ids) != len(set(ids)) or any(i < 0 or i >= len(texts) for i in ids):
        raise ValueError("ranking must contain distinct valid passage ids")
    gold = set(question["gold_bands"])
    answer = question["answer"]
    hits = [i for i, band in enumerate(ranking) if int(band) in gold]
    strict = [i for i in ids if _answer_occurs(answer, texts[i])]
    canonical = [i for i in ids if _answer_occurs(canon(answer), canon(texts[i]))]
    return {
        "recall": float(bool(gold.intersection(ids))),
        "reciprocal_rank": 1 / (hits[0] + 1) if hits else 0.0,
        "strict_grounded_recovery": float(bool(gold.intersection(strict))),
        "strict_gold_region_available": float(any(_answer_occurs(answer, texts[i]) for i in gold)),
        "strict_any_band_recovery": float(bool(strict)),
        "canonical_grounded_recovery": float(bool(gold.intersection(canonical))),
        "wrong_band_only_support": float(bool(strict) and not gold.intersection(strict)),
        # Canonical equivalence is not exact correctness. A collision here
        # must never be promoted to a correctly extracted financial value.
        "canonical_only_recovery": float(bool(gold.intersection(canonical))
                                         and not gold.intersection(strict)),
    }


def paired_document_interval(receipts, left: str, right: str, metric: str, *, seed=1729):
    """Resample documents, not correlated questions. Differences are left-right."""
    values = []
    for doc in sorted({r["document_id"] for r in receipts}):
        scores = {}
        for arm in [left, right]:
            rows = [r[metric] for r in receipts if r["document_id"] == doc and r["arm"] == arm]
            if not rows:
                raise ValueError("paired arms must cover identical documents")
            scores[arm] = float(np.mean(rows))
        values.append(scores[left] - scores[right])
    if not values:
        raise ValueError("no paired documents")
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = np.mean(values[rng.integers(0, len(values), size=(2000, len(values)))], axis=1)
    return {"document_count": len(values), "macro_document_difference": float(values.mean()),
            "bootstrap_95pct": np.quantile(means, [.025, .975]).tolist(), "seed": seed}


def evaluate_receipt(texts, questions, *, condition, document_id, model=None):
    build_start = time.perf_counter()
    bm, bmc = BM25(texts), BM25([canon(t) for t in texts])
    dense = Dense(model.encode(texts)) if model else None
    build_ms = (time.perf_counter() - build_start) * 1000
    rows = []
    for q in questions:
        ranks, timings = {}, {}
        for name, index, query in [("bm25", bm, q["query"]), ("bm25-canon", bmc, canon(q["query"]))]:
            start = time.perf_counter()
            ranks[name] = np.argsort(-index.scores(query), kind="stable")
            timings[name] = (time.perf_counter() - start) * 1000
        if dense is not None:
            start = time.perf_counter()
            vector = model.encode([q["query"]])[0]
            ranks["dense"] = np.argsort(-dense.scores(vector), kind="stable")
            timings["dense"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            ranks["hybrid"] = np.argsort(-rrf([ranks["bm25-canon"], ranks["dense"]]), kind="stable")
            timings["hybrid"] = timings["bm25-canon"] + timings["dense"] + (time.perf_counter() - start) * 1000
        for name, ranking in ranks.items():
            rows.append({"document_id": document_id, "category": q["category"],
                         "arm": condition + "/" + name, "ranking": ranking.tolist(),
                         "query_latency_ms_including_encoding": timings[name],
                         "index_build_ms_all_arms": build_ms,
                         **score_ranking(ranking, q, texts)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tessdata-dir", type=Path, required=True)
    ap.add_argument("--lang", default="eng+ind")
    ap.add_argument("--split", choices=["test", "validation"], default="test")
    ap.add_argument("--psm", type=int, choices=[6, 11], default=6)
    ap.add_argument("--dense", action="store_true")
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from PIL import Image

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise SystemExit("Existing report will not be overwritten; choose a fresh output directory")
    version = subprocess.check_output(["tesseract", "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
    language_hashes = {lang: digest((args.tessdata_dir / (lang + ".traineddata")).read_bytes())
                       for lang in args.lang.split("+")}
    code = {name: digest(Path(__file__).with_name(name).read_bytes())
            for name in ["real_document_eval.py", "retrieval.py", "heldout_eval.py", "minilm.py"]}
    parquet_hash = digest(args.parquet.read_bytes())
    expected = PARQUET_SHA256 if args.split == "test" else VALIDATION_SHA256
    if parquet_hash != expected:
        raise SystemExit("Input does not match the pinned publisher CORD split parquet")
    model_files = {}
    if args.dense:
        from minilm import _snapshot
        snapshot = Path(_snapshot())
        model_files = {name: digest((snapshot / name).read_bytes())
                       for name in ["model.safetensors", "tokenizer.json"]}
    protocol = {"dataset": "naver-clova-ix/cord-v2", "revision": REVISION,
                "parquet_sha256": parquet_hash, "split": args.split, "model_files": model_files,
                "bands": 12, "k": 3, "questions": QUESTIONS, "tesseract": version,
                "language": args.lang, "language_hashes": language_hashes,
                "psm": args.psm, "tessedit_create_tsv": True, "dense": args.dense, "code": code}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise SystemExit("Protocol changed; use a fresh output directory")
    protocol_path.write_text(json.dumps(protocol, indent=2))
    model = None
    if args.dense:
        from minilm import MiniLM
        model = MiniLM()
    receipts, documents, excluded = [], [], Counter()
    records = pq.read_table(args.parquet).to_pylist()
    for i, record in enumerate(records):
        gt = json.loads(record["ground_truth"])
        if gt["meta"]["split"] != ("valid" if args.split == "validation" else "test"):
            raise ValueError("Publisher split metadata does not match the declared split")
        doc = f"cord-{args.split}-{gt['meta']['image_id']}"
        reference, questions, omissions = reference_receipt(gt)
        excluded.update(omissions)
        data = record["image"]["bytes"]
        image = Image.open(io.BytesIO(data))
        if image.height != gt["meta"]["image_size"]["height"]:
            raise ValueError("image and annotation coordinates differ")
        tsv_path = output / (doc + ".tsv")
        ocr_started = time.perf_counter()
        if not tsv_path.exists():
            with tempfile.TemporaryDirectory(prefix="cord-ocr-") as temp:
                path = Path(temp) / "receipt.png"
                image.save(path)
                result = subprocess.run(["tesseract", str(path), "stdout", "--tessdata-dir",
                                         str(args.tessdata_dir), "-l", args.lang,
                                         "--psm", str(args.psm), "-c", "tessedit_create_tsv=1"],
                                        capture_output=True, timeout=60, check=True)
            tsv_passages(result.stdout.decode(), image.height, 12)
            tsv_path.write_bytes(result.stdout)
            (output / (doc + ".ocr-stderr.txt")).write_bytes(result.stderr)
            elapsed = (time.perf_counter() - ocr_started) * 1000
            (output / (doc + ".ocr-time.json")).write_text(json.dumps({"ocr_ms": elapsed}))
        ocr_ms = json.loads((output / (doc + ".ocr-time.json")).read_text())["ocr_ms"]
        texts, confidence = tsv_passages(tsv_path.read_text(), image.height, 12)
        documents.append({"document_id": doc, "image_sha256": digest(data), "ocr_ms": ocr_ms,
                          "annotation_sha256": digest(record["ground_truth"].encode()),
                          "tsv_sha256": digest(tsv_path.read_bytes()), "questions": questions,
                          "reference_passages": reference, "ocr_passages": texts,
                          "ocr_word_confidence": confidence})
        for condition, passages in [("annotated_reference", reference), ("actual_ocr", texts)]:
            receipts.extend(evaluate_receipt(passages, questions, condition=condition,
                                             document_id=doc, model=model))
        print(f"{i+1}/{len(records)} {doc}: {len(questions)} fields", flush=True)
    metrics = ["recall", "reciprocal_rank", "strict_grounded_recovery", "strict_gold_region_available", "strict_any_band_recovery",
               "canonical_grounded_recovery", "wrong_band_only_support", "canonical_only_recovery"]
    summaries = {}
    for arm in sorted({r["arm"] for r in receipts}):
        rows = [r for r in receipts if r["arm"] == arm]
        summaries[arm] = {m: float(np.mean([r[m] for r in rows])) for m in metrics}
        summaries[arm].update(n_questions=len(rows), n_documents=len({r["document_id"] for r in rows}),
                              query_p50_ms=float(np.median([r["query_latency_ms_including_encoding"] for r in rows])),
                              query_p95_ms=float(np.quantile([r["query_latency_ms_including_encoding"] for r in rows], .95)))
    comparisons = {}
    for retriever in ["bm25", "bm25-canon"] + (["dense", "hybrid"] if model else []):
        comparisons[retriever] = paired_document_interval(receipts, "actual_ocr/" + retriever,
                                                         "annotated_reference/" + retriever,
                                                         "strict_grounded_recovery")
    report = {"protocol": protocol, "summary": summaries, "ocr_minus_reference": comparisons,
              "excluded_fields": dict(excluded), "documents": documents, "receipts": receipts,
              "scope": "Declared released CORD split; selected-document retrieval over fixed spatial bands. Published parse/box labels are independent of this project. Annotated reference is released labeled text, not complete native PDF text. No template/family-disjoint claim or generated-answer evaluation. English MiniLM is a baseline on Indonesian receipts, not a multilingual quality guarantee. Query timing includes dense encoding; OCR and indexing are separate costs. Canonical-only recovery can conflate distinct values and is never exact correctness. No new independent human adjudication was performed. Method development after inspecting test diagnostics is exploratory, not untouched confirmatory evaluation."}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"summary": summaries, "paired": comparisons, "excluded": dict(excluded)}, indent=2))


if __name__ == "__main__":
    main()
