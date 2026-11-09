"""Score the conservative extractor against CORD publisher labels.

This runner is for explicitly exploratory scoring on the already-inspected
CORD validation split. It is not a fresh final-split claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from real_document_eval import VALIDATION_SHA256, reference_receipt
from receipt_extraction import extract_fields, load_tesseract_tsv, parse_amount

MAPPING = {"total.total_price": "total", "sub_total.subtotal_price": "subtotal",
           "sub_total.tax_price": "tax"}


def document_bootstrap(rows, seed=6406, draws=2000):
    by_doc = defaultdict(list)
    for row in rows:
        by_doc[row["document_id"]].append(row)
    docs = sorted(by_doc)
    if not docs:
        return {"documents": 0, "mean": None, "ci95": None, "seed": seed, "draws": draws}
    rng = random.Random(seed)
    values = [sum(r["correct"] for r in by_doc[d]) / len(by_doc[d]) for d in docs]
    means = [sum(values[rng.randrange(len(values))] for _ in values) / len(values)
             for _ in range(draws)]
    means.sort()
    return {"documents": len(docs), "mean": sum(values) / len(values),
            "ci95": [means[int(.025 * (draws - 1))], means[int(.975 * (draws - 1))]],
            "seed": seed, "draws": draws}


def score_document(gt, tsv, *, width, height, source_hash, document_id):
    _, questions, exclusions = reference_receipt(gt)
    words, _, _ = load_tesseract_tsv(tsv)
    pred = extract_fields(words, image_width=width, image_height=height,
                          source_hash=source_hash)
    qmap = {q["category"]: q["answer"] for q in questions if q["category"] in MAPPING}
    rows = []
    for category, field in MAPPING.items():
        gold = qmap.get(category)
        if gold is None:
            continue
        target = parse_amount(gold)
        if target is None:
            exclusions["ambiguous_publisher_amount"] = exclusions.get("ambiguous_publisher_amount", 0) + 1
            continue
        result = pred["fields"][field]
        exact = bool(target and result["value"] is not None and result["value"] == target.value)
        rows.append({"document_id": document_id, "category": category,
                     "status": result["status"], "accepted": result["status"] == "ACCEPT",
                     "gold_value": target.value if target else None,
                     "predicted_value": result["value"], "correct": exact,
                     "reason_codes": result["reason_codes"]})
    return rows, exclusions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--tsv-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--exploratory", action="store_true",
                    help="required acknowledgement that validation was inspected previously")
    args = ap.parse_args()
    if not args.exploratory:
        raise SystemExit("only exploratory validation scoring is supported; pass --exploratory")
    if args.output.exists():
        raise SystemExit("output exists; choose a new output path")
    data = args.parquet.read_bytes()
    dataset_hash = hashlib.sha256(data).hexdigest()
    if dataset_hash != VALIDATION_SHA256:
        raise SystemExit("input is not the pinned CORD validation parquet")
    import pyarrow.parquet as pq
    import PIL.Image
    import io
    records = pq.read_table(args.parquet).to_pylist()
    rows, exclusions, documents = [], defaultdict(int), set()
    for record in records:
        gt = json.loads(record["ground_truth"])
        if gt["meta"]["split"] != "valid":
            raise ValueError("publisher split metadata mismatch")
        image_bytes = record["image"]["bytes"]
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        with PIL.Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
        image_id = str(gt["meta"]["image_id"])
        tsv_path = args.tsv_dir / f"cord-validation-{image_id}.tsv"
        if not tsv_path.is_file():
            raise FileNotFoundError(f"missing OCR evidence: {tsv_path}")
        scored, omitted = score_document(gt, tsv_path.read_text(), width=width,
            height=height, source_hash=image_hash, document_id=image_id)
        rows.extend(scored)
        documents.add(image_id)
        for key, count in omitted.items():
            exclusions[key] += count
    accepted = [row for row in rows if row["accepted"]]
    summary = {"eligible_fields": len(rows), "documents_with_eligible_fields": len(documents),
               "accepted_fields": len(accepted),
               "coverage": len(accepted) / len(rows) if rows else None,
               "accepted_exact_numeric": sum(row["correct"] for row in accepted),
               "accepted_error_rate": (1 - sum(row["correct"] for row in accepted) / len(accepted)
                                       if accepted else None),
               "all_field_exact_numeric": sum(row["correct"] for row in rows),
               "all_field_exact_rate": sum(row["correct"] for row in rows) / len(rows) if rows else None,
               "accepted_risk_document_bootstrap": document_bootstrap(accepted),
               "all_fields_document_bootstrap": document_bootstrap(rows),
               "exclusions": dict(exclusions),
               "interpretation": "exploratory only: CORD validation OCR/layout results were inspected previously; not template-held-out or independent human validation"}
    args.output.mkdir(parents=True)
    (args.output / "rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    protocol = {"parquet_sha256": dataset_hash, "split": "validation", "tesseract_data": "pre-existing psm6 raw TSV, eng+ind; prior inspection means exploratory",
                "tsv_files": len(documents), "extraction_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
