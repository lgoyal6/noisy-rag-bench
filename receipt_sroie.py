"""Pinned SROIE augmentation, with actual OCR and company-isolated splits.

SROIE labels only total. Publisher words/boxes supervise or score that field;
they never become primary model inputs. Nonunique amount/source labels remain
explicitly ambiguous and cannot establish source correctness.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
from io import BytesIO

from receipt_dataset import freeze_groups, image_signature
from receipt_extraction import parse_amount, load_tesseract_tsv, amount_values_equal

REVISION = "bffe40c26759f3376ec2b3ae9031dbba54cd587c"
FILES = {"train-00000-of-00001.parquet": "b18c16b4d8481e5e4537a1700e4616907fe4acd92d6362a7e430b0e866213887",
         "test-00000-of-00001.parquet": "04f8f31b45944cc6e6459a7a95c851a721fc93ffec0a5c29ece9ded734a684c2"}
EXPOSED_CARD_IDS = {"X00016469669", "X00016469672", "X00016469676"}


def verify(path):
    from receipt_model_study import sha256
    if path.name not in FILES or sha256(path) != FILES[path.name]:
        raise ValueError("SROIE shard does not match pinned source")


def records(paths):
    import pyarrow.parquet as pq
    for path in paths:
        verify(path)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4):
            for record in batch.to_pylist():
                record["_source_file"] = path.name
                record["_publisher_split"] = "test" if path.name.startswith("test-") else "train"
                yield record


def as_cord(record):
    target = parse_amount(record["entities"]["total"])
    matches = []
    for text, box in zip(record["words"], record["bboxes"]):
        amount = parse_amount(text)
        if target and amount and amount_values_equal(amount.value, target.value):
            matches.append((text, box))
    lines = []
    for text, box in zip(record["words"], record["bboxes"]):
        x0, y0, x1, y1 = box
        # Only a unique supported amount has an independently identified region.
        category = "total.total_price" if len(matches) == 1 and (text, box) == matches[0] else "other"
        lines.append({"category": category, "words": [{"text": text, "quad":
            {"x1": x0, "y1": y0, "x2": x1, "y2": y0, "x3": x1, "y3": y1, "x4": x0, "y4": y1}}]})
    gt = {"meta": {"image_id": "sroie/" + record["key"], "split": "train",
                   "annotated_fields": ["total"], "source_label_ambiguous": len(matches) != 1},
          "gt_parse": {"total": {"total_price": record["entities"]["total"]}},
          "valid_line": lines}
    return {"image": record["image"], "ground_truth": json.dumps(gt)}


def prepare(args):
    from receipt_model_study import sha256
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "manifest.json").exists():
        raise ValueError("manifest already frozen")
    cord = json.loads(args.cord_manifest.read_text())
    if sha256(args.cord_manifest) != args.cord_manifest.with_suffix(".sha256").read_text().strip():
        raise ValueError("CORD manifest changed")
    ocr_dir = args.out / "ocr"
    ocr_dir.mkdir(exist_ok=True)
    # Hard links preserve the exact OCR bytes without a second cache copy.
    for member in cord["members"]:
        source = args.cord_ocr_dir / (member["image_sha256"] + ".tsv")
        target = ocr_dir / source.name
        if not target.exists():
            target.hardlink_to(source)
    combined = [{**member, "source": "cord"} for member in cord["members"]]

    def process(record):
        from PIL import Image
        data = record["image"]["bytes"]
        digest = hashlib.sha256(data).hexdigest()
        path = ocr_dir / (digest + ".tsv")
        with Image.open(BytesIO(data)) as image:
            rejected = image.width * image.height > 12_000_000
            normalized = BytesIO()
            if not rejected:
                image.convert("RGB").save(normalized, format="PNG")
        if not path.exists():
            if rejected:
                content = b"level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            else:
                result = subprocess.run(["tesseract", "stdin", "stdout", "--tessdata-dir", str(args.tessdata),
                    "-l", "eng+ind", "--psm", "6", "-c", "tessedit_create_tsv=1"],
                    input=normalized.getvalue(), capture_output=True, timeout=30, check=True)
                content = result.stdout
            load_tesseract_tsv(content.decode())
            partial = path.with_suffix(f".{threading.get_ident()}.partial")
            partial.write_bytes(content)
            partial.replace(path)
        company = re.sub("[^a-z0-9]", "", record["entities"]["company"].lower())
        return {"document_id": "sroie/" + record["key"], "source": "sroie",
                "publisher_split": record["_publisher_split"],
                "source_file": record["_source_file"],
                "exposed": record["key"] in EXPOSED_CARD_IDS,
                "input_rejected": rejected,
                "input_rejection_reason": "over_12_megapixels" if rejected else None,
                "exposure_note": "Dataset-card examples are exposed; other public train/test images have not been inspected by this study. Both splits are repartitioned by merchant before fitting.",
                "image_sha256": digest, "ocr_sha256": sha256(path), "company_group": company,
                "signature": image_signature(data, path.read_text())}

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Bound in-flight image memory instead of executor.map consuming all rows.
        pending = []
        for record in records(args.sroie_parquet):
            pending.append(record)
            if len(pending) == 4:
                combined.extend(pool.map(process, pending))
                pending = []
        if pending:
            combined.extend(pool.map(process, pending))
    manifest = freeze_groups(combined, balanced_roles=True)
    manifest.update({"cord_manifest_sha256": sha256(args.cord_manifest), "sroie_revision": REVISION,
                     "ocr_identity": {key: cord[key] for key in ["tesseract", "psm", "languages", "tessdata"]},
                     "sroie_files": FILES, "license": "CC-BY-4.0; Huang et al., ICDAR2019 SROIE organizers, jsdnrs repackaging",
                     "model_inputs": "actual Tesseract eng+ind PSM6 only; company annotation is grouping metadata, not a model feature",
                     "label_scope": "CORD total/subtotal/tax; SROIE total only. Nonunique SROIE source labels are ambiguous, never silently assigned to the rightmost amount."})
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (args.out / "manifest.json").write_bytes(encoded)
    (args.out / "manifest.sha256").write_text(hashlib.sha256(encoded).hexdigest() + "\n")
    print(json.dumps({"counts": manifest["counts"], "group_counts": manifest["group_counts"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sroie-parquet", type=Path, action="append", required=True)
    parser.add_argument("--cord-manifest", type=Path, required=True)
    parser.add_argument("--cord-ocr-dir", type=Path, required=True)
    parser.add_argument("--tessdata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args())
