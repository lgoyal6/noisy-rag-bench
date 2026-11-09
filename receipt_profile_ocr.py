"""Profile the exact OCR stage on frozen development images only."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cord-parquet", type=Path, action="append", required=True)
    parser.add_argument("--sroie-parquet", type=Path, action="append", required=True)
    parser.add_argument("--tessdata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("profile output exists")
    from PIL import Image
    import pyarrow.parquet as pq
    from receipt_sroie import records
    from receipt_model_study import sha256, verify_cord
    from receipt_model_worker import ocr_identity
    manifest = json.loads(args.manifest.read_text())
    if sha256(args.manifest) != args.manifest.with_suffix(".sha256").read_text().strip():
        raise ValueError("manifest changed")
    identity = ocr_identity(tesseract="tesseract", tessdata_dir=args.tessdata, languages="eng+ind")
    if identity != manifest["ocr_identity"]:
        raise ValueError("OCR profile identity differs from calibration inputs")
    members = {m["image_sha256"]: m for m in manifest["members"] if m["role"] == "development"}

    def images():
        for path in args.cord_parquet:
            verify_cord(path, "train")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2, columns=["image"]):
                for record in batch.to_pylist():
                    yield record["image"]["bytes"]
        for record in records(args.sroie_parquet):
            yield record["image"]["bytes"]

    rows, warmed = [], False
    for data in images():
        digest = hashlib.sha256(data).hexdigest()
        member = members.get(digest)
        if member is None:
            continue
        with Image.open(BytesIO(data)) as image:
            if image.width * image.height > 12_000_000:
                rows.append({"document_id": member["document_id"], "status": "INVALID", "reason": "over_12_megapixels"})
                continue
            t0 = time.perf_counter()
            png = BytesIO()
            image.convert("RGB").save(png, format="PNG")
            normalized_at = time.perf_counter()
        command = ["tesseract", "stdin", "stdout", "--tessdata-dir", str(args.tessdata),
                   "-l", "eng+ind", "--psm", "6", "-c", "tessedit_create_tsv=1"]
        if not warmed:
            subprocess.run(command, input=png.getvalue(), capture_output=True, timeout=20, check=True)
            warmed = True
        ocr_started = time.perf_counter()
        process = subprocess.run(command, input=png.getvalue(), capture_output=True, timeout=20, check=True)
        rows.append({"document_id": member["document_id"], "group": member["group"], "status": "OK",
                     "normalization_seconds": normalized_at - t0,
                     "ocr_seconds": time.perf_counter() - ocr_started,
                     "ocr_sha256": hashlib.sha256(process.stdout).hexdigest(),
                     "matches_frozen_ocr": hashlib.sha256(process.stdout).hexdigest() == member["ocr_sha256"]})
        if sum(r["status"] == "OK" for r in rows) >= 16:
            break
    args.out.mkdir(parents=True)
    (args.out / "profile.json").write_text(json.dumps({"ocr_identity": identity,
        "selection": "first16supporteddevelopmentimagesinpinnedsourceorder, oneuntimedwarmup",
        "rows": rows, "scope": "CPU original-pixel normalization and actualOCR, no model/assignment/service"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
