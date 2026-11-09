"""Bounded, local-only receipt extraction and append-only reviewer corrections."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from io import BytesIO

from receipt_extraction import extract_fields, load_tesseract_tsv

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 12_000_000
MAX_PAGES = 1
OCR_TIMEOUT_SECONDS = 20


def _ensure_schema(db: sqlite3.Connection) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS predictions (
        source_hash TEXT PRIMARY KEY, result_hash TEXT NOT NULL,
        result_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    db.execute("""CREATE TABLE IF NOT EXISTS corrections (
        correction_id TEXT PRIMARY KEY, source_hash TEXT NOT NULL,
        field TEXT NOT NULL CHECK(field IN ('total','subtotal','tax')),
        value TEXT NOT NULL, currency TEXT, note TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(source_hash) REFERENCES predictions(source_hash))""")
    db.execute("""CREATE TABLE IF NOT EXISTS prediction_versions (
        source_hash TEXT NOT NULL, model_identity TEXT NOT NULL,
        calibration_policy_sha256 TEXT NOT NULL, result_hash TEXT NOT NULL,
        result_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(source_hash, model_identity, calibration_policy_sha256))""")
    db.execute("""CREATE TABLE IF NOT EXISTS correction_versions (
        correction_id TEXT PRIMARY KEY, source_hash TEXT NOT NULL,
        model_identity TEXT NOT NULL, calibration_policy_sha256 TEXT NOT NULL,
        field TEXT NOT NULL CHECK(field IN ('total','subtotal','tax')),
        value TEXT NOT NULL, currency TEXT, note TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(source_hash, model_identity, calibration_policy_sha256)
        REFERENCES prediction_versions(source_hash, model_identity, calibration_policy_sha256))""")


def analyze_image(data: bytes, *, tesseract: str = "tesseract", model_worker=None,
                  tessdata_dir: str | Path | None = None, languages: str = "eng") -> dict:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("image must be nonempty and no larger than 10 MiB")
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        with Image.open(BytesIO(data)) as source:
            if getattr(source, "n_frames", 1) > MAX_PAGES:
                raise ValueError("only one-page images are supported")
            if source.getexif().get(274, 1) != 1:
                raise ValueError("rotated EXIF images are unsupported to preserve source coordinates")
            image = source
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("image exceeds 12 megapixels")
            image.load()
            width, height = image.size
            output = BytesIO()
            image.convert("RGB").save(output, format="PNG")
            normalized = output.getvalue()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid or unsupported image") from exc
    source_hash = hashlib.sha256(data).hexdigest()
    command = [tesseract, "stdin", "stdout"]
    if tessdata_dir is not None:
        command += ["--tessdata-dir", str(tessdata_dir)]
    command += ["-l", languages, "--psm", "6", "-c", "tessedit_create_tsv=1"]
    try:
        proc = subprocess.run(command,
                              input=normalized, capture_output=True,
                              timeout=OCR_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("OCR timed out after 20 seconds") from exc
    if proc.returncode:
        raise RuntimeError("Tesseract failed: " + proc.stderr.decode("utf-8", "replace")[:500])
    tsv = proc.stdout.decode("utf-8", "replace")
    words, _, _ = load_tesseract_tsv(tsv)
    result = extract_fields(words, image_width=width, image_height=height,
                            source_hash=source_hash,
                            model_revision="tesseract-5-psm6-noisy-rag-v1")
    result["document"] = {"source_hash": source_hash, "width": width, "height": height,
                          "original_bytes": len(data), "enhancement": "none"}
    result["ocr"] = {"engine": "tesseract-5-psm6", "words": [
        {"text": w.text, "left": w.left, "top": w.top, "width": w.width,
         "height": w.height, "page": w.page, "line_id": w.line_id,
         "confidence": w.confidence} for w in words]}
    if model_worker is not None:
        from PIL import Image
        with Image.open(BytesIO(data)) as source:
            model_fields = model_worker.submit(source.convert("RGB"), words, source_hash,
                                               source_bytes=data)
        result["fields"] = model_fields
        result["model"] = {"identity": next(iter(model_fields.values()))["model_revision"] if model_fields else None,
                            "calibration_policy_sha256": next(iter(model_fields.values()))["calibration_policy_sha256"] if model_fields else None}
        payload = {key: value for key, value in result.items() if key != "result_hash"}
        result["result_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return result


def persist_prediction(db_path: str | Path, result: dict) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        _ensure_schema(db)
        model = result.get("model")
        if model:
            model_identity = model.get("identity")
            policy_hash = model.get("calibration_policy_sha256")
            if not model_identity or not policy_hash:
                raise ValueError("model predictions require model and calibration identities")
            db.execute("""INSERT OR IGNORE INTO prediction_versions
                (source_hash,model_identity,calibration_policy_sha256,result_hash,result_json)
                VALUES(?,?,?,?,?)""",
                (result["document"]["source_hash"], model_identity, policy_hash,
                 result["result_hash"], json.dumps(result, sort_keys=True)))
            prior = db.execute("""SELECT result_hash FROM prediction_versions
                WHERE source_hash=? AND model_identity=? AND calibration_policy_sha256=?""",
                (result["document"]["source_hash"], model_identity, policy_hash)).fetchone()
            if prior[0] != result["result_hash"]:
                raise ValueError("source already has a different immutable prediction for this model and policy")
            return
        db.execute("INSERT OR IGNORE INTO predictions(source_hash,result_hash,result_json) VALUES(?,?,?)",
                   (result["document"]["source_hash"], result["result_hash"],
                    json.dumps(result, sort_keys=True)))
        old = db.execute("SELECT result_hash FROM predictions WHERE source_hash=?",
                         (result["document"]["source_hash"],)).fetchone()
        if old[0] != result["result_hash"]:
            raise ValueError("source already has a different immutable prediction")


def append_correction(db_path: str | Path, *, correction_id: str, source_hash: str,
                      field: str, value: str, currency: str | None, note: str = "",
                      model_identity: str | None = None,
                      calibration_policy_sha256: str | None = None) -> None:
    if not correction_id or len(correction_id) > 128 or len(value) > 64 or len(note) > 500:
        raise ValueError("correction fields exceed bounds")
    if field not in {"total", "subtotal", "tax"}:
        raise ValueError("unsupported field")
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA foreign_keys=ON")
        _ensure_schema(db)
        if model_identity is not None or calibration_policy_sha256 is not None:
            if not model_identity or not calibration_policy_sha256:
                raise ValueError("model corrections require both model and policy identities")
            try:
                db.execute("""INSERT OR IGNORE INTO correction_versions
                    (correction_id,source_hash,model_identity,calibration_policy_sha256,
                     field,value,currency,note) VALUES(?,?,?,?,?,?,?,?)""",
                    (correction_id, source_hash, model_identity, calibration_policy_sha256,
                     field, value, currency, note))
            except sqlite3.IntegrityError as exc:
                raise ValueError("unknown model prediction version") from exc
            prior = db.execute("""SELECT source_hash,model_identity,calibration_policy_sha256,
                field,value,currency,note FROM correction_versions
                WHERE correction_id=?""", (correction_id,)).fetchone()
            current = (source_hash, model_identity, calibration_policy_sha256,
                       field, value, currency, note)
            if prior != current:
                raise ValueError("correction_id already exists with different content")
            return
        try:
            db.execute("""INSERT OR IGNORE INTO corrections
                (correction_id,source_hash,field,value,currency,note) VALUES(?,?,?,?,?,?)""",
                       (correction_id, source_hash, field, value, currency, note))
        except sqlite3.IntegrityError as exc:
            raise ValueError("unknown prediction source hash") from exc
        prior = db.execute("SELECT source_hash,field,value,currency,note FROM corrections WHERE correction_id=?",
                           (correction_id,)).fetchone()
        current = (source_hash, field, value, currency, note)
        if prior != current:
            raise ValueError("correction_id already exists with different content")
