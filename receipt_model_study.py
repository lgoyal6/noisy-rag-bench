"""Pinned CORD model-study runner for LayoutLMv3 and Qwen2.5-VL.

This runner never writes or promotes a model implicitly. The CORD validation
split was previously inspected and all results made by this script on it must
be labelled exploratory. The current device has no CUDA runtime, so GPU modes
are prepared but cannot be represented as executed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Iterator

from receipt_extraction import parse_amount, load_tesseract_tsv, amount_values_equal

CORD_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
CORD_VALIDATION_SHA256 = "0d0f6dac11fdcc549de2746aa9f53136a3bc22a2a1aff2b0b847f7622ad60c15"
CORD_TEST_SHA256 = "51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7"
CORD_TRAIN_SHA256 = {
    "train-00000-of-00004-b4aaeceff1d90ecb.parquet": "da3994eee1bf9bd3c57f0d53a72c3a6812c8696c5ba26245987949ddf73483cc",
    "train-00001-of-00004-7dbbe248962764c5.parquet": "cce4def16a0d6a6c75f80be712f7494c56c318a8829b712f5c62650155c9e58e",
    "train-00002-of-00004-688fe1305a55e5cc.parquet": "591e2db8fe8b1d364b054f46e8c375b7f00e72578914ba46a573b6858162cab2",
    "train-00003-of-00004-2d0cd200555ed7fd.parquet": "1ffd9de8d6fbcee7630fd4cdfedff05b9b7fabc0fae4fc17557c5fe7cf178748",
}
_VERIFIED_FILES = set()
MODEL_LAYOUTLM = "microsoft/layoutlmv3-base"
LAYOUTLM_REVISION = "cfbbbff0762e6aab37086fdd4739ad14fe7d5db4"
MODEL_QWEN = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
FIELD_TO_CATEGORY = {"total": "total.total_price",
                     "subtotal": "sub_total.subtotal_price",
                     "tax": "sub_total.tax_price"}
LABELS = ["O", "B-TOTAL", "I-TOTAL", "B-SUBTOTAL", "I-SUBTOTAL", "B-TAX", "I-TAX"]
LABEL_ID = {name: i for i, name in enumerate(LABELS)}
FIELD_LABEL = {"total": ("B-TOTAL", "I-TOTAL"),
               "subtotal": ("B-SUBTOTAL", "I-SUBTOTAL"),
               "tax": ("B-TAX", "I-TAX")}


@dataclass
class Sample:
    image: object
    words: list[str]
    boxes: list[list[int]]
    tags: list[int]
    gold: dict[str, str | None]
    document_id: str
    image_sha256: str = ""
    gold_boxes: dict = dataclass_field(default_factory=dict)
    input_supported: bool = True
    ocr_words: list = dataclass_field(default_factory=list)
    invalid_source_boxes: bool = False
    input_truncated: bool = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_cord(path: Path, split: str) -> None:
    expected = {"validation": CORD_VALIDATION_SHA256, "test": CORD_TEST_SHA256}.get(split)
    if split == "train":
        expected = CORD_TRAIN_SHA256.get(path.name)
        if expected is None:
            raise ValueError(f"{path.name} is not one of the four pinned CORD train shards")
    if expected:
        st = path.stat()
        identity = (str(path.resolve()), st.st_size, st.st_mtime_ns, expected)
        if identity not in _VERIFIED_FILES:
            if sha256(path) != expected:
                raise ValueError(f"{path} does not match pinned CORD {split} parquet")
            _VERIFIED_FILES.add(identity)


def _quad_box(word: dict, width: int, height: int) -> list[int] | None:
    quad = word.get("quad", {})
    try:
        xs = [float(quad[f"x{i}"]) for i in range(1, 5)]
        ys = [float(quad[f"y{i}"]) for i in range(1, 5)]
    except (KeyError, TypeError, ValueError):
        return None
    x0, x1 = max(0, min(xs)), min(width, max(xs))
    y0, y1 = max(0, min(ys)), min(height, max(ys))
    if x1 <= x0 or y1 <= y0 or width <= 0 or height <= 0:
        return None
    return [round(x0 * 1000 / width), round(y0 * 1000 / height),
            round(x1 * 1000 / width), round(y1 * 1000 / height)]


def _gold_value(gt: dict, field: str) -> str | None:
    group, key = FIELD_TO_CATEGORY[field].split(".")
    section = gt.get("gt_parse", {}).get(group, {})
    value = section.get(key) if isinstance(section, dict) else None
    amount = parse_amount(value) if isinstance(value, str) else None
    if amount is None and isinstance(value, str):
        for line in gt.get("valid_line", []):
            if line.get("category") == FIELD_TO_CATEGORY[field]:
                context = " ".join(str(w.get("text", "")) for w in line.get("words", []))
                if any(code.lower() in context.lower() for code in ("Rp", "IDR")):
                    amount = parse_amount("Rp " + value)
                    break
    return amount.value if amount else None


def oracle_sample_from_record(record: dict) -> tuple[Sample, dict]:
    """Publisher OCR diagnostic only, never a primary model input."""
    from PIL import Image
    from io import BytesIO
    gt = json.loads(record["ground_truth"])
    data = record["image"]["bytes"]
    image = Image.open(BytesIO(data)).convert("RGB")
    width, height = image.size
    words, boxes, tags = [], [], []
    for line in gt.get("valid_line", []):
        category = line.get("category", "")
        field = next((f for f, c in FIELD_TO_CATEGORY.items() if c == category), None)
        line_words = line.get("words", [])
        for index, word in enumerate(line_words):
            box = _quad_box(word, width, height)
            text = str(word.get("text", "")).strip()
            if box is None or not text:
                continue
            words.append(text)
            boxes.append(box)
            if field is None:
                tags.append(LABEL_ID["O"])
            else:
                begin, inside = FIELD_LABEL[field]
                tags.append(LABEL_ID[begin if index == 0 else inside])
    sample = Sample(image, words, boxes, tags,
                    {field: _gold_value(gt, field) for field in FIELD_TO_CATEGORY},
                    str(gt["meta"]["image_id"]))
    return sample, gt


def sample_from_record(record: dict, *, ocr_tsv: str, supervised: bool = False) -> tuple[Sample, dict]:
    """Actual OCR inputs; only training labels may align to publisher boxes.

    Changing gold annotations cannot change evaluation words, boxes, or tags.
    Gold values travel separately to the scorer and are never processor inputs.
    """
    from PIL import Image
    from io import BytesIO
    image = Image.open(BytesIO(record["image"]["bytes"])).convert("RGB")
    gt = json.loads(record["ground_truth"])
    from receipt_model_worker import MAX_MODEL_WORDS
    raw_observed, _, _ = load_tesseract_tsv(ocr_tsv)
    valid = [w for w in raw_observed if w.valid_box(image.width, image.height)]
    observed = valid[:MAX_MODEL_WORDS]
    words = [w.text for w in observed]
    boxes = [[round(w.left * 1000 / image.width), round(w.top * 1000 / image.height),
              round((w.left + w.width) * 1000 / image.width),
              round((w.top + w.height) * 1000 / image.height)] for w in observed]
    tags = [LABEL_ID["O"]] * len(words)
    if supervised:
        annotations = []
        for line in gt.get("valid_line", []):
            field = next((f for f, category in FIELD_TO_CATEGORY.items()
                          if category == line.get("category")), None)
            if field:
                annotations += [(field, _quad_box(w, image.width, image.height))
                                for w in line.get("words", [])]
        previous = None
        for i, box in enumerate(boxes):
            matches = [(field, box_iou(box, target)) for field, target in annotations if target]
            field, overlap = max(matches, key=lambda x: x[1], default=(None, 0))
            if overlap >= .5:
                tags[i] = LABEL_ID[FIELD_LABEL[field][int(previous == field)]]
                previous = field
            else:
                previous = None
    # An OCR failure stays in the evaluation denominator instead of vanishing.
    if not words:
        words, boxes, tags = ["__EMPTY_OCR__"], [[0, 0, 1, 1]], [LABEL_ID["O"]]
    gold_boxes = {}
    for field, category in FIELD_TO_CATEGORY.items():
        gold_boxes[field] = [box for line in gt.get("valid_line", []) if line.get("category") == category
                             for word in line.get("words", [])
                             if (box := _quad_box(word, image.width, image.height)) is not None]
    return Sample(image, words, boxes, tags,
                  {field: _gold_value(gt, field) for field in gt.get("meta", {}).get("annotated_fields", FIELD_TO_CATEGORY)},
                  str(gt["meta"]["image_id"]),
                  hashlib.sha256(record["image"]["bytes"]).hexdigest(), gold_boxes,
                  image.width * image.height <= 12_000_000, observed,
                  len(valid) != len(raw_observed), len(valid) > MAX_MODEL_WORDS), gt


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def iter_cord_samples(parquet_paths: list[Path], *, split: str,
                      batch_size: int = 2, ocr_dir: Path,
                      members: dict | None = None, role: str | None = None,
                      supervised: bool = False) -> Iterator[tuple[Sample, dict]]:
    import pyarrow.parquet as pq
    seen = set()
    for path in parquet_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        verify_cord(path, split)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size,
                                         columns=["image", "ground_truth"]):
            for record in batch.to_pylist():
                image_hash = hashlib.sha256(record["image"]["bytes"]).hexdigest()
                if image_hash in seen:
                    continue
                if members is not None:
                    member = members.get(image_hash)
                    if member is None or member["role"] != role:
                        continue
                    identity = str(json.loads(record["ground_truth"])["meta"]["image_id"])
                    if identity != str(member["document_id"]):
                        continue  # Preserve duplicate publisher aliases, evaluate canonical image once.
                    expected_ocr = member["ocr_sha256"]
                else:
                    expected_ocr = None
                ocr_path = ocr_dir / (image_hash + ".tsv")
                if expected_ocr is not None and sha256(ocr_path) != expected_ocr:
                    raise ValueError("OCR input changed after the split freeze")
                sample, gt = sample_from_record(record, ocr_tsv=ocr_path.read_text(),
                                                supervised=supervised)
                seen.add(image_hash)
                actual = "validation" if gt["meta"]["split"] == "valid" else gt["meta"]["split"]
                if actual != split:
                    raise ValueError(f"publisher metadata says {actual}, expected {split}")
                yield sample, gt


def encode_sample(processor, sample: Sample, max_length: int = 512):
    if not sample.input_supported:
        raise ValueError("image exceeds the supported 12 megapixel API limit")
    enc = processor(sample.image, sample.words, boxes=sample.boxes,
                    truncation=True, padding="max_length", max_length=max_length,
                    return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    labels, seen = [], set()
    for word_id in word_ids:
        if word_id is None or word_id in seen or word_id >= len(sample.tags):
            labels.append(-100)
        else:
            labels.append(sample.tags[word_id])
            seen.add(word_id)
    enc["labels"] = __import__("torch").tensor([labels], dtype=__import__("torch").long)
    return enc, word_ids


def preflight(protocol_path: Path, *, require_cuda: bool) -> dict:
    import importlib.metadata as md
    versions = {}
    for package in ["torch", "transformers", "peft", "accelerate", "pyarrow", "Pillow"]:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = None
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        devices = [{"name": torch.cuda.get_device_name(i),
                    "memory_total_bytes": torch.cuda.get_device_properties(i).total_memory}
                   for i in range(torch.cuda.device_count())]
        cuda_version = torch.version.cuda
    except ImportError:
        cuda_available, devices, cuda_version = False, [], None
    result = {"protocol": str(protocol_path), "package_versions": versions,
              "cuda_available": cuda_available, "cuda_version": cuda_version,
              "devices": devices, "layoutlm_revision": LAYOUTLM_REVISION,
              "qwen_revision": QWEN_REVISION, "cord_revision": CORD_REVISION}
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required for this mode; no CUDA device is available")
    return result


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _collate(items, torch):
    keys = [k for k in items[0].keys() if k in {"input_ids", "attention_mask", "bbox", "pixel_values", "labels"}]
    return {key: torch.cat([item[key] for item in items], dim=0) for key in keys}


def study_manifest(args):
    if not args.split_manifest or not args.ocr_dir:
        raise ValueError("primary study requires --split-manifest and --ocr-dir")
    path = args.split_manifest
    if sha256(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("split manifest hash mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") not in {2, 3}:
        raise ValueError("actual-OCR grouped split version 2 or 3 required")
    group_roles = {}
    for member in manifest["members"]:
        group, role = member["group"], member["role"]
        if group in group_roles and group_roles[group] != role:
            raise ValueError("template group crosses split roles")
        group_roles[group] = role
        if not role.startswith("excluded_") and (member.get("exposed", False) or
                (member.get("source", "cord") == "cord" and member["publisher_split"] != "train")):
            raise ValueError("previously exposed publisher split appears in fresh roles")
    members = {row["image_sha256"]: row for row in manifest["members"]}
    if len(members) != len(manifest["members"]):
        raise ValueError("frozen manifest contains duplicate image identities")
    if dict(Counter(member["role"] for member in members.values())) != manifest["counts"]:
        raise ValueError("frozen manifest role counts do not match members")
    if not all(manifest["counts"].get(role, 0) for role in
               ["train", "development", "calibration_fit", "calibration_select", "final"]):
        raise ValueError("fresh template grouping left an empty required role; choose another eligible source")
    return members


def study_samples(args, role, *, supervised=False):
    members = study_manifest(args)
    paths = [Path(p) for p in args.train_parquet]
    if {p.name for p in paths} != set(CORD_TRAIN_SHA256):
        raise ValueError("all four pinned train shards required for the grouped study")
    yield from iter_cord_samples(paths, split="train", ocr_dir=args.ocr_dir,
                                members=members, role=role, supervised=supervised)
    if args.sroie_parquet:
        from receipt_sroie import records, as_cord
        seen = set()
        for record in records([Path(p) for p in args.sroie_parquet]):
            digest = hashlib.sha256(record["image"]["bytes"]).hexdigest()
            if digest in seen:
                continue
            member = members.get(digest)
            if member is None or member["role"] != role:
                continue
            if "sroie/" + record["key"] != member["document_id"]:
                continue
            path = args.ocr_dir / (digest + ".tsv")
            if sha256(path) != member["ocr_sha256"]:
                raise ValueError("SROIE OCR changed after freeze")
            converted = as_cord(record)
            sample, gt = sample_from_record(converted, ocr_tsv=path.read_text(), supervised=supervised)
            seen.add(digest)
            if supervised and gt["meta"]["source_label_ambiguous"]:
                continue  # No unambiguous token target; retained in evaluation.
            yield sample, gt


def validate_denominator(rows, members, role):
    expected = {key: value for key, value in members.items() if value["role"] == role}
    seen = set()
    for row in rows:
        key = row["image_sha256"]
        if key not in expected or key in seen:
            raise ValueError("unexpected or duplicate image in evaluation denominator")
        seen.add(key)
        member = expected[key]
        names = [field["field"] for field in row["fields"]]
        required = {"total"} if member.get("source", "cord") == "sroie" else set(FIELD_TO_CATEGORY)
        if len(names) != len(required) or set(names) != required:
            raise ValueError("missing, duplicate or unsupported field in evaluation denominator")
        if str(row["document_id"]) != str(member["document_id"]):
            raise ValueError("document identity differs from frozen evaluation manifest")
    if seen != set(expected):
        raise ValueError("missing images in evaluation denominator")


def annotate_rows(args, rows, model_identity, *, preflight=False):
    members = study_manifest(args)
    manifest = json.loads(args.split_manifest.read_text())
    ocr_identity = manifest.get("ocr_identity", {key: manifest.get(key) for key in
                                                ["tesseract", "psm", "languages", "tessdata"]})
    if not preflight:
        validate_denominator(rows, members, args.eval_role)
    for row in rows:
        member = members[row["image_sha256"]]
        row.update(group=member["group"], role=member["role"], model_identity=model_identity,
                   ocr_identity=ocr_identity,
                   split_manifest_sha256=sha256(args.split_manifest))
    if args.eval_role == "final":
        if not args.calibration:
            raise ValueError("final evaluation requires a separately frozen --calibration policy")
        from receipt_calibration import risk_curve, final_intervals
        policy = json.loads(args.calibration.read_text())
        if sha256(args.calibration) != args.calibration.with_suffix(".sha256").read_text().strip():
            raise ValueError("calibration policy changed after freeze")
        if policy.get("model_identity") != model_identity or policy.get("split_manifest_sha256") != sha256(args.split_manifest):
            raise ValueError("policy belongs to another model or split")
        if policy.get("ocr_identity") != ocr_identity:
            raise ValueError("policy OCR identity differs from evaluation inputs")
        if policy.get("autoaccept_fields") != ["total"]:
            raise ValueError("policy must freeze the supported total-only autoaccept scope")
        if {row["group"] for row in rows} & set(policy["fit_groups"] + policy["select_groups"]):
            raise ValueError("final and calibration groups overlap")
        scope = policy["autoaccept_fields"]
        report = risk_curve(rows, policy["isotonic"], thresholds=[policy["threshold"]], autoaccept_fields=scope)
        return {"frozen_policy_sha256": sha256(args.calibration), "frozen_policy_result": report[0],
                "autoaccept_fields": scope,
                "frozen_policy_intervals": final_intervals(rows, policy["isotonic"], policy["threshold"], autoaccept_fields=scope),
                "risk_coverage_curve": risk_curve(rows, policy["isotonic"], autoaccept_fields=scope),
                "interpretation": "Final curve is descriptive only; policy is not retuned."}
    return {}


def checkpoint_identity(path):
    files = sorted(p for folder in [path / "checkpoint", path / "processor"]
                   for p in folder.rglob("*") if p.is_file())
    if not files:
        raise ValueError("checkpoint and processor files missing")
    return hashlib.sha256("\n".join(str(p.relative_to(path)) + ":" + sha256(p) for p in files).encode()).hexdigest()


def scored_prediction_record(sample: Sample, predictions, *, source: str, truncated=False) -> dict:
    """Use the production adapter and original OCR pixels before gold scoring."""
    from receipt_model_worker import fields_from_predictions
    policy = {"isotonic": [], "threshold": 1.000001,
              "autoaccept_fields": ["total"], "model_identity": source}
    results = fields_from_predictions(sample.ocr_words, predictions,
        width=sample.image.width, height=sample.image.height, policy=policy,
        policy_hash="candidate-only", input_truncated=truncated or sample.input_truncated,
        invalid_source_boxes=sample.invalid_source_boxes)
    fields = []
    for name, gold in sample.gold.items():
        result = results[name]
        box = next(([b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"]]
                    for b in result["source_boxes"]), None)
        normalized = ([value * 1000 / size for value, size in zip(box,
            [sample.image.width, sample.image.height, sample.image.width, sample.image.height])] if box else None)
        support = result["source_supported"]
        fields.append({"field": name, "gold": gold, "value": result["value"],
            "box": box, "original_text": result["original_text"],
            "confidence": result["candidate_score"], "currency": result["currency"],
            "source_supported": support, "input_truncated": truncated or sample.input_truncated,
            "invalid_source_boxes": sample.invalid_source_boxes,
            "source_iou": max((box_iou(normalized, target) for target in sample.gold_boxes.get(name, [])), default=0) if box else 0,
            "candidate_count": result["candidate_count"],
            "unique_candidate_count": result["unique_candidate_count"],
            "exact": amount_values_equal(result["value"], gold),
            "auto_accept_at_0_8": bool(support and result["candidate_score"] >= .8),
            "source": source})
    return {"document_id": sample.document_id, "image_sha256": sample.image_sha256, "fields": fields}


def _prediction_record(sample: Sample, logits, word_ids, boxes, *, source: str) -> dict:
    from receipt_model_worker import _label_confidence
    predictions = _label_confidence(logits, word_ids)
    truncated = len(predictions) < len(sample.words)
    return scored_prediction_record(sample, predictions, source=source, truncated=truncated)


def invalid_input_record(sample):
    return {"document_id": sample.document_id, "image_sha256": sample.image_sha256,
            "fields": [{"field": name, "gold": gold, "value": None, "box": None,
                        "confidence": None, "source_supported": False, "source_iou": 0,
                        "exact": False, "auto_accept_at_0_8": False,
                        "status": "INVALID", "reason": "over_12_megapixels"}
                       for name, gold in sample.gold.items()]}


def observed_currency(words):
    from receipt_extraction import _currency_for
    return _currency_for(" ".join(words))


def run_reference(args):
    """Matched geometry and historical fixed-band retrieval field baselines."""
    from receipt_extraction import extract_fields
    from real_document_eval import tsv_passages, QUESTIONS
    from retrieval import BM25
    rows = []
    identity = hashlib.sha256((Path(__file__).read_text() + args.mode).encode()).hexdigest()
    if args.eval_role == "final":
        annotate_rows(args, [], identity, preflight=True)
    for sample, _ in study_samples(args, args.eval_role):
        if not sample.input_supported:
            rows.append(invalid_input_record(sample))
            continue
        path = args.ocr_dir / (sample.image_sha256 + ".tsv")
        words, _, _ = load_tesseract_tsv(path.read_text())
        start = time.perf_counter()
        if args.mode == "eval-geometry":
            prediction = extract_fields(words, image_width=sample.image.width,
                                        image_height=sample.image.height, source_hash=sample.image_sha256)
            fields = []
            for name, gold in sample.gold.items():
                result = prediction["fields"][name]
                boxes = [[b["left"] * 1000 / sample.image.width, b["top"] * 1000 / sample.image.height,
                          (b["left"] + b["width"]) * 1000 / sample.image.width,
                          (b["top"] + b["height"]) * 1000 / sample.image.height] for b in result["source_boxes"]]
                fields.append({"field": name, "gold": gold, "value": result["value"],
                               "currency": result["currency"], "box": result["source_boxes"],
                               "original_text": result["original_text"],
                               "confidence": result.get("candidate_score") or 0,
                               "source_supported": result["status"] == "ACCEPT",
                               "source_iou": max((box_iou(a, b) for a in boxes for b in sample.gold_boxes.get(name, [])), default=0),
                               "exact": amount_values_equal(result["value"], gold),
                               "unique_candidate_count": 1 if result["value"] is not None else 0,
                               "auto_accept_at_0_8": result["status"] == "ACCEPT"})
        else:
            texts, _ = tsv_passages(path.read_text(), sample.image.height, 12)
            retriever = BM25(texts)
            fields = []
            currency = observed_currency(sample.words)
            for name, gold in sample.gold.items():
                scores = retriever.scores(QUESTIONS[FIELD_TO_CATEGORY[name]])
                # The old retrieval evaluator does not output a field. This
                # frozen adaptation accepts only one exact amount in its top
                # band and never chooses a candidate using the answer.
                best = int(scores.argmax())
                candidates = []
                if scores[best] > 0:
                    for word in words:
                        band = min(11, int((word.top + word.height / 2) * 12 / sample.image.height))
                        amount = parse_amount(((currency + " ") if currency else "") + word.text)
                        if band == best and amount:
                            candidates.append((word, amount))
                selected = candidates[0] if len(candidates) == 1 else None
                box = ([selected[0].left, selected[0].top, selected[0].left + selected[0].width,
                        selected[0].top + selected[0].height] if selected else None)
                normalized = ([x * 1000 / n for x, n in zip(box, [sample.image.width, sample.image.height,
                               sample.image.width, sample.image.height])] if box else None)
                fields.append({"field": name, "gold": gold, "value": selected[1].value if selected else None,
                               "box": box, "currency": currency, "confidence": 1.0 if selected else 0,
                               "source_supported": bool(selected and currency),
                               "source_iou": max((box_iou(normalized, b) for b in sample.gold_boxes.get(name, [])), default=0) if selected else 0,
                               "exact": bool(selected and amount_values_equal(selected[1].value, gold)),
                               "unique_candidate_count": len(candidates), "auto_accept_at_0_8": bool(selected and currency)})
        rows.append({"document_id": sample.document_id, "image_sha256": sample.image_sha256,
                     "fields": fields, "assignment_seconds": time.perf_counter() - start})
    calibration_report = annotate_rows(args, rows, identity)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(args.output_dir / "predictions.jsonl", rows)
    metrics = {**summarize_predictions(rows), **calibration_report,
               "model_identity": identity, "role": args.eval_role,
               "split_manifest_sha256": sha256(args.split_manifest),
               "baseline": args.mode, "timing_scope": "assignment/retrieval only; cached OCR preparation separately measured"}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def run_layoutlm(args, protocol):
    import torch
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    processor = LayoutLMv3Processor.from_pretrained(MODEL_LAYOUTLM,
        revision=LAYOUTLM_REVISION, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_LAYOUTLM,
        revision=LAYOUTLM_REVISION, num_labels=len(LABELS),
        id2label={i: x for i, x in enumerate(LABELS)},
        label2id=LABEL_ID, ignore_mismatched_sizes=True)
    if args.method == "lora" and args.mode == "train-layoutlm":
        cfg = LoraConfig(task_type=TaskType.TOKEN_CLS, r=8, lora_alpha=16,
                         lora_dropout=0.05, target_modules=["query", "value"],
                         modules_to_save=["classifier"])
        model = get_peft_model(model, cfg)
    model.to("cuda")
    if args.mode == "eval-layoutlm":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for evaluation")
        checkpoint_path = args.checkpoint / "checkpoint"
        args.method = "lora" if (checkpoint_path / "adapter_config.json").exists() else "full"
        if (checkpoint_path / "adapter_config.json").exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, checkpoint_path).cuda()
        else:
            model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint_path).cuda()
        started, losses = time.perf_counter(), []
        model_identity = checkpoint_identity(args.checkpoint)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        train_paths = [Path(p) for p in args.train_parquet]
        if len(train_paths) != 4:
            raise ValueError("supply all four pinned official CORD train shards")
        for path in train_paths:
            verify_cord(path, "train")
        started = time.perf_counter()
        model.train()
        losses = []
        for epoch in range(args.epochs):
            pending = []
            for sample, _ in study_samples(args, "train", supervised=True):
                if not sample.input_supported or not sample.ocr_words:
                    continue
                encoded, _ = encode_sample(processor, sample)
                pending.append(encoded)
                if len(pending) == args.batch_size:
                    batch = {k: v.to("cuda") for k, v in _collate(pending, torch).items()}
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(**batch).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    pending = []
            if pending:
                batch = {k: v.to("cuda") for k, v in _collate(pending, torch).items()}
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(args.output_dir / "checkpoint")
        processor.save_pretrained(args.output_dir / "processor")
        model_identity = checkpoint_identity(args.output_dir)
    if args.eval_role == "final":
        annotate_rows(args, [], model_identity, preflight=True)  # Fail before opening final labels.
    model.eval()
    training_seconds = time.perf_counter() - started if args.mode == "train-layoutlm" else 0
    torch.cuda.reset_peak_memory_stats()
    rows = []
    with torch.inference_mode():
        for sample, _ in study_samples(args, args.eval_role):
            if not sample.input_supported:
                rows.append(invalid_input_record(sample))
                continue
            if not sample.ocr_words:
                rows.append(scored_prediction_record(sample, {}, source="empty-ocr"))
                continue
            t0 = time.perf_counter()
            enc, word_ids = encode_sample(processor, sample)
            model_inputs = {k: v.to("cuda") for k, v in enc.items() if k != "labels"}
            torch.cuda.synchronize()
            encoded_at = time.perf_counter()
            logits = model(**model_inputs).logits
            torch.cuda.synchronize()
            inferred_at = time.perf_counter()
            prediction = _prediction_record(sample, logits, word_ids, sample.boxes,
                                              source=f"layoutlmv3-{args.method}")
            prediction["timing"] = {"encoding_and_h2d_seconds": encoded_at - t0,
                                    "model_seconds": inferred_at - encoded_at,
                                    "assignment_and_d2h_seconds": time.perf_counter() - inferred_at}
            rows.append(prediction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_report = annotate_rows(args, rows, model_identity)
    _write_jsonl(args.output_dir / "predictions.jsonl", rows)
    metrics = {**summarize_predictions(rows), **calibration_report}
    metrics.update({"training_epochs": args.epochs, "method": args.method,
                    "steps": len(losses), "mean_training_loss": sum(losses) / max(1, len(losses)),
                    "training_seconds": training_seconds,
                    "total_seconds": time.perf_counter() - started,
                    "peak_device_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "timing_scope": "synchronized batch1 perdocument; actualOCR preparation separatelymeasured; notserviceendtoend",
                    "status": "grouped_actual_ocr_" + args.eval_role,
                    "model_identity": model_identity, "split_manifest_sha256": sha256(args.split_manifest)})
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def summarize_predictions(rows) -> dict:
    fields = [f for row in rows for f in row["fields"] if f["gold"] is not None]
    accepted = [f for f in fields if f["auto_accept_at_0_8"]]
    docs = defaultdict(list)
    for row in rows:
        relevant = [f for f in row["fields"] if f["gold"] is not None]
        if relevant:
            docs[row["document_id"]] = [float(f["exact"]) for f in relevant]
    return {"documents": len(rows), "eligible_fields": len(fields),
            "exact_fields": sum(f["exact"] for f in fields),
            "exact_rate_all_fields": sum(f["exact"] for f in fields) / len(fields) if fields else None,
            "accepted_fields_at_frozen_0_8": len(accepted),
            "coverage_at_frozen_0_8": len(accepted) / len(fields) if fields else None,
            "accepted_error_rate_at_frozen_0_8": (1 - sum(f["exact"] for f in accepted) / len(accepted)
                if accepted else None),
            "accepted_threshold": 0.8,
            "confidence_is_uncalibrated": True,
            "uncertainty": "cluster document ids; use cluster bootstrap only for exploratory interval"}


def qwen_output_counts(rows):
    """Count invalid inputs separately from attempted but malformed generation."""
    flat = [field for row in rows for field in row["fields"] if field["gold"] is not None]
    exact = sum(field["exact"] for field in flat)
    return {"documents": len(rows), "eligible_fields": len(flat), "exact_fields": exact,
            "exact_rate": exact / len(flat) if flat else None,
            "validated_box_fields": sum(bool(field.get("box_valid")) for field in flat),
            "invalid_input_documents": sum(any(field.get("status") == "INVALID" for field in row["fields"]) for row in rows),
            "invalid_json_documents": sum(any(field.get("json_valid") is False for field in row["fields"]) for row in rows)}


def run_qwen(args):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    processor = AutoProcessor.from_pretrained(MODEL_QWEN, revision=QWEN_REVISION)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_QWEN, revision=QWEN_REVISION, torch_dtype=torch.bfloat16,
        device_map="cuda")
    model.eval()
    prompt = ("Read this receipt. Return JSON only with keys total, subtotal, tax. "
              "Each value must be {text: original visible text, box: [x0,y0,x1,y1]} "
              "using pixel coordinates in the supplied image, or null when unsupported. "
              "Never calculate or guess a value.")
    model_identity = hashlib.sha256(json.dumps({"model": MODEL_QWEN, "revision": QWEN_REVISION,
        "prompt": prompt, "max_new_tokens": 256, "do_sample": False}, sort_keys=True).encode()).hexdigest()
    if args.eval_role == "final":
        annotate_rows(args, [], model_identity, preflight=True)
    rows = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for sample, _ in study_samples(args, args.eval_role):
        if not sample.input_supported:
            rows.append(invalid_input_record(sample))
            continue
        t0 = time.perf_counter()
        messages = [{"role": "user", "content": [{"type": "image", "image": sample.image},
                     {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[sample.image], return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        encoded_at = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize()
        inferred_at = time.perf_counter()
        response = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        try:
            parsed = json.loads(response)
            valid_json = isinstance(parsed, dict)
        except json.JSONDecodeError:
            parsed, valid_json = None, False
        fields = []
        for name in sample.gold:
            raw = parsed.get(name) if valid_json else None
            raw_text = raw.get("text") if isinstance(raw, dict) else None
            currency = observed_currency(sample.words)
            amount = parse_amount(((currency + " ") if currency else "") + raw_text) if isinstance(raw_text, str) else None
            box = raw.get("box") if isinstance(raw, dict) else None
            valid_box = (isinstance(box, list) and len(box) == 4 and
                         all(isinstance(x, (int, float)) for x in box) and
                         0 <= box[0] < box[2] <= sample.image.width and
                         0 <= box[1] < box[3] <= sample.image.height)
            gold = sample.gold[name]
            normalized_box = [x * 1000 / n for x, n in zip(box, [sample.image.width, sample.image.height,
                                sample.image.width, sample.image.height])] if valid_box else None
            support = []
            for word, observed_box in zip(sample.words, sample.boxes):
                observed_amount = parse_amount(((currency + " ") if currency else "") + word)
                if amount and observed_amount and amount_values_equal(amount.value, observed_amount.value) and valid_box and box_iou(normalized_box, observed_box) >= .3:
                    support.append(word)
            fields.append({"field": name, "gold": gold,
                           "value": amount.value if amount else None,
                           "box": box if valid_box else None,
                           "json_valid": valid_json,
                           "box_valid": valid_box,
                           "source_supported": bool(support and currency and not sample.invalid_source_boxes
                                                    and not sample.input_truncated),
                           "currency": currency, "original_text": raw_text,
                           "confidence": 1.0 if support else 0.0,
                           "confidence_kind": "binary OCR source-support indicator, not VLM probability; separately calibrated",
                           "unique_candidate_count": 1 if support else 0,
                           "source_iou": max((box_iou(normalized_box, target) for target in sample.gold_boxes.get(name, [])), default=0) if valid_box else 0,
                           "exact": bool(amount and amount_values_equal(amount.value, gold)),
                           "status": "candidate" if amount and valid_box else "review"})
        rows.append({"document_id": sample.document_id, "image_sha256": sample.image_sha256, "raw_response": response,
                     "timing": {"encoding_and_h2d_seconds": encoded_at - t0,
                                "model_generation_seconds": inferred_at - encoded_at,
                                "parsing_and_grounding_seconds": time.perf_counter() - inferred_at},
                     "fields": fields})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    calibration_report = annotate_rows(args, rows, model_identity)
    _write_jsonl(args.output_dir / "predictions.jsonl", rows)
    metrics = {**calibration_report, **qwen_output_counts(rows),
               "seconds": time.perf_counter() - started,
               "peak_device_allocated_bytes": torch.cuda.max_memory_allocated(),
               "status": "grouped_actual_ocr_" + args.eval_role,
               "model_identity": model_identity, "split_manifest_sha256": sha256(args.split_manifest)}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def export_layoutlm_onnx(args):
    """Export a trained LayoutLM study checkpoint and attempt TensorRT parsing/build.

    The output explicitly distinguishes an ONNX file, an engine build, and
    numerical parity. A successful engine build alone is not a verified speedup.
    """
    import torch
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    root = args.checkpoint
    if not root.exists():
        raise FileNotFoundError(root)
    processor_path = root / "processor"
    checkpoint_path = root / "checkpoint"
    processor = LayoutLMv3Processor.from_pretrained(processor_path, apply_ocr=False)
    base = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_LAYOUTLM,
        revision=LAYOUTLM_REVISION, num_labels=len(LABELS),
        id2label={i: x for i, x in enumerate(LABELS)}, label2id=LABEL_ID,
        ignore_mismatched_sizes=True)
    if (checkpoint_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, checkpoint_path)
    else:
        model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint_path)
    model.eval().cuda()
    sample, _ = next(iter(study_samples(args, "development")))
    enc, _ = encode_sample(processor, sample)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    onnx_path = args.output_dir / "layoutlmv3_token_classifier.onnx"

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, input_ids, bbox, attention_mask, pixel_values):
            return self.inner(input_ids=input_ids, bbox=bbox,
                              attention_mask=attention_mask,
                              pixel_values=pixel_values).logits

    wrapper = Wrapper(model).eval()
    inputs = tuple(enc[k].to("cuda") for k in ("input_ids", "bbox", "attention_mask", "pixel_values"))
    result = {"checkpoint": str(root), "onnx_path": str(onnx_path),
              "validation_image_id": sample.document_id,
              "opset": 17, "tensorrt_engine_built": False,
              "parity_tested": False, "speedup_claim": False}
    try:
        torch.onnx.export(wrapper, inputs, str(onnx_path), opset_version=17,
            input_names=["input_ids", "bbox", "attention_mask", "pixel_values"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "bbox": {0: "batch", 1: "sequence"},
                          "attention_mask": {0: "batch", 1: "sequence"},
                          "pixel_values": {0: "batch"}, "logits": {0: "batch", 1: "sequence"}})
        result["onnx_sha256"] = sha256(onnx_path)
    except Exception as exc:
        result.update({"status": "onnx_export_failed", "error": repr(exc)[:2000]})
        (args.output_dir / "export-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise
    if not args.calibration:
        raise ValueError("export study requires the frozen checkpoint-bound --calibration policy")
    from receipt_model_worker import load_frozen_policy
    from receipt_tensorrt import compare_export
    manifest = json.loads(args.split_manifest.read_text())
    policy, policy_hash = load_frozen_policy(args.calibration, checkpoint_identity(root), manifest["ocr_identity"])
    study = compare_export(onnx_path, wrapper, processor,
                           (sample for sample, _ in study_samples(args, "development")),
                           args.output_dir, policy)
    result.update(status=study["status"], parity_tested="parity_pass" in study,
                  parity_pass=study.get("parity_pass", False),
                  tensorrt_engine_built=study.get("engine_built", False),
                  calibration_policy_sha256=policy_hash)
    (args.output_dir / "export-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def smoke_model_api(args):
    """Exercise the actual configured GPU backend through loopback HTTP."""
    import io
    import threading
    from receipt_gpu_worker import BoundedHTTPServer, make_handler
    from receipt_model_worker import BoundedModelWorker, LocalLayoutLMBackend, RemoteLayoutLMBackend
    if args.eval_role != "development" or not args.checkpoint or not args.calibration:
        raise ValueError("API smoke requires a checkpoint, frozen policy and development role")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    identity_path = args.output_dir / "upstream-ocr-identity.json"
    identity_path.write_text(json.dumps(json.loads(args.split_manifest.read_text())["ocr_identity"]))
    backend = LocalLayoutLMBackend(args.checkpoint, args.calibration, ocr_identity_file=identity_path)
    worker = BoundedModelWorker(backend, max_pending=1, request_timeout=45)
    server = BoundedHTTPServer(("127.0.0.1", 0), make_handler(worker,
        model_identity=backend.model_identity, policy_hash=backend.policy_hash,
        ocr_identity=backend.ocr_identity))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    records = []
    try:
        client = RemoteLayoutLMBackend(f"http://127.0.0.1:{server.server_port}",
            expected_ocr_identity=backend.ocr_identity,
            expected_model_identity=backend.model_identity, expected_policy_hash=backend.policy_hash)
        for sample, _ in study_samples(args, "development"):
            if not sample.input_supported or not sample.ocr_words or sample.invalid_source_boxes or sample.input_truncated:
                continue
            buffer = io.BytesIO()
            sample.image.save(buffer, format="PNG")
            data = buffer.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            reference = backend.predict(sample.image, sample.ocr_words, source_hash=digest)
            start = time.perf_counter()
            actual = client.predict(sample.image, sample.ocr_words, source_hash=digest, source_bytes=data)
            elapsed = time.perf_counter() - start
            keys = ["value", "currency", "source_boxes", "status", "outcome", "unique_candidate_count"]
            same = all({key: actual[name][key] for key in keys} ==
                       {key: reference[name][key] for key in keys} for name in FIELD_LABEL)
            records.append({"document_id": sample.document_id, "original_image_sha256": sample.image_sha256,
                "smoke_png_sha256": digest, "http_seconds": elapsed,
                "native_and_http_task_equal": same, "fields": actual})
            if len(records) == 3:
                break
    finally:
        server.shutdown()
        server.server_close()
        worker.close()
        thread.join(timeout=5)
    result = {"status": "verified" if len(records) == 3 and all(r["native_and_http_task_equal"] for r in records) else "failed",
        "model_identity": backend.model_identity, "calibration_policy_sha256": backend.policy_hash,
        "ocr_identity": backend.ocr_identity, "samples": records,
        "scope": "Real checkpoint inference, HTTP and operator-pinned client; cached Mac OCR; reencoded identical RGB public development pixels. This is not live OCR or end-to-end interactive latency."}
    (args.output_dir / "api-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "verified":
        raise RuntimeError("GPU model API smoke did not satisfy task parity")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["preflight", "train-layoutlm", "eval-layoutlm", "eval-qwen", "eval-geometry", "eval-band", "export-onnx", "smoke-api"], required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--validation-parquet", default="", help="deprecated; primary study uses frozen roles from train shards")
    ap.add_argument("--split-manifest", type=Path)
    ap.add_argument("--ocr-dir", type=Path)
    ap.add_argument("--eval-role", choices=["development", "calibration_fit", "calibration_select", "final"], default="development")
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--train-parquet", action="append", default=[])
    ap.add_argument("--sroie-parquet", action="append", default=[])
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--method", choices=["full", "lora"], default="lora")
    ap.add_argument("--seed", type=int, default=6406)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    args = ap.parse_args()
    report = preflight(args.protocol, require_cuda=args.mode not in {"preflight", "eval-geometry", "eval-band"})
    if args.mode == "preflight":
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if not args.output_dir:
        raise SystemExit("--output-dir is required")
    if args.output_dir.exists():
        raise SystemExit("output directory already exists; choose a new path")
    study_manifest(args)
    if args.eval_role == "final" and (not args.calibration or not args.calibration.is_file()):
        raise SystemExit("final inference requires a separately frozen calibration policy")
    if args.mode == "train-layoutlm" and args.eval_role != "development":
        raise SystemExit("fit first using development only, freeze policy, then evaluate checkpoint")
    if args.mode in {"eval-geometry", "eval-band"}:
        print(json.dumps(run_reference(args), indent=2, sort_keys=True))
    elif args.mode == "eval-qwen":
        print(json.dumps(run_qwen(args), indent=2, sort_keys=True))
    elif args.mode in {"train-layoutlm", "eval-layoutlm"}:
        if args.epochs < 1 or args.epochs > 10 or args.batch_size not in range(1, 9):
            raise SystemExit("epochs must be 1-10 and batch size 1-8")
        if args.learning_rate <= 0 or args.learning_rate > 1e-3:
            raise SystemExit("learning rate outside conservative bound")
        print(json.dumps(run_layoutlm(args, report), indent=2, sort_keys=True))
    elif args.mode == "export-onnx":
        if not args.checkpoint:
            raise SystemExit("--checkpoint must name a trained LayoutLM study directory")
        export_layoutlm_onnx(args)
    elif args.mode == "smoke-api":
        print(json.dumps(smoke_model_api(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
