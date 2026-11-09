"""Freeze image/OCR-only receipt groups before fitting or final label scoring.

Official validation/test receipts are exposure sentinels. Every connected
component touching one is excluded from the fresh study. Grouping is a
conservative heuristic, not an independently audited merchant identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import time
import threading

from receipt_extraction import load_tesseract_tsv

SALT = "noisy-rag-cord-actual-ocr-groups-v2"
ROLES = ((55, "train"), (65, "development"), (75, "calibration_fit"),
         (85, "calibration_select"), (100, "final"))


def unique_image_members(members):
    """Keep one deterministic record per image, preserving publisher aliases."""
    by_hash = {}
    for member in members:
        by_hash.setdefault(member["image_sha256"], []).append(member)
    result = []
    for digest, aliases in sorted(by_hash.items()):
        if len({(m["group"], m["role"], m["ocr_sha256"]) for m in aliases}) != 1:
            raise ValueError("duplicate image crosses group/role/OCR identities")
        aliases = sorted(aliases, key=lambda m: (m.get("source", "cord"), str(m["document_id"]), m["publisher_split"]))
        member = dict(aliases[0])
        if len(aliases) > 1:
            member["image_aliases"] = [{key: alias[key] for key in
                ["document_id", "publisher_split", "source", "source_file"] if key in alias}
                for alias in aliases]
        result.append(member)
    return result


def image_signature(data: bytes, tsv: str) -> dict:
    from PIL import Image
    image = Image.open(BytesIO(data)).convert("L")
    small = list(image.resize((9, 8)).tobytes())
    dhash = sum((small[y * 9 + x] > small[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8) for x in range(8))
    words, _, _ = load_tesseract_tsv(tsv)
    lines = {}
    for word in words:
        if word.top < image.height * .25:
            lines.setdefault(word.line_id, []).append(word)
    header = []
    for line in sorted(lines.values(), key=lambda ws: min(w.top for w in ws))[:3]:
        text = " ".join(w.text for w in sorted(line, key=lambda w: w.left))
        normalized = re.sub(r"[^a-z]", "", text.lower())
        if len(normalized) >= 5:
            header.append(normalized)
    return {"dhash": f"{dhash:016x}", "header": header,
            "width": image.width, "height": image.height}


def freeze_groups(records: list[dict], *, balanced_roles=False) -> dict:
    """Union duplicate images, near duplicates and matching merchant headers."""
    records = sorted(records, key=lambda r: (r["image_sha256"], r["document_id"]))
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)

    for i, left in enumerate(records):
        for j in range(i):
            right = records[j]
            same_image = left["image_sha256"] == right["image_sha256"]
            a, b = left["signature"], right["signature"]
            same_header = bool(set(a["header"]) & set(b["header"]))
            same_company = bool(left.get("company_group") and left.get("company_group") == right.get("company_group"))
            aspect_close = abs(a["width"] / a["height"] - b["width"] / b["height"]) < .03
            near_image = aspect_close and (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count() <= 5
            if same_image or same_header or same_company or near_image:
                union(i, j)
    components = {}
    for i, record in enumerate(records):
        components.setdefault(root(i), []).append(record)
    members, groups = [], []
    for component in components.values():
        identity = "\n".join(sorted({r["image_sha256"] for r in component}))
        group = hashlib.sha256(identity.encode()).hexdigest()
        exposed = any(r.get("exposed", r["publisher_split"] != "train") for r in component)
        # Unreadable headers cannot establish a fresh template family.
        unresolved = any(not r["signature"]["header"] and not r.get("company_group") for r in component)
        bucket = int(hashlib.sha256((SALT + group).encode()).hexdigest(), 16) % 100
        role = "excluded_exposed" if exposed else ("excluded_unresolved" if unresolved else
                next(name for upper, name in ROLES if bucket < upper))
        groups.append({"group": group, "role": role, "documents": len(component)})
        for record in component:
            members.append({**record, "group": group, "role": role})
    if balanced_roles:
        available = sorted((g for g in groups if not g["role"].startswith("excluded_")),
                           key=lambda g: hashlib.sha256((SALT + ":v3:" + g["group"]).encode()).hexdigest())
        if len(available) < 270:
            raise ValueError(f"only {len(available)} fresh groups; need >=270 for 100 risk-validation, 60 final, 40 confidence-fit, 20 development and >=50 training groups")
        for i, group in enumerate(available):
            group["role"] = ("calibration_select" if i < 100 else "final" if i < 160 else
                             "calibration_fit" if i < 200 else "development" if i < 220 else "train")
        lookup = {g["group"]: g["role"] for g in groups}
        for member in members:
            member["role"] = lookup[member["group"]]
    members = unique_image_members(members)
    unique_counts = Counter(member["group"] for member in members)
    for group in groups:
        group["documents"] = unique_counts[group["group"]]
    return {"schema_version": 3 if balanced_roles else 2, "salt": SALT, "roles": ROLES if not balanced_roles else
            {"calibration_select_groups": 100, "final_groups": 60, "calibration_fit_groups": 40, "development_groups": 20, "train_groups": "remaining, at least 50"},
            "group_rule": "connected components: identical image, equal normalized top-quarter header line, or aspect delta<.03 and 64-bit dHash distance<=5; components touching historical validation/test or unreadable headers excluded",
            "grouping_limit": "Heuristic merchant/template isolation, not human-verified identity; undetected merchant aliases remain possible. No distribution-shift guarantee.",
            "groups": sorted(groups, key=lambda g: g["group"]),
            "members": sorted(members, key=lambda r: (r["image_sha256"], r["document_id"])),
            "counts": dict(Counter(r["role"] for r in members)),
            "group_counts": dict(Counter(g["role"] for g in groups))}


def prepare(parquets: list[Path], out: Path, tessdata: Path, *, workers=2):
    import pyarrow.parquet as pq
    from receipt_model_study import verify_cord, sha256
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("manifest already frozen; use a new output directory")
    ocr = out / "ocr"
    ocr.mkdir(exist_ok=True)
    version = subprocess.check_output(["tesseract", "--version"], stderr=subprocess.STDOUT, text=True).splitlines()[0]
    inputs = []
    started = time.time()

    def process(record):
        gt = json.loads(record["ground_truth"])
        split = gt["meta"]["split"]
        split = "validation" if split == "valid" else split
        data = record["image"]["bytes"]
        digest = hashlib.sha256(data).hexdigest()
        path = ocr / (digest + ".tsv")
        if not path.exists():
            result = subprocess.run(["tesseract", "stdin", "stdout", "--tessdata-dir", str(tessdata),
                                     "-l", "eng+ind", "--psm", "6", "-c", "tessedit_create_tsv=1"], input=data,
                                    capture_output=True, timeout=30, check=True)
            load_tesseract_tsv(result.stdout.decode())
            temporary = path.with_suffix(f".{threading.get_ident()}.partial")
            temporary.write_bytes(result.stdout)
            temporary.replace(path)
        text = path.read_text()
        return {"document_id": str(gt["meta"]["image_id"]), "publisher_split": split,
                "image_sha256": digest, "ocr_sha256": sha256(path),
                "signature": image_signature(data, text)}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path in parquets:
            split = "train" if path.name.startswith("train-") else "validation" if path.name.startswith("validation-") else "test"
            verify_cord(path, split)
            inputs.append({"file": path.name, "sha256": sha256(path)})
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=["image", "ground_truth"]):
                records.extend(pool.map(process, batch.to_pylist()))
            print(json.dumps({"prepared": path.name, "documents": len(records)}), flush=True)
    manifest = freeze_groups(records)
    manifest.update({"prepared_epoch": time.time(), "seconds": time.time() - started,
                     "inputs": inputs, "tesseract": version, "psm": 6, "languages": "eng+ind",
                     "tessdata": {p.name: sha256(p) for p in sorted(tessdata.glob("*.traineddata"))},
                     "gold_values_inspected": False})
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (out / "manifest.json").write_bytes(encoded)
    (out / "manifest.sha256").write_text(hashlib.sha256(encoded).hexdigest() + "\n")
    print(json.dumps({"counts": manifest["counts"], "group_counts": manifest["group_counts"]}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tessdata", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    prepare(args.parquet, args.out, args.tessdata, workers=args.workers)
