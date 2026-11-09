"""Bounded local LayoutLMv3 extraction worker.

The production backend is loaded only from administrator-configured local files.
The HTTP request supplies pixels and OCR evidence, never a model or policy path.
"""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import queue
import base64
import io
import ipaddress
import subprocess
import threading
from urllib.error import URLError, HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from receipt_extraction import OCRWord, _currency_for, amount_values_equal, parse_amount

LABELS = ["O", "B-TOTAL", "I-TOTAL", "B-SUBTOTAL", "I-SUBTOTAL", "B-TAX", "I-TAX"]
FIELD_LABEL = {"total": ("B-TOTAL", "I-TOTAL"), "subtotal": ("B-SUBTOTAL", "I-SUBTOTAL"),
               "tax": ("B-TAX", "I-TAX")}
MAX_MODEL_WORDS = 2048


class WorkerBusy(RuntimeError):
    pass


class WorkerTimeout(TimeoutError):
    pass


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_identity(root: Path) -> str:
    files = sorted(p for folder in (root / "checkpoint", root / "processor")
                   for p in folder.rglob("*") if p.is_file())
    if not files:
        raise ValueError("configured checkpoint and processor files are missing")
    material = "\n".join(f"{p.relative_to(root)}:{file_sha256(p)}" for p in files)
    return hashlib.sha256(material.encode()).hexdigest()


def ocr_identity(*, tesseract: str, tessdata_dir: str | Path, languages: str) -> dict:
    if languages != "eng+ind":
        raise ValueError("model service OCR must use calibrated languages eng+ind")
    tessdata = Path(tessdata_dir)
    hashes = {}
    for language in languages.split("+"):
        file = tessdata / f"{language}.traineddata"
        if not file.is_file():
            raise ValueError(f"configured OCR language data is missing: {language}")
        hashes[file.name] = file_sha256(file)
    try:
        version = subprocess.check_output([tesseract, "--version"], stderr=subprocess.STDOUT,
                                          text=True, timeout=5).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        raise ValueError("could not identify configured Tesseract executable") from exc
    return {"tesseract": version, "psm": 6, "languages": languages,
            "tessdata": hashes}


def load_frozen_policy(path: Path, expected_model_identity: str,
                       expected_ocr_identity: dict) -> tuple[dict, str]:
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != digest:
        raise ValueError("frozen calibration policy hash mismatch")
    policy = json.loads(encoded)
    if policy.get("version") != 1 or policy.get("model_identity") != expected_model_identity:
        raise ValueError("calibration policy belongs to another model")
    manifest_hash = policy.get("split_manifest_sha256")
    if (not isinstance(manifest_hash, str) or len(manifest_hash) != 64 or
            any(ch not in "0123456789abcdef" for ch in manifest_hash)):
        raise ValueError("calibration policy is not bound to a frozen split manifest")
    if policy.get("ocr_identity") != expected_ocr_identity:
        raise ValueError("calibration policy OCR identity does not match configured Tesseract")
    if policy.get("autoaccept_fields") != ["total"]:
        raise ValueError("calibration policy must freeze total-only autoaccept scope")
    blocks = policy.get("isotonic")
    threshold = policy.get("threshold")
    if not isinstance(blocks, list) or not isinstance(threshold, (int, float)):
        raise ValueError("invalid frozen calibration policy")
    if not 0 <= float(threshold) <= 1.000001:
        raise ValueError("invalid frozen threshold")
    prev = float("-inf")
    for block in blocks:
        score, probability, count = (block.get("max_score"), block.get("probability"), block.get("count"))
        if (not isinstance(score, (int, float)) or score < prev or
                not isinstance(probability, (int, float)) or not 0 <= probability <= 1 or
                not isinstance(count, int) or count < 1):
            raise ValueError("invalid isotonic calibration blocks")
        prev = score
    return policy, digest


def calibrated_probability(blocks: list[dict], score: float | None) -> float:
    if score is None or not blocks:
        return 0.0
    for block in blocks:
        if score <= block["max_score"]:
            return float(block["probability"])
    return float(blocks[-1]["probability"])


def _label_confidence(logits, word_ids):
    import torch
    probabilities = torch.softmax(logits[0], dim=-1)
    out = {}
    for i, word_id in enumerate(word_ids):
        if word_id is None or word_id in out:
            continue
        label_id = int(torch.argmax(probabilities[i]).item())
        out[word_id] = (LABELS[label_id], float(probabilities[i, label_id].item()))
    return out


def fields_from_predictions(words: list[OCRWord], word_predictions: dict[int, tuple[str, float]],
                            *, width: int, height: int, policy: dict, policy_hash: str,
                            input_truncated=False, invalid_source_boxes=False) -> dict:
    currency = _currency_for(" ".join(word.text for word in words))
    fields = {}
    blocks, threshold = policy["isotonic"], float(policy["threshold"])
    for name, allowed in FIELD_LABEL.items():
        candidates = []
        invalid_candidate = False
        for word_id, (label, confidence) in word_predictions.items():
            if label not in allowed or not 0 <= word_id < len(words):
                continue
            word = words[word_id]
            amount = parse_amount(((currency + " ") if currency else "") + word.text)
            if amount is None:
                continue
            if not word.valid_box(width, height):
                invalid_candidate = True
                continue
            candidates.append((word_id, word, amount, confidence))
        unique = []
        for candidate in candidates:
            value = candidate[2].value
            if not any(amount_values_equal(value, prior) for prior in unique):
                unique.append(value)
        selected = max(candidates, key=lambda x: x[3]) if candidates else None
        reasons = []
        in_scope = name in policy.get("autoaccept_fields", [])
        if not in_scope:
            reasons.append("field_outside_frozen_autoaccept_scope")
        if input_truncated:
            reasons.append("ocr_words_truncated_at_model_limit")
        if invalid_source_boxes:
            reasons.append("invalid_ocr_boxes_present")
        if not selected:
            reasons.append("model_candidate_missing")
            if invalid_candidate:
                reasons.append("model_candidate_box_invalid")
        if len(unique) > 1:
            reasons.append("multiple_model_candidates")
        if not currency:
            reasons.append("explicit_currency_missing_or_ambiguous")
        confidence = selected[3] if selected else None
        probability = calibrated_probability(blocks, confidence)
        if selected and probability < threshold:
            reasons.append("below_frozen_calibration_threshold")
        source_supported = bool(selected and len(unique) == 1 and currency and not
                                (invalid_candidate or invalid_source_boxes or input_truncated))
        status = ("ACCEPT" if in_scope and source_supported and
                  probability >= threshold and not invalid_candidate else "REVIEW")
        if selected:
            _, word, amount, _ = selected
            value, currency_value = amount.value, amount.currency or currency
            boxes = [{"left": word.left, "top": word.top, "width": word.width, "height": word.height}]
            original = word.text
        else:
            value = currency_value = original = None
            boxes = []
        outcome = ("ACCEPT" if status == "ACCEPT" else
                   "INVALID" if invalid_candidate else "MISSING" if not candidates else "REVIEW")
        fields[name] = {"value": value, "currency": currency_value, "original_text": original,
                        "source_boxes": boxes, "page": selected[1].page if selected else None,
                        "candidate_score": confidence, "calibrated_probability": probability,
                        "candidate_count": len(candidates), "unique_candidate_count": len(unique),
                        "source_supported": source_supported,
                        "status": status, "outcome": outcome,
                        "reason_codes": sorted(set(reasons)),
                        "source_hash": None, "model_revision": policy["model_identity"],
                        "calibration_policy_sha256": policy_hash}
    return fields


class LocalLayoutLMBackend:
    """Real backend. All paths and weights are fixed before serving requests."""
    def __init__(self, checkpoint_root: str | Path, policy_path: str | Path, *,
                 tessdata_dir: str | Path | None = None,
                 ocr_identity_file: str | Path | None = None,
                 tesseract: str = "tesseract",
                 languages: str = "eng+ind", device: str = "cuda"):
        root, policy_path = Path(checkpoint_root), Path(policy_path)
        identity = checkpoint_identity(root)
        if (tessdata_dir is None) == (ocr_identity_file is None):
            raise ValueError("configure exactly one OCR identity source")
        if ocr_identity_file is not None:
            configured_ocr = json.loads(Path(ocr_identity_file).read_text())
            if not isinstance(configured_ocr, dict):
                raise ValueError("configured OCR identity must be a JSON object")
        else:
            configured_ocr = ocr_identity(tesseract=tesseract, tessdata_dir=tessdata_dir,
                                          languages=languages)
        policy, policy_hash = load_frozen_policy(policy_path, identity, configured_ocr)
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; local model worker was not started")
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
        from receipt_model_study import LAYOUTLM_REVISION, MODEL_LAYOUTLM
        processor_path, checkpoint_path = root / "processor", root / "checkpoint"
        self.processor = LayoutLMv3Processor.from_pretrained(processor_path, apply_ocr=False,
                                                               local_files_only=True)
        base = LayoutLMv3ForTokenClassification.from_pretrained(
            MODEL_LAYOUTLM, revision=LAYOUTLM_REVISION, num_labels=len(LABELS),
            id2label={i: label for i, label in enumerate(LABELS)},
            label2id={label: i for i, label in enumerate(LABELS)},
            ignore_mismatched_sizes=True, local_files_only=True)
        if (checkpoint_path / "adapter_config.json").exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, checkpoint_path, local_files_only=True)
        else:
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                checkpoint_path, local_files_only=True)
        self.device = device
        self.model.eval().to(device)
        self.policy, self.policy_hash, self.model_identity = policy, policy_hash, identity
        self.ocr_identity = configured_ocr

    def predict(self, image, words: list[OCRWord], *, source_hash: str):
        import torch
        width, height = image.size
        if not words:
            fields = fields_from_predictions([], {}, width=width, height=height,
                                             policy=self.policy, policy_hash=self.policy_hash)
            for field in fields.values():
                field["source_hash"] = source_hash
            return fields
        valid_words = [word for word in words if word.valid_box(width, height)]
        normalized_boxes = [[round(w.left * 1000 / width), round(w.top * 1000 / height),
                             round((w.left + w.width) * 1000 / width),
                             round((w.top + w.height) * 1000 / height)] for w in valid_words]
        input_words = valid_words[:MAX_MODEL_WORDS]
        input_boxes = normalized_boxes[:MAX_MODEL_WORDS]
        encoded = self.processor(image, [w.text for w in input_words], boxes=input_boxes,
                                 truncation=True, padding="max_length", max_length=512,
                                 return_tensors="pt")
        word_ids = encoded.word_ids(batch_index=0)
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        predictions = _label_confidence(logits, word_ids)
        represented = {word_id for word_id in word_ids if word_id is not None}
        truncated = len(input_words) < len(valid_words) or len(represented) < len(input_words)
        invalid_boxes = len(valid_words) != len(words)
        fields = fields_from_predictions(input_words, predictions, width=width, height=height,
                                         policy=self.policy, policy_hash=self.policy_hash,
                                         input_truncated=truncated, invalid_source_boxes=invalid_boxes)
        for field in fields.values():
            field["source_hash"] = source_hash
        return fields


def _loopback_url(url: str) -> str:
    parsed = urlsplit(url)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
    except ValueError as exc:
        raise ValueError("GPU worker URL must use a numeric loopback address") from exc
    if (parsed.scheme != "http" or not address.is_loopback or parsed.username or parsed.password or
            parsed.query or parsed.fragment or parsed.path not in ("", "/") or
            parsed.port is None or not 1 <= parsed.port <= 65535):
        raise ValueError("GPU worker URL must be a plain HTTP loopback host with an explicit port")
    host = f"[{address}]" if address.version == 6 else str(address)
    return f"http://{host}:{parsed.port}"


class RemoteLayoutLMBackend:
    """Mac-side client for a GPU worker exposed only through an SSH loopback forward."""
    def __init__(self, url: str, *, expected_ocr_identity: dict,
                 expected_model_identity: str | None = None,
                 expected_policy_hash: str | None = None, timeout: float = 45.0):
        if not 0 < timeout <= 180:
            raise ValueError("remote model timeout must be in (0, 180] seconds")
        self.base_url, self.timeout = _loopback_url(url), timeout
        if not expected_model_identity or not expected_policy_hash:
            raise ValueError("remote model and policy identities must be pinned by the operator")
        self.expected_ocr_identity = expected_ocr_identity
        health = self._request("GET", "/health", None, timeout=min(timeout, 5))
        if health.get("ocr_identity") != expected_ocr_identity:
            raise ValueError("GPU worker OCR identity does not match local calibration")
        if (health.get("model_identity") != expected_model_identity or
                health.get("calibration_policy_sha256") != expected_policy_hash):
            raise ValueError("GPU worker model or policy identity differs from operator pins")
        self.model_identity = health["model_identity"]
        self.policy_hash = health["calibration_policy_sha256"]

    def _request(self, method: str, path: str, payload: dict | None, *, timeout: float):
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        request = Request(self.base_url + path, data=data, method=method,
                          headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"GPU worker returned HTTP {response.status}")
                raw = response.read(2 * 1024 * 1024 + 1)
        except (URLError, TimeoutError, HTTPError) as exc:
            raise RuntimeError("GPU worker unavailable or timed out") from exc
        if len(raw) > 2 * 1024 * 1024:
            raise RuntimeError("GPU worker response exceeds 2 MiB")
        try:
            result = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("GPU worker returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise RuntimeError("GPU worker returned invalid response shape")
        return result

    def predict(self, image, words: list[OCRWord], *, source_hash: str,
                source_bytes: bytes | None = None):
        if source_bytes is None or hashlib.sha256(source_bytes).hexdigest() != source_hash:
            raise ValueError("original image bytes are required and must match source hash")
        payload = {"source_hash": source_hash,
                   "image_base64": base64.b64encode(source_bytes).decode("ascii"),
                   "image_sha256": hashlib.sha256(source_bytes).hexdigest(),
                   "width": image.width, "height": image.height,
                   "ocr_identity": self.expected_ocr_identity,
                   "words": [{"text": w.text, "left": w.left, "top": w.top,
                              "width": w.width, "height": w.height, "page": w.page,
                              "line_id": w.line_id, "confidence": w.confidence} for w in words]}
        result = self._request("POST", "/predict", payload, timeout=self.timeout)
        if (result.get("source_hash") != source_hash or
                result.get("model_identity") != self.model_identity or
                result.get("calibration_policy_sha256") != self.policy_hash or
                result.get("ocr_identity") != self.expected_ocr_identity or
                not isinstance(result.get("fields"), dict) or
                set(result["fields"]) != set(FIELD_LABEL)):
            raise RuntimeError("GPU worker response identity or source mismatch")
        for name, field in result["fields"].items():
            if name not in FIELD_LABEL or not isinstance(field, dict):
                raise RuntimeError("GPU worker returned malformed fields")
            if field.get("source_hash") != source_hash or field.get("model_revision") != self.model_identity or field.get("calibration_policy_sha256") != self.policy_hash:
                raise RuntimeError("GPU field provenance mismatch")
        return result["fields"]


@dataclass
class _Job:
    image: object
    words: list[OCRWord]
    source_hash: str
    source_bytes: bytes | None
    future: Future
    cancelled: threading.Event


class BoundedModelWorker:
    """One dedicated worker and a bounded pending queue; timed-out results are discarded."""
    def __init__(self, backend, *, max_pending: int = 1, request_timeout: float = 45.0):
        if max_pending < 1 or request_timeout <= 0:
            raise ValueError("invalid worker bounds")
        self.backend, self.request_timeout = backend, request_timeout
        self.queue: queue.Queue[_Job | None] = queue.Queue(maxsize=max_pending)
        self._closed = False
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="receipt-layoutlm", daemon=True)
        self.thread.start()

    def submit(self, image, words: list[OCRWord], source_hash: str, *, source_bytes: bytes | None = None):
        future: Future = Future()
        job = _Job(image, words, source_hash, source_bytes, future, threading.Event())
        with self._lock:
            if self._closed:
                raise WorkerBusy("model worker is closed")
            try:
                self.queue.put_nowait(job)
            except queue.Full as exc:
                raise WorkerBusy("model worker is busy") from exc
        try:
            return future.result(timeout=self.request_timeout)
        except TimeoutError as exc:
            job.cancelled.set()
            future.cancel()
            raise WorkerTimeout("model request timed out; result discarded") from exc

    def _run(self):
        while True:
            job = self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            try:
                if not job.cancelled.is_set() and job.future.set_running_or_notify_cancel():
                    if isinstance(self.backend, RemoteLayoutLMBackend):
                        result = self.backend.predict(job.image, job.words,
                                                      source_hash=job.source_hash,
                                                      source_bytes=job.source_bytes)
                    else:
                        result = self.backend.predict(job.image, job.words, source_hash=job.source_hash)
                    if not job.cancelled.is_set():
                        job.future.set_result(result)
            except BaseException as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self.queue.task_done()

    def close(self, *, wait: bool = True):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Pending jobs are discarded immediately; running native model calls cannot be preempted.
            while True:
                try:
                    job = self.queue.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    job.cancelled.set()
                    job.future.cancel()
                self.queue.task_done()
            self.queue.put_nowait(None)
        if wait:
            self.thread.join(timeout=self.request_timeout)
