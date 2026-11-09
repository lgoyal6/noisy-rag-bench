"""Loopback-only HTTP endpoint for the configured local LayoutLMv3 GPU worker."""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import io
import json
from pathlib import Path
import threading
import time

from PIL import Image

from receipt_extraction import OCRWord
from receipt_model_worker import (BoundedModelWorker, LocalLayoutLMBackend,
                                  MAX_MODEL_WORDS, WorkerBusy, WorkerTimeout)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_WORD_TEXT = 256
BODY_TIMEOUT_SECONDS = 10.0


def _parse_request(body: bytes, expected_ocr_identity: dict):
    try:
        payload = json.loads(body)
        image_data = base64.b64decode(payload["image_base64"], validate=True)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON request or base64 image") from exc
    if not image_data or len(image_data) > MAX_IMAGE_BYTES:
        raise ValueError("image must be nonempty and no larger than 10 MiB")
    source_hash = payload.get("source_hash")
    image_hash = hashlib.sha256(image_data).hexdigest()
    if (not isinstance(source_hash, str) or source_hash != image_hash or
            payload.get("image_sha256") != image_hash):
        raise ValueError("image source hash mismatch")
    if payload.get("ocr_identity") != expected_ocr_identity:
        raise ValueError("request OCR identity mismatch")
    try:
        with Image.open(io.BytesIO(image_data)) as source:
            if getattr(source, "n_frames", 1) != 1 or source.getexif().get(274, 1) != 1:
                raise ValueError("single-page, unrotated image required")
            if source.width * source.height > 12_000_000:
                raise ValueError("image exceeds 12 megapixels")
            source.load()
            image = source.convert("RGB")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid image") from exc
    if image.width * image.height > 12_000_000:
        raise ValueError("image exceeds 12 megapixels")
    if payload.get("width") != image.width or payload.get("height") != image.height:
        raise ValueError("image dimensions mismatch")
    raw_words = payload.get("words")
    if not isinstance(raw_words, list) or len(raw_words) > MAX_MODEL_WORDS:
        raise ValueError("OCR word list exceeds configured limit")
    words = []
    for item in raw_words:
        if not isinstance(item, dict):
            raise ValueError("invalid OCR word")
        text = item.get("text")
        if not isinstance(text, str) or not text or len(text) > MAX_WORD_TEXT:
            raise ValueError("OCR word text exceeds configured limit")
        values = [item.get(k) for k in ("left", "top", "width", "height", "page")]
        if any(type(value) is not int for value in values):
            raise ValueError("OCR coordinates must be integers")
        confidence = item.get("confidence")
        if confidence is not None and (not isinstance(confidence, (int, float)) or not 0 <= confidence <= 100):
            raise ValueError("invalid OCR confidence")
        line_id = item.get("line_id", "")
        if not isinstance(line_id, str) or len(line_id) > 128:
            raise ValueError("invalid OCR line identifier")
        word = OCRWord(text, *values[:4], page=values[4], line_id=line_id,
                       confidence=float(confidence) if confidence is not None else None)
        if not word.valid_box(image.width, image.height):
            raise ValueError("OCR source box is outside original pixels")
        words.append(word)
    return image, words, source_hash


def make_handler(worker: BoundedModelWorker, *, model_identity: str,
                 policy_hash: str, ocr_identity: dict):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ReceiptGPUWorker/1"

        def _json(self, status: int, value: dict):
            data = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self._json(404, {"error": "not found"})
            return self._json(200, {"model_identity": model_identity,
                                    "calibration_policy_sha256": policy_hash,
                                    "ocr_identity": ocr_identity})

        def do_POST(self):
            if self.path != "/predict":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                return self._json(400, {"error": "invalid content length"})
            if length < 0 or length > 15 * 1024 * 1024:
                return self._json(413, {"error": "request must be at most 15 MiB"})
            deadline = time.monotonic() + BODY_TIMEOUT_SECONDS
            chunks = []
            remaining = length
            try:
                while remaining:
                    wait = deadline - time.monotonic()
                    if wait <= 0:
                        raise TimeoutError("request body deadline exceeded")
                    self.connection.settimeout(wait)
                    chunk = self.rfile.read1(min(65536, remaining))
                    if not chunk:
                        raise ValueError("incomplete request body")
                    chunks.append(chunk)
                    remaining -= len(chunk)
            except TimeoutError:
                return self._json(408, {"error": "request body deadline exceeded"})
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            body = b"".join(chunks)
            try:
                image, words, source_hash = _parse_request(body, ocr_identity)
                fields = worker.submit(image, words, source_hash)
                return self._json(200, {"source_hash": source_hash,
                                        "model_identity": model_identity,
                                        "calibration_policy_sha256": policy_hash,
                                        "ocr_identity": ocr_identity, "fields": fields})
            except WorkerBusy as exc:
                return self._json(503, {"error": str(exc), "retryable": True})
            except WorkerTimeout as exc:
                return self._json(504, {"error": str(exc), "retryable": True})
            except ValueError as exc:
                return self._json(400, {"error": str(exc)[:200]})
            except RuntimeError as exc:
                return self._json(503, {"error": str(exc)[:200]})

        def log_message(self, fmt, *args):
            pass
    return Handler


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 2

    def __init__(self, *args, **kwargs):
        self._slots = threading.BoundedSemaphore(2)
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(10)
        return request, address

    def process_request(self, request, client_address):
        self._slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--ocr-identity", type=Path, required=True,
                        help="operator-installed identity JSON, copied from the calibrated Mac configuration")
    parser.add_argument("--request-timeout", type=float, default=45.0)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 0 < args.request_timeout <= 180:
        raise SystemExit("port or request timeout outside configured bounds")
    if not args.ocr_identity.is_file():
        raise SystemExit("configured OCR identity file does not exist")
    backend = LocalLayoutLMBackend(args.checkpoint, args.calibration,
                                   ocr_identity_file=args.ocr_identity)
    worker = BoundedModelWorker(backend, max_pending=1,
                                request_timeout=args.request_timeout)
    server = BoundedHTTPServer(("127.0.0.1", args.port),
        make_handler(worker, model_identity=backend.model_identity,
                     policy_hash=backend.policy_hash, ocr_identity=backend.ocr_identity))
    print(f"Loopback GPU worker at http://127.0.0.1:{args.port}/health", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        worker.close()


if __name__ == "__main__":
    main()
