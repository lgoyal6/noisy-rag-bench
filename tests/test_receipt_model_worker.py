"""CPU-only worker contract tests using fake backend outputs, not model validation."""
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from receipt_extraction import OCRWord
from receipt_model_worker import (BoundedModelWorker, WorkerBusy, WorkerTimeout,
                                  fields_from_predictions, load_frozen_policy,
                                  ocr_identity, RemoteLayoutLMBackend)


def frozen_policy(tmp_path, model="a" * 64, *, threshold=.5):
    path = tmp_path / "frozen.json"
    ocr = {"tesseract": "tesseract 5.3", "psm": 6, "languages": "eng+ind",
           "tessdata": {"eng.traineddata": "e" * 64, "ind.traineddata": "i" * 64}}
    data = {"version": 1, "model_identity": model, "split_manifest_sha256": "d" * 64,
            "ocr_identity": ocr, "autoaccept_fields": ["total"],
            "isotonic": [{"max_score": 1.0, "probability": 1.0, "count": 10}],
            "threshold": threshold}
    encoded = (json.dumps(data, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    path.with_suffix(".sha256").write_text(hashlib.sha256(encoded).hexdigest() + "\n")
    return path, data


def test_policy_is_hash_and_model_bound(tmp_path):
    path, _ = frozen_policy(tmp_path)
    loaded, digest = load_frozen_policy(path, "a" * 64, loaded_ocr := loaded_ocr_identity(path))
    assert loaded["threshold"] == .5
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="another model"):
        load_frozen_policy(path, "b" * 64, loaded_ocr)
    with pytest.raises(ValueError, match="OCR identity"):
        load_frozen_policy(path, "a" * 64, {**loaded_ocr, "languages": "eng"})
    path.with_suffix(".sha256").write_text("0" * 64)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_policy(path, "a" * 64, loaded_ocr)


def loaded_ocr_identity(path):
    return json.loads(path.read_text())["ocr_identity"]


def test_output_preserves_ocr_box_and_reviews_missing_or_ambiguous_values(tmp_path):
    policy_path, policy = frozen_policy(tmp_path)
    digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    words = [OCRWord("TOTAL", 1, 1, 20, 8, line_id="1"),
             OCRWord("Rp", 30, 1, 10, 8, line_id="1"),
             OCRWord("5000", 45, 1, 20, 8, line_id="1"),
             OCRWord("900", 70, 1, 15, 8, line_id="1")]
    fields = fields_from_predictions(words, {2: ("B-TOTAL", .9), 3: ("B-TOTAL", .8)},
                                     width=100, height=40, policy=policy, policy_hash=digest)
    total = fields["total"]
    assert total["status"] == "REVIEW"
    assert "multiple_model_candidates" in total["reason_codes"]
    assert total["source_boxes"] == [{"left": 45, "top": 1, "width": 20, "height": 8}]
    assert total["calibration_policy_sha256"] == digest
    assert fields["tax"]["status"] == "REVIEW"
    assert fields["tax"]["outcome"] == "MISSING"
    assert "model_candidate_missing" in fields["tax"]["reason_codes"]


def test_invalid_model_candidate_has_invalid_outcome(tmp_path):
    path, policy = frozen_policy(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    words = [OCRWord("Rp", 1, 1, 10, 8, line_id="1"),
             OCRWord("5000", 120, 1, 20, 8, line_id="1")]
    fields = fields_from_predictions(words, {1: ("B-TOTAL", .9)}, width=100, height=40,
                                     policy=policy, policy_hash=digest)
    assert fields["total"]["status"] == "REVIEW"
    assert fields["total"]["outcome"] == "INVALID"


def test_unique_currency_candidate_can_be_accepted_only_by_frozen_policy(tmp_path):
    path, policy = frozen_policy(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    words = [OCRWord("Rp", 1, 1, 10, 8, line_id="1"), OCRWord("5000", 20, 1, 20, 8, line_id="1")]
    fields = fields_from_predictions(words, {1: ("B-TOTAL", .9)}, width=100, height=40,
                                     policy=policy, policy_hash=digest)
    assert fields["total"]["status"] == "ACCEPT"
    assert fields["total"]["value"] == "5000"
    policy["threshold"] = 1.000001  # review-all fail-closed policy
    fields = fields_from_predictions(words, {1: ("B-TOTAL", .9)}, width=100, height=40,
                                     policy=policy, policy_hash=digest)
    assert fields["total"]["status"] == "REVIEW"


def test_subtotal_and_tax_require_review_outside_frozen_scope(tmp_path):
    path, policy = frozen_policy(tmp_path)
    words = [OCRWord("RM12.90", 1, 1, 40, 8)]
    for label, name in [("B-SUBTOTAL", "subtotal"), ("B-TAX", "tax")]:
        fields = fields_from_predictions(words, {0: (label, .99)}, width=100, height=40,
                                         policy=policy, policy_hash="a" * 64)
        assert fields[name]["status"] == "REVIEW"
        assert "field_outside_frozen_autoaccept_scope" in fields[name]["reason_codes"]


def test_duplicate_decimal_spellings_are_one_numeric_candidate(tmp_path):
    path, policy = frozen_policy(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    words = [OCRWord("RM", 1, 1, 10, 8, line_id="1"),
             OCRWord("12.90", 20, 1, 20, 8, line_id="1"),
             OCRWord("12.9", 45, 1, 15, 8, line_id="1")]
    fields = fields_from_predictions(words, {1: ("B-TOTAL", .9), 2: ("I-TOTAL", .8)},
                                     width=100, height=40, policy=policy, policy_hash=digest)
    assert fields["total"]["status"] == "ACCEPT"


def test_ocr_identity_binds_tesseract_language_data_and_segmentation(tmp_path, monkeypatch):
    import receipt_model_worker
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    for language, contents in (("eng", b"english model"), ("ind", b"indonesian model")):
        (tessdata / f"{language}.traineddata").write_bytes(contents)
    monkeypatch.setattr(receipt_model_worker.subprocess, "check_output",
                        lambda *a, **k: "tesseract 5.3.0\nversion details")
    identity = ocr_identity(tesseract="configured-tesseract", tessdata_dir=tessdata,
                            languages="eng+ind")
    assert identity == {"tesseract": "tesseract 5.3.0", "psm": 6,
                        "languages": "eng+ind",
                        "tessdata": {"eng.traineddata": hashlib.sha256(b"english model").hexdigest(),
                                     "ind.traineddata": hashlib.sha256(b"indonesian model").hexdigest()}}


class BlockingBackend:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def predict(self, image, words, *, source_hash):
        self.calls += 1
        self.started.set()
        self.release.wait(2)
        return {"source_hash": source_hash}


def test_worker_bounds_pending_queue_and_discards_cancelled_results():
    backend = BlockingBackend()
    worker = BoundedModelWorker(backend, max_pending=1, request_timeout=.12)
    pool = ThreadPoolExecutor(max_workers=2)
    first = pool.submit(worker.submit, object(), [], "first")
    assert backend.started.wait(1)
    second = pool.submit(worker.submit, object(), [], "second")
    time.sleep(.02)
    with pytest.raises(WorkerBusy):
        worker.submit(object(), [], "third")
    with pytest.raises(WorkerTimeout):
        first.result(timeout=1)
    with pytest.raises(WorkerTimeout):
        second.result(timeout=1)
    backend.release.set()
    worker.close()
    pool.shutdown(wait=True)
    assert backend.calls <= 1  # cancelled queued request never reaches backend


def test_service_model_path_preserves_original_hash_and_ocr_provenance(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    import io
    import subprocess
    import receipt_service

    original = io.BytesIO()
    Image.new("RGB", (32, 20), "white").save(original, format="PNG")
    data = original.getvalue()
    tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tleft\ttop\twidth\theight\tconf\ttext\n"
           "5\t1\t1\t1\t1\t2\t3\t10\t5\t90\tRp\n"
           "5\t1\t1\t1\t1\t15\t3\t12\t5\t88\t5000\n")
    monkeypatch.setattr(receipt_service.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a[0], 0, tsv.encode(), b""))

    class CaptureWorker:
        def submit(self, image, words, source_hash, *, source_bytes=None):
            assert image.size == (32, 20)
            assert [w.text for w in words] == ["Rp", "5000"]
            assert source_bytes == data
            return {"total": {"value": "5000", "status": "REVIEW",
                              "source_hash": source_hash, "model_revision": "fake",
                              "calibration_policy_sha256": "c" * 64}}

    result = receipt_service.analyze_image(data, model_worker=CaptureWorker())
    assert result["document"]["source_hash"] == hashlib.sha256(data).hexdigest()
    assert [w["text"] for w in result["ocr"]["words"]] == ["Rp", "5000"]
    assert result["fields"]["total"]["source_hash"] == result["document"]["source_hash"]
    assert result["result_hash"] == hashlib.sha256(json.dumps(
        {key: value for key, value in result.items() if key != "result_hash"}, sort_keys=True).encode()).hexdigest()


def test_model_prediction_versions_and_review_corrections_are_immutable(tmp_path):
    import sqlite3
    from receipt_service import append_correction, persist_prediction

    db = tmp_path / "versions.sqlite"
    source = "a" * 64
    geometry = {"result_hash": "g" * 64, "document": {"source_hash": source}}
    persist_prediction(db, geometry)
    versions = []
    for identity, policy_hash, result_hash in (("m1", "p1", "r1"), ("m2", "p2", "r2")):
        result = {"result_hash": result_hash, "document": {"source_hash": source},
                  "model": {"identity": identity, "calibration_policy_sha256": policy_hash}}
        persist_prediction(db, result)
        persist_prediction(db, result)  # safe idempotent retry
        versions.append((identity, policy_hash))
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM predictions").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM prediction_versions").fetchone()[0] == 2
    model, policy = versions[0]
    append_correction(db, correction_id="review-1", source_hash=source, field="total",
                      value="15.00", currency="MYR", model_identity=model,
                      calibration_policy_sha256=policy)
    append_correction(db, correction_id="review-1", source_hash=source, field="total",
                      value="15.00", currency="MYR", model_identity=model,
                      calibration_policy_sha256=policy)  # retry stays idempotent
    def save_racing_retry(_):
        append_correction(db, correction_id="review-race", source_hash=source,
                          field="tax", value="1.00", currency="MYR",
                          model_identity=model, calibration_policy_sha256=policy)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(save_racing_retry, range(6)))
    with pytest.raises(ValueError, match="different content"):
        append_correction(db, correction_id="review-1", source_hash=source, field="total",
                          value="16.00", currency="MYR", model_identity=model,
                          calibration_policy_sha256=policy)
    with pytest.raises(ValueError, match="unknown model prediction"):
        append_correction(db, correction_id="unknown", source_hash="b" * 64, field="total",
                          value="1", currency="MYR", model_identity=model,
                          calibration_policy_sha256=policy)
    with pytest.raises(ValueError, match="unknown prediction source"):
        append_correction(db, correction_id="unknown-legacy", source_hash="b" * 64,
                          field="total", value="1", currency="MYR")


def test_remote_backend_uses_only_loopback_and_verifies_worker_identity():
    from http.server import BaseHTTPRequestHandler
    import base64
    import io
    from PIL import Image

    identity = loaded_ocr_identity(frozen_policy(Path(__import__('tempfile').mkdtemp()))[0])
    seen = {}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"model_identity": "m" * 64,
                               "calibration_policy_sha256": "p" * 64,
                               "ocr_identity": identity}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            source = seen["source_hash"]
            fields = {name: {"source_hash": source, "model_revision": "m" * 64,
                            "calibration_policy_sha256": "p" * 64}
                      for name in ["total", "subtotal", "tax"]}
            body = json.dumps({"source_hash": source, "model_identity": "m" * 64,
                               "calibration_policy_sha256": "p" * 64,
                               "ocr_identity": identity, "fields": fields}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def log_message(self, *args): pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        remote = RemoteLayoutLMBackend(f"http://127.0.0.1:{server.server_port}",
                                       expected_ocr_identity=identity,
                                       expected_model_identity="m" * 64, expected_policy_hash="p" * 64)
        image_bytes = io.BytesIO(); Image.new("RGB", (10, 10), "white").save(image_bytes, format="PNG")
        data = image_bytes.getvalue(); source_hash = hashlib.sha256(data).hexdigest()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        result = remote.predict(image, [], source_hash=source_hash, source_bytes=data)
        assert "total" in result
        assert base64.b64decode(seen["image_base64"]) == data
        assert seen["ocr_identity"] == identity
        with pytest.raises(ValueError, match="operator pins"):
            RemoteLayoutLMBackend(endpoint := f"http://127.0.0.1:{server.server_port}",
                expected_ocr_identity=identity, expected_model_identity="wrong", expected_policy_hash="p" * 64)
        remote._request = lambda *a, **k: {"source_hash": source_hash,
            "model_identity": "m" * 64, "calibration_policy_sha256": "p" * 64,
            "ocr_identity": identity, "fields": {}}
        with pytest.raises(RuntimeError, match="identity or source"):
            remote.predict(image, [], source_hash=source_hash, source_bytes=data)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
    with pytest.raises(ValueError, match="loopback"):
        RemoteLayoutLMBackend("http://example.com:8877", expected_ocr_identity=identity)


def test_gpu_worker_http_contract_rejects_bad_source_and_bounds_words():
    from http.server import BaseHTTPRequestHandler
    from PIL import Image
    import base64
    import io
    import receipt_gpu_worker

    identity = {"tesseract": "tesseract 5.3", "psm": 6, "languages": "eng+ind",
                "tessdata": {"eng.traineddata": "e" * 64, "ind.traineddata": "i" * 64}}
    class Worker:
        def submit(self, image, words, source_hash):
            return {"total": {"source_hash": source_hash}}
    server = receipt_gpu_worker.BoundedHTTPServer(("127.0.0.1", 0),
        receipt_gpu_worker.make_handler(Worker(), model_identity="m" * 64,
                                        policy_hash="p" * 64, ocr_identity=identity))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        buf = io.BytesIO(); Image.new("RGB", (20, 20), "white").save(buf, format="PNG")
        data = buf.getvalue(); digest = hashlib.sha256(data).hexdigest()
        request = {"source_hash": digest, "image_sha256": digest,
                   "image_base64": base64.b64encode(data).decode(), "width": 20, "height": 20,
                   "ocr_identity": identity,
                   "words": [{"text": "TOTAL", "left": 1, "top": 1, "width": 5,
                              "height": 5, "page": 1, "line_id": "1", "confidence": 90}]}
        endpoint = f"http://127.0.0.1:{server.server_port}"
        def post(payload):
            req = Request(endpoint + "/predict", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
            return json.loads(urlopen(req, timeout=2).read())
        result = post(request)
        assert result["source_hash"] == digest and result["model_identity"] == "m" * 64
        bad = dict(request, source_hash="0" * 64)
        req = Request(endpoint + "/predict", data=json.dumps(bad).encode(),
                      headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(req, timeout=2)
        assert error.value.code == 400
        too_many = dict(request, words=[request["words"][0]] * (receipt_gpu_worker.MAX_MODEL_WORDS + 1))
        with pytest.raises(HTTPError) as error:
            urlopen(Request(endpoint + "/predict", data=json.dumps(too_many).encode(),
                            headers={"Content-Type": "application/json"}), timeout=2)
        assert error.value.code == 400
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
