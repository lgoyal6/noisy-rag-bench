"""Loopback-only, single-request-at-a-time extraction/review UI for licensed samples."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import threading

from receipt_service import MAX_UPLOAD_BYTES, analyze_image, append_correction, persist_prediction
from receipt_model_worker import (BoundedModelWorker, LocalLayoutLMBackend,
                                  RemoteLayoutLMBackend, WorkerBusy, WorkerTimeout,
                                  ocr_identity)

_PAGE = b'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Receipt evidence review</title><style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}canvas{max-width:100%;border:1px solid #aaa}pre{white-space:pre-wrap;background:#f3f3f3;padding:1rem}.row{margin:.6rem 0}</style>
<h1>Receipt evidence review</h1><p>OCR runs on this computer. Model suggestions use the configured worker. Only totals within the frozen calibration scope can be accepted; subtotal and tax suggestions require review.</p>
<input id="file" type="file" accept="image/png,image/jpeg"><button id="run">Extract</button><p id="state"></p><canvas id="view"></canvas><div id="fields"></div><pre id="raw"></pre>
<script>
let last=null;const $=id=>document.getElementById(id);
$("run").onclick=async()=>{let f=$("file").files[0];if(!f)return;$("state").textContent="Running local OCR...";
let form=new FormData();form.append("image",f);let r=await fetch("/extract",{method:"POST",body:f});let j=await r.json();if(!r.ok){$("state").textContent=j.error;return}last=j;$("state").textContent="Source SHA-256 "+j.document.source_hash;
let c=$("view"),ctx=c.getContext("2d"),img=new Image();img.onload=()=>{c.width=img.width;c.height=img.height;ctx.drawImage(img,0,0);for(let [name,x] of Object.entries(j.fields)){for(let b of x.source_boxes){ctx.strokeStyle=x.status==="ACCEPT"?"#087a36":"#b20";ctx.lineWidth=Math.max(2,img.width/400);ctx.strokeRect(b.left,b.top,b.width,b.height)}}};img.src=URL.createObjectURL(f);
$("fields").innerHTML="";for(let [name,x] of Object.entries(j.fields)){let d=document.createElement("div");d.className="row";d.innerHTML=`<b>${name}</b>: ${x.value??"(no supported value)"} ${x.currency??""} [${x.status}] ${x.reason_codes.join(", ")}`;if(x.status!=="ACCEPT"){let valueInput=document.createElement("input");valueInput.setAttribute("aria-label","Corrected "+name+" value");valueInput.placeholder="Corrected value";valueInput.maxLength=64;valueInput.value=x.value??"";
let currencyInput=document.createElement("input");currencyInput.setAttribute("aria-label","Currency for "+name);currencyInput.placeholder="Currency code";currencyInput.maxLength=3;currencyInput.size=5;currencyInput.value=x.currency??"";
let b=document.createElement("button");b.textContent="Save "+name+" correction";b.onclick=async()=>{let value=valueInput.value.trim(),currency=currencyInput.value.trim().toUpperCase();if(!value||!currency){$("state").textContent="Enter a reviewed value and currency code";return}b.disabled=true;try{let out=await fetch("/corrections",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({source_hash:j.document.source_hash,field:name,value,currency,correction_id:crypto.randomUUID(),model_identity:j.model?.identity,calibration_policy_sha256:j.model?.calibration_policy_sha256})});$("state").textContent=out.ok?"Correction saved as an append-only record":"Correction save failed"}catch{$("state").textContent="Correction save failed"}finally{b.disabled=false}};d.append(valueInput,currencyInput,b)}$("fields").appendChild(d)}$("raw").textContent=JSON.stringify(j,null,2)};
</script>'''


class BoundedHTTPServer(ThreadingHTTPServer):
    """Accept at most one active and one queued request handler."""
    daemon_threads = True
    request_queue_size = 5

    def __init__(self, *args, **kwargs):
        self._handler_slots = threading.BoundedSemaphore(2)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        self._handler_slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._handler_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handler_slots.release()


def make_handler(db_path, model_worker=None, *, tesseract="tesseract",
                 tessdata_dir=None, languages="eng"):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ReceiptReview/1"
        def _json(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/":
                return self._json(404, {"error": "not found"})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_PAGE)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(_PAGE)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                return self._json(400, {"error": "invalid content length"})
            if length < 0 or length > MAX_UPLOAD_BYTES:
                return self._json(413, {"error": "request must be at most 10 MiB"})
            body = self.rfile.read(length)
            if self.path == "/extract":
                try:
                    result = analyze_image(body, tesseract=tesseract, model_worker=model_worker,
                                            tessdata_dir=tessdata_dir, languages=languages)
                    persist_prediction(db_path, result)
                    return self._json(200, result)
                except WorkerBusy as exc:
                    return self._json(503, {"error": str(exc), "retryable": True})
                except WorkerTimeout as exc:
                    return self._json(504, {"error": str(exc), "retryable": True})
                except ValueError as exc:
                    return self._json(400, {"error": str(exc)})
                except RuntimeError as exc:
                    return self._json(503, {"error": str(exc)})
            if self.path == "/corrections":
                try:
                    request = json.loads(body)
                    append_correction(db_path, correction_id=request["correction_id"],
                        source_hash=request["source_hash"], field=request["field"],
                        value=request["value"], currency=request.get("currency"),
                        note=request.get("note", ""),
                        model_identity=request.get("model_identity"),
                        calibration_policy_sha256=request.get("calibration_policy_sha256"))
                    return self._json(201, {"saved": True, "correction_id": request["correction_id"]})
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    return self._json(400, {"error": str(exc)[:200]})
            return self._json(404, {"error": "not found"})

        def log_message(self, fmt, *args):
            pass
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--db", required=True)
    ap.add_argument("--checkpoint", help="administrator-configured local LayoutLM checkpoint root")
    ap.add_argument("--calibration", help="administrator-configured frozen policy JSON")
    ap.add_argument("--gpu-worker-url", help="operator-configured SSH-forwarded GPU worker, for example http://127.0.0.1:8877")
    ap.add_argument("--gpu-worker-model-identity", help="expected checkpoint/processor SHA-256")
    ap.add_argument("--gpu-worker-policy-sha256", help="expected frozen calibration policy SHA-256")
    ap.add_argument("--worker-timeout", type=float, default=45.0)
    ap.add_argument("--tesseract", default="tesseract", help="administrator-configured Tesseract executable")
    ap.add_argument("--tessdata-dir", help="administrator-configured pinned Tesseract language-data directory")
    ap.add_argument("--languages", help="calibrated OCR language set (eng+ind for this model)")
    args = ap.parse_args()
    if not 1024 <= args.port <= 65535:
        raise SystemExit("port must be between 1024 and 65535")
    if bool(args.checkpoint) != bool(args.calibration):
        raise SystemExit("--checkpoint and --calibration must be supplied together")
    if args.checkpoint and args.gpu_worker_url:
        raise SystemExit("configure either a local checkpoint or a GPU worker URL")
    if args.gpu_worker_url and not (args.gpu_worker_model_identity and args.gpu_worker_policy_sha256):
        raise SystemExit("remote worker requires operator-pinned model and calibration identities")
    if (args.checkpoint or args.gpu_worker_url) and not args.tessdata_dir:
        raise SystemExit("--tessdata-dir is required when the calibrated model worker is enabled")
    languages = args.languages or ("eng+ind" if (args.checkpoint or args.gpu_worker_url) else "eng")
    if args.worker_timeout <= 0:
        raise SystemExit("worker timeout must be positive")
    worker = None
    if args.checkpoint:
        # Load configured weights and policy once before binding the local service.
        worker = BoundedModelWorker(LocalLayoutLMBackend(
            args.checkpoint, args.calibration, tessdata_dir=args.tessdata_dir,
            tesseract=args.tesseract, languages=languages),
                                   max_pending=1, request_timeout=args.worker_timeout)
    elif args.gpu_worker_url:
        # OCR runs here with the calibrated Mac tessdata; remote process only sees pixels and OCR evidence.
        identity = ocr_identity(tesseract=args.tesseract, tessdata_dir=args.tessdata_dir,
                                languages=languages)
        remote = RemoteLayoutLMBackend(args.gpu_worker_url,
                                       expected_ocr_identity=identity,
                                       expected_model_identity=args.gpu_worker_model_identity,
                                       expected_policy_hash=args.gpu_worker_policy_sha256,
                                       timeout=args.worker_timeout)
        worker = BoundedModelWorker(remote, max_pending=1,
                                    request_timeout=args.worker_timeout)
    server = BoundedHTTPServer(("127.0.0.1", args.port), make_handler(
        args.db, worker, tesseract=args.tesseract, tessdata_dir=args.tessdata_dir,
        languages=languages))
    print(f"Local receipt reviewer at http://127.0.0.1:{args.port}/ (Ctrl-C to stop)")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
