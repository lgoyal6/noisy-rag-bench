# Receipt evidence review baseline

This extension adds a real image-to-field workflow on top of the existing
retrieval study. It runs Tesseract word-level OCR, associates explicit total,
subtotal and tax labels with same-line numeric candidates, parses amounts with
`Decimal`, and returns each value with a source box, page, original OCR text,
input hash, processor revision, status and reason codes. A value is accepted
only when exactly one nearby labeled candidate and an explicit same-line
currency are present. A bounded bipartite assignment uses SciPy's linear-sum
solver, maximizing assigned fields before minimizing position/alignment and
OCR-confidence cost. Source candidates have unit capacity; candidates shared
by incompatible fields always require review. Unsupported, conflicting, locale-ambiguous or
out-of-bounds evidence stays in review. Scores are deterministic geometry
heuristics, not calibrated probabilities.

The local UI stores original model outputs by content hash in SQLite and records
human corrections as append-only events. It never overwrites an extraction.
The default path runs locally. An operator may configure a GPU worker reached
through an SSH loopback forward; that path sends image bytes and locally
computed OCR evidence to the configured worker. It binds only to `127.0.0.1`, accepts
one image up to 10 MiB and 12 megapixels, invokes Tesseract with a 20-second
timeout, and bounds request handlers and model inference queues. Rotated EXIF images
are rejected so source coordinates cannot drift silently. Use licensed public
receipts for the demo.

Install the existing real-document dependencies, ensure `tesseract` and English
trained data are present, then run from the repository root:

```sh
uv run --with-requirements requirements-real-documents.txt -- python receipt_http.py --db /tmp/receipt-review.sqlite
```

Open `http://127.0.0.1:8765/`, select one public PNG/JPEG receipt and inspect the
overlay and status. The default local OCR language is English. The historical
CORD study used English and Indonesian language models. Calibrated model
serving requires `--tessdata-dir` and the exact Tesseract version, PSM, language
set and trained-data hashes recorded by the frozen policy. The English-only
default is a geometry demonstration, not calibrated model inference.

For an additional parser result on the old raw CORD validation TSVs:

```sh
uv run --with pytest --with numpy --with pyarrow --with pillow -- \
  python receipt_field_eval.py --exploratory \
  --parquet "$INPUTS/validation-00000-of-00001-cc3c5779fe22e8ca.parquet" \
  --tsv-dir "$INPUTS/validation-v2-psm6" --output "$OUTPUTS/field-eval"
```

The explicit `--exploratory` flag is required because validation OCR and layout
results were examined previously. It is not a fresh final test, a template
family split, an independent human audit, or a generalization guarantee. The
field scorer uses publisher parse values only after prediction creation; it
counts ambiguous publisher amount formats as exclusions. Report coverage,
accepted numeric error and document-cluster uncertainty separately. It does not
score currency because the supported labels do not consistently provide it.

## Known gates

- LayoutLMv3 full fine-tuning and LoRA, Qwen2.5-VL, separate calibration and
  TensorRT conversion are implemented in `receipt_model_study.py` and
  `receipt_campaign.sh`. GPU execution remains unverified until an authorized
  campaign produces raw outputs. LayoutLM uses actual Tesseract words/boxes;
  publisher geometry supplies training targets and scoring only. The separate
  oracle diagnostic must never be reported as primary OCR performance.
- The frozen CORD plus SROIE manifest contains 214 training, 20 development,
  40 confidence-fit, 100 independent policy-validation and 60 final heuristic
  merchant/template groups. Exposed and unresolved groups are excluded.
  Final predictions must cover every expected image and annotated field,
  including invalid/oversize inputs. These groups are heuristic, not a human
  verified identity audit or a guarantee against pretraining overlap.
- SROIE labels totals only. Every model policy explicitly restricts
  autoacceptance to `total`; subtotal and tax remain review suggestions.
  Isotonic confidence and one threshold candidate are fit on confidence-fit
  groups, then independently validated once. Failure to establish the fixed
  5% group-any-error bound selects review-all. The bound is not a field-risk
  or distribution-shift guarantee. All policies freeze before final scoring.
- The pinned LayoutLMv3 weights use CC-BY-NC-SA-4.0 terms; Qwen2.5-VL-3B uses
  Qwen's Research License. Preserve those restrictions, attribution and
  non-commercial scope; do not redistribute model weights or imply permissive
  licensing.
- `receipt_model_worker.py` verifies checkpoint, policy, split and OCR identity
  at startup. `receipt_gpu_worker.py` binds only to loopback and accepts pixels
  plus the calibrated upstream OCR; it never substitutes Linux OCR. Configure
  `receipt_http.py --gpu-worker-url` only through an SSH loopback forward,
  with `--gpu-worker-model-identity` and `--gpu-worker-policy-sha256` pinned
  from verified campaign artifacts. Health responses cannot choose the
  expected identities. Remote responses require all three field keys.
  Timed-out work cannot persist a late prediction. Model prediction and
  correction versions are additive to legacy geometry records.
- Conversion feasibility, task parity, calibrated GPU API behavior and the
  interactive model demonstration still require execution evidence.

The previous OCR retriever metrics in `docs/real-documents.md` remain a separate
evidence-recovery study. They are not field extraction accuracy and are not
promoted as the performance result of this extension.
