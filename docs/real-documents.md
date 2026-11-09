# Real-receipt evidence recovery

`real_document_eval.py` adds an external-data check beside the generated noise
study. It runs actual Tesseract OCR on the complete released CORD test or validation split and
compares BM25, OCR-canonical BM25, and optionally the existing NumPy MiniLM and
hybrid retrievers. No model is trained or tuned on these test answers.

## What is being tested

The user has already selected one receipt. Retrieval chooses among twelve fixed
horizontal image bands, which are constructed without answer labels. Seven
predeclared numeric-field query templates cover total, cash, change, subtotal,
tax, service and discount. The publisher's `gt_parse` values and annotated
locations supply the answer oracle. Missing, repeated or internally unsupported
publisher fields are counted as exclusions.

The annotated-reference arm uses the released word transcriptions, not complete
native PDF text. The OCR arm runs on the actual receipt images with fixed
Tesseract page-segmentation and language settings. The two arms use identical
queries, geometry and retrieval methods.

Report retrieval recall and literal answer recovery separately. A value found
only in another band does not rescue a corrupted target. Canonical-only recovery
is separate because replacing letters with digits can conflate distinct values.
Finding a value somewhere in the target band is weaker than correctly extracting
its field; this study does not measure semantic answer generation.

## Source and attribution

CORD v2 is published by NAVER CLOVA under CC BY 4.0:

- [Original release and description](https://github.com/clovaai/cord)
- [License](https://github.com/clovaai/cord/blob/327310ce58c1623255821d062b3a759ff3789e3c/LICENSE-CC-BY)
- [Publisher dataset](https://huggingface.co/datasets/naver-clova-ix/cord-v2)

Park, Shin, Lee, Lee, Surh, Seo and Lee, *CORD: A Consolidated Receipt Dataset
for Post-OCR Parsing*, NeurIPS Document Intelligence Workshop, 2019. The receipt
images and published annotations are external material; OCR text and benchmark
outputs are derived in the local run. They are not committed to this repository.

Pinned dataset revision: `7f0115a4b758a71d6473b8d085751692da2fef98`.
File: `data/test-00000-of-00001-9c204eb3f4e11791.parquet`.
SHA-256: `51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7`.
The runner rejects another input instead of labeling it as this release.

The separate validation parquet is
`data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet`, SHA-256
`0d0f6dac11fdcc549de2746aa9f53136a3bc22a2a1aff2b0b847f7622ad60c15`.
Use `--split validation` for this file. Its embedded publisher metadata says
`valid`; the runner checks that mapping explicitly.

## Reproduce

Use Python 3.12 and install `requirements-real-documents.txt`. Tesseract must be
installed separately. Obtain the pinned publisher parquet and use a private
directory for inputs and outputs. Supply a tessdata directory containing the
English and Indonesian language files. The runner records their hashes and the
Tesseract version. The evaluated Indonesian fast model comes from
`tesseract-ocr/tessdata_fast` revision
`923915d4ced2a7235221788285785a29c4a42d4a`.

```sh
python real_document_eval.py \
  --parquet "$STUDY_INPUTS/test-00000-of-00001-9c204eb3f4e11791.parquet" \
  --tessdata-dir "$STUDY_INPUTS/tessdata" \
  --output-dir "$STUDY_OUTPUTS"
```

Add `--dense` to include MiniLM and hybrid. The existing MiniLM loader needs
`model.safetensors` and `tokenizer.json` from
`sentence-transformers/all-MiniLM-L6-v2` in the local HF cache. The protocol
records their actual hashes. The verified model revision is
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

Outputs retain the frozen protocol, source/image/annotation hashes, raw OCR TSV,
OCR durations, per-question rankings, paired document-bootstrap intervals and
summary metrics. Existing reports are never overwritten. A changed code/config
protocol requires a fresh output directory. Non-TSV OCR output fails visibly;
successful Tesseract exit status alone is insufficient.

`--psm 6` uses Tesseract's uniform-block segmentation; `--psm 11` uses its existing
sparse-text mode. A follow-up compares those modes on validation data before
choosing a layout for another diagnostic test run. The initial test results were
already inspected to diagnose OCR failures, so subsequent test comparisons are
exploratory and must not be described as untouched confirmatory evidence.
`strict_gold_region_available` measures whether OCR preserved the exact value in
the labeled band at all, independently of retrieval. Keep that OCR ceiling
separate from retrieval-plus-recovery.

## Limits

- This is one external receipt dataset, not the planned four-family study.
- The publisher's official test split is used, but no template-disjoint split
  or additional independent human adjudication is claimed.
- Band-level support is coarser than exact field attribution.
- English MiniLM is a baseline on Indonesian receipts, not a multilingual model
  quality claim.
- Query latency includes dense encoding. OCR and index construction are separate
  costs; single-machine timing is descriptive, not a production speedup claim.
- Repeated questions on the same receipt are correlated. Confidence intervals
  resample documents, and their estimand is the macro-document paired difference.
- No new OCR correction algorithm should be accepted merely because it improves
  a canonical string metric; harmful value changes need their own oracle.

## Recorded local results, October 31, 2026

The initial diagnostic processed all 100 official test images. The seven field
definitions yielded 338 eligible questions on 99 receipts; 362 absent/repeated
field opportunities were reported as exclusions. With actual PSM6 OCR:

| Retriever | Labeled-band recall at 3 | Exact value recovery in labeled band at 3 |
| --- | ---: | ---: |
| BM25 | 40.5% | 26.6% |
| OCR-canonical BM25 | 41.1% | 26.9% |
| NumPy MiniLM | 63.3% | 30.5% |
| Hybrid | 47.3% | 27.8% |

MiniLM's same-region recovery was 88.2% on the released annotated transcriptions.
That is an OCR-and-retrieval diagnostic, not a 57.7-point model improvement or a
semantic extraction-accuracy claim. All 2,704 original per-field metric rows
were reconstructed exactly from retained rankings and passages.

A follow-up compared existing Tesseract sparse-text PSM11 with PSM6 on the
separate 100-image validation split, using 359 eligible fields. The primary
measure was exact value availability in the labeled region before retrieval.
PSM11 minus PSM6 was -4.03 percentage points under the macro-document paired
estimand; its document-bootstrap 95% interval was [-11.08, +3.33] points. The
secondary BM25 retrieval-plus-recovery difference was -4.52 points, interval
[-10.80, +2.25]. The candidate was not adopted. These intervals include zero;
neither a reliable improvement nor a reliable degradation is established.

The complete offline suite passed 81 tests. The study records include pinned
inputs, OCR language hashes, raw TSV, rankings and uncertainty inputs in private
run state. The broader multi-family/template-disjoint study and additional human
adjudication remain open. This result provides one external dataset evaluation.
