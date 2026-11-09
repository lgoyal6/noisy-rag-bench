#!/usr/bin/env bash
# Same frozen split and policy lifecycle for all arms; run under provider cap.
set -euo pipefail
: "${DATA:?}" "${SROIE:?}" "${MANIFEST:?}" "${OCR:?}" "${OUT:?}" "${PROTOCOL:?}" "${DEADLINE_EPOCH:?}"
PY=${PY:-python}
mkdir -p "$OUT"
exec 9>"$OUT/.campaign.lock"
flock -n 9 || exit 2
common=(--protocol "$PROTOCOL" --split-manifest "$MANIFEST" --ocr-dir "$OCR")
for file in "$DATA"/train-*.parquet; do common+=(--train-parquet "$file"); done
for file in "$SROIE"/*.parquet; do common+=(--sroie-parquet "$file"); done
stage() {
  local name=$1; shift
  [ ! -e "$OUT/$name.done" ] || return 0
  local remaining=$((DEADLINE_EPOCH - $(date +%s)))
  [ "$remaining" -gt 0 ] || exit 124
  local previous="" arg
  for arg in "$@"; do
    if { [ "$previous" = --output-dir ] || [ "$previous" = --out ]; } && [ -e "$arg" ]; then
      mkdir -p "$OUT/interrupted"
      mv -- "$arg" "$OUT/interrupted/$name-$(date +%s)"
    fi
    previous=$arg
  done
  [ ! -e "$OUT/$name.log" ] || mv "$OUT/$name.log" "$OUT/$name-interrupted-$(date +%s).log"
  if timeout --signal=INT --kill-after=20 "$remaining" "$@" >"$OUT/$name.log" 2>&1; then
    printf '{"stage":"%s","exit":0}\n' "$name" >>"$OUT/stages.jsonl"
    touch "$OUT/$name.done"
  else
    local rc=$?
    printf '{"stage":"%s","exit":%s}\n' "$name" "$rc" >>"$OUT/stages.jsonl"
    return "$rc"
  fi
}
arms=(geometry band layoutlm-full layoutlm-lora qwen)
for arm in "${arms[@]}"; do
  mkdir -p "$OUT/$arm"
  case "$arm" in
    geometry|band) mode="eval-$arm"; extra=();;
    qwen) mode=eval-qwen; extra=();;
    layoutlm-*)
      method=${arm#layoutlm-}
      stage "$arm-train" "$PY" receipt_model_study.py --mode train-layoutlm --method "$method" \
        "${common[@]}" --eval-role development --epochs 3 --seed 6406 --batch-size 2 \
        --learning-rate 0.00002 --output-dir "$OUT/$arm/training"
      mode=eval-layoutlm; extra=(--checkpoint "$OUT/$arm/training");;
  esac
  for role in development calibration_fit calibration_select; do
    stage "$arm-$role" "$PY" receipt_model_study.py --mode "$mode" "${common[@]}" \
      "${extra[@]}" --eval-role "$role" --output-dir "$OUT/$arm/$role"
  done
  stage "$arm-calibrate" "$PY" receipt_calibration.py \
    --fit "$OUT/$arm/calibration_fit/predictions.jsonl" \
    --select "$OUT/$arm/calibration_select/predictions.jsonl" --out "$OUT/$arm/calibration.json"
done
# Exercise actual checkpoint/policy inference through the internal HTTP worker.
for arm in layoutlm-full layoutlm-lora; do
  stage "$arm-api-smoke" "$PY" receipt_model_study.py --mode smoke-api "${common[@]}" \
    --checkpoint "$OUT/$arm/training" --calibration "$OUT/$arm/calibration.json" \
    --eval-role development --output-dir "$OUT/$arm/api-smoke"
done
# No arm reads final labels until every training recipe and policy is frozen.
for arm in "${arms[@]}"; do
  case "$arm" in
    geometry|band) mode="eval-$arm"; extra=();;
    qwen) mode=eval-qwen; extra=();;
    layoutlm-*) mode=eval-layoutlm; extra=(--checkpoint "$OUT/$arm/training");;
  esac
  stage "$arm-final" "$PY" receipt_model_study.py --mode "$mode" "${common[@]}" \
    "${extra[@]}" --eval-role final --calibration "$OUT/$arm/calibration.json" --output-dir "$OUT/$arm/final"
done
for arm in layoutlm-full layoutlm-lora; do
  # Conversion failure is evidence; the report must keep its exact reason.
  stage "$arm-export" "$PY" receipt_model_study.py --mode export-onnx "${common[@]}" \
    --checkpoint "$OUT/$arm/training" --calibration "$OUT/$arm/calibration.json" \
    --output-dir "$OUT/$arm/export" || true
done
stage report "$PY" receipt_study_report.py --runs "$OUT" --arms "${arms[@]}" --out "$OUT/report"
