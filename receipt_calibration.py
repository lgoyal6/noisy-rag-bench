"""Fit confidence separately, select a bounded-risk policy, then freeze it.

The simultaneous binomial bound is over independent accepted template groups
with any error, not correlated fields. Its exchangeability assumption can fail
under template shift. Final field risk is measured separately without retuning.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import random
from pathlib import Path

THRESHOLDS = [.5, .6, .7, .8, .9, .95, .99, 1.0]


def isotonic_fit(points):
    grouped = defaultdict(list)
    for score, correct in points:
        grouped[float(score)].append(bool(correct))
    blocks = []
    for score, values in sorted(grouped.items()):
        blocks.append([score, sum(values), len(values)])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([right[0], left[1] + right[1], left[2] + right[2]])
    return [{"max_score": s, "probability": total / count, "count": count}
            for s, total, count in blocks]


def probability(blocks, score):
    if not blocks or score is None:
        return 0.0
    return next((b["probability"] for b in blocks if score <= b["max_score"]), blocks[-1]["probability"])


def binomial_upper(errors, count, alpha):
    if not count or errors == count:
        return 1.0
    lo, hi = errors / count, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        terms = [math.lgamma(count + 1) - math.lgamma(k + 1) - math.lgamma(count - k + 1)
                 + k * math.log(p) + (count - k) * math.log1p(-p) for k in range(errors + 1)]
        largest = max(terms)
        cdf = math.exp(largest) * sum(math.exp(t - largest) for t in terms)
        if cdf > alpha:
            lo = p
        else:
            hi = p
    return hi


def flatten(rows):
    for row in rows:
        for field in row["fields"]:
            yield {**field, "document_id": row["document_id"], "group": row["group"]}


def supported(field):
    return bool(field.get("value") is not None and field.get("box") and
                field.get("source_supported", True) and field.get("unique_candidate_count", 1) == 1)


def correct(field):
    return bool(field.get("gold") is not None and field.get("exact") and field.get("source_iou", 0) >= .5)


def risk_curve(rows, blocks, thresholds=THRESHOLDS, alpha=.05, autoaccept_fields=None):
    fields = list(flatten(rows))
    eligible = sum(f.get("gold") is not None for f in fields)
    curve = []
    for threshold in thresholds:
        accepted = [f for f in fields if (autoaccept_fields is None or f.get("field") in autoaccept_fields)
                    and supported(f) and probability(blocks, f.get("confidence")) >= threshold]
        groups = defaultdict(list)
        for field in accepted:
            groups[field["group"]].append(correct(field))
        errors = sum(not all(values) for values in groups.values())
        curve.append({"threshold": threshold, "accepted_fields": len(accepted),
                      "eligible_fields": eligible,
                      "coverage": sum(f.get("gold") is not None for f in accepted) / eligible if eligible else 0,
                      "field_errors": sum(not correct(f) for f in accepted),
                      "field_risk": sum(not correct(f) for f in accepted) / len(accepted) if accepted else None,
                      "unsupported_accepts": sum(f.get("gold") is None for f in accepted),
                      "accepted_groups": len(groups), "groups_with_any_error": errors,
                      "group_any_error_upper": binomial_upper(errors, len(groups), alpha / len(thresholds))})
    return curve


def fit_policy(fit_rows, select_rows, *, target_risk=.05):
    fit_groups = {r["group"] for r in fit_rows}
    select_groups = {r["group"] for r in select_rows}
    if fit_groups & select_groups:
        raise ValueError("confidence-fit and threshold-selection template groups overlap")
    if not fit_rows or not select_rows:
        raise ValueError("separate nonempty calibration-fit and calibration-select sets required")
    # SROIE annotates total only. Freeze one globally enforceable field scope;
    # unlabelled subtotal/tax must never inherit its risk validation.
    scope = ["total"]
    blocks = isotonic_fit((f.get("confidence") or 0, correct(f))
                          for f in flatten(fit_rows) if f.get("field") in scope and supported(f))
    # Choose exactly one candidate using confidence-fit data. Independent
    # selection data validates that fixed policy once, or forces review-all.
    fit_curve = risk_curve(fit_rows, blocks, autoaccept_fields=scope)
    candidates = [p for p in fit_curve if p["accepted_groups"] >= 10 and
                  p["groups_with_any_error"] / p["accepted_groups"] <= target_risk / 2]
    candidate = max(candidates, key=lambda point: point["coverage"], default=None)
    curve = risk_curve(select_rows, blocks, autoaccept_fields=scope)
    checked = risk_curve(select_rows, blocks, [candidate["threshold"]], autoaccept_fields=scope)[0] if candidate else None
    chosen = checked if checked and checked["accepted_groups"] >= 10 and checked["group_any_error_upper"] <= target_risk else None
    return {"version": 1, "autoaccept_fields": scope,
            "isotonic": blocks, "threshold": chosen["threshold"] if chosen else 1.000001,
            "target_risk": target_risk, "confidence_fit_curve": fit_curve,
            "fixed_candidate_validation": checked, "selection_curve": curve,
            "fit_groups": sorted(fit_groups), "select_groups": sorted(select_groups),
            "interpretation": "Confidence-fit data chooses one candidate; separate groups validate it once with a one-sided 95% Clopper-Pearson bound on accepted template groups with any numeric/source error. Groups must be independent/exchangeable. The grid curve is descriptive with simultaneous Bonferroni bounds, never used to choose another candidate. This is not a field-risk or distribution-shift guarantee. Failed or insufficient validation selects review-all."}


def final_intervals(rows, blocks, threshold, *, seed=6406, draws=2000, autoaccept_fields=None):
    """Resample whole template groups, retaining correlated fields/documents."""
    groups = defaultdict(lambda: [0, 0, 0, 0])
    for field in flatten(rows):
        counts = groups[field["group"]]
        counts[0] += field.get("gold") is not None
        accepted = ((autoaccept_fields is None or field.get("field") in autoaccept_fields)
                    and supported(field) and probability(blocks, field.get("confidence")) >= threshold)
        counts[1] += accepted
        counts[2] += accepted and not correct(field)
        counts[3] += accepted and field.get("gold") is not None
    values = list(groups.values())
    rng = random.Random(seed)
    risks, coverage = [], []
    for _ in range(draws if values else 0):
        sample = [values[rng.randrange(len(values))] for _ in values]
        totals = [sum(v[i] for v in sample) for i in range(4)]
        if totals[0]:
            coverage.append(totals[3] / totals[0])
        if totals[1]:
            risks.append(totals[2] / totals[1])

    def interval(numbers):
        numbers.sort()
        return [numbers[int(.025 * (len(numbers) - 1))], numbers[int(.975 * (len(numbers) - 1))]] if numbers else None
    return {"template_groups": len(values), "bootstrap_draws": draws, "seed": seed,
            "field_coverage_ci95": interval(coverage), "accepted_field_risk_ci95": interval(risks),
            "draws_with_accepted_fields": len(risks),
            "scope": "Template-cluster percentile bootstrap, conditional on this dataset and fixed policy. Few groups and distribution shift limit coverage; this is not a selective-risk guarantee."}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-risk", type=float, default=.05)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("policy already exists; never rewrite a frozen policy")
    fit_rows, select_rows = read_rows(args.fit), read_rows(args.select)
    if any(row.get("role") != "calibration_fit" for row in fit_rows) or any(row.get("role") != "calibration_select" for row in select_rows):
        raise SystemExit("calibration inputs must have their declared frozen roles")
    identities = {(row.get("model_identity"), row.get("split_manifest_sha256"))
                  for row in fit_rows + select_rows}
    if len(identities) != 1 or not all(next(iter(identities))):
        raise SystemExit("calibration inputs must bind one model and frozen dataset")
    policy = fit_policy(fit_rows, select_rows, target_risk=args.target_risk)
    policy["model_identity"], policy["split_manifest_sha256"] = next(iter(identities))
    ocr_identities = {json.dumps(row.get("ocr_identity"), sort_keys=True) for row in fit_rows + select_rows}
    if len(ocr_identities) != 1 or json.loads(next(iter(ocr_identities))) is None:
        raise SystemExit("calibration inputs must bind the exact OCR pipeline")
    policy["ocr_identity"] = json.loads(next(iter(ocr_identities)))
    policy["inputs"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.fit, args.select]}
    encoded = (json.dumps(policy, indent=2, sort_keys=True) + "\n").encode()
    args.out.write_bytes(encoded)
    args.out.with_suffix(".sha256").write_text(hashlib.sha256(encoded).hexdigest() + "\n")
