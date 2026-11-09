"""Report frozen-policy receipt outcomes from immutable per-document outputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from receipt_calibration import supported, probability, correct


def policy_counts(rows, policy):
    groups = defaultdict(lambda: [0, 0, 0, 0])
    for row in rows:
        for field in row["fields"]:
            counts = groups[row["group"]]
            counts[0] += field["gold"] is not None
            accepted = (field.get("field") in policy["autoaccept_fields"] and supported(field)
                        and probability(policy["isotonic"], field.get("confidence")) >= policy["threshold"])
            counts[1] += accepted and field["gold"] is not None
            counts[2] += accepted
            counts[3] += accepted and not correct(field)
    return dict(groups)


def paired_coverage(left, right, *, seed=6406, draws=2000):
    if set(left) != set(right):
        raise ValueError("arms do not cover identical template groups")
    groups = sorted(left)
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        selected = [groups[rng.randrange(len(groups))] for _ in groups]
        a = [sum(left[g][i] for g in selected) for i in range(2)]
        b = [sum(right[g][i] for g in selected) for i in range(2)]
        if a[0] and b[0]:
            values.append(a[1] / a[0] - b[1] / b[0])
    values.sort()
    return {"template_groups": len(groups), "candidate_minus_baseline_ci95":
            [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]] if values else None,
            "scope": "paired whole-template bootstrap; fixed policies, local public-dataset sample only"}


def report(root, arms):
    result = {"arms": {}, "paired_coverage": {}, "performance_win_established": False}
    counts = {}
    ids = None
    for arm in arms:
        path = root / arm / "final"
        rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
        policy = json.loads((root / arm / "calibration.json").read_text())
        current = {(row["image_sha256"], f["field"]) for row in rows for f in row["fields"]}
        if ids is None:
            ids = current
        elif current != ids:
            raise ValueError("arms changed field/image denominator")
        metrics = json.loads((path / "metrics.json").read_text())
        counts[arm] = policy_counts(rows, policy)
        result["arms"][arm] = metrics
    baseline = arms[0]
    for arm in arms[1:]:
        result["paired_coverage"][arm] = paired_coverage(counts[arm], counts[baseline])
    result["interpretation"] = ("Each policy is frozen before final data. Review-all is a valid adverse outcome, "
        "not a coverage win. Numeric/source risk and unsupported accepts remain in denominators. "
        "Currency is reported from visible OCR support; no publisher currency ground truth is assumed. "
        "Heuristic merchant/template grouping and possible pretraining overlap limit generalization. "
        "No performance-win claim is computed automatically from a single favorable arm.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("report output already exists")
    result = report(args.runs, args.arms)
    args.out.mkdir(parents=True)
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Receipt field study", "", "Frozen policies, same image/field denominator. Final curves are descriptive only.", "",
             "| Arm | Documents | Eligible fields | Frozen-policy coverage | Accepted risk |", "|---|---:|---:|---:|---:|"]
    for arm, metrics in result["arms"].items():
        p = metrics["frozen_policy_result"]
        lines.append(f"| {arm} | {metrics['documents']} | {p['eligible_fields']} | {p['coverage']:.4f} | {p['field_risk']} |")
    lines += ["", result["interpretation"], "", "Paired coverage intervals:", "", "```json", json.dumps(result["paired_coverage"], indent=2), "```", ""]
    (args.out / "report.md").write_text("\n".join(lines))
