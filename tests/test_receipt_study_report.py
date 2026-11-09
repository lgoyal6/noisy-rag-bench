import pytest

from receipt_study_report import paired_coverage
from receipt_calibration import final_intervals, fit_policy


def test_paired_bootstrap_keeps_template_groups_and_denominators():
    a = {"a": [2, 1, 1, 0], "b": [10, 5, 5, 0]}
    b = {"a": [2, 0, 0, 0], "b": [10, 0, 0, 0]}
    result = paired_coverage(a, b, draws=100)
    assert result["template_groups"] == 2
    assert result["candidate_minus_baseline_ci95"] == [.5, .5]
    with pytest.raises(ValueError, match="identical"):
        paired_coverage(a, {"a": b["a"]})


def test_validating_candidate_does_not_search_a_replacement_threshold():
    def rows(prefix, n, exact):
        return [{"document_id": f"{prefix}{i}", "group": f"{prefix}{i}", "fields": [
            {"field": "total", "value": "1", "gold": "1", "box": [0, 0, 1, 1],
             "confidence": .9, "exact": exact, "source_iou": 1,
             "unique_candidate_count": 1}]} for i in range(n)]
    fit = rows("fit", 40, True)
    select = rows("select", 100, False)
    policy = fit_policy(fit, select)
    assert policy["fixed_candidate_validation"]["groups_with_any_error"] == 100
    assert policy["threshold"] > 1  # No lower-coverage threshold chosen on validation.
    interval = final_intervals(rows("final", 4, True), policy["isotonic"], policy["threshold"], draws=100)
    assert interval["template_groups"] == 4
    assert interval["accepted_field_risk_ci95"] is None
