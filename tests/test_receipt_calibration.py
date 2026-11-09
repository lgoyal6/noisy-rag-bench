import math

import pytest

from receipt_calibration import binomial_upper, fit_policy, isotonic_fit, probability, risk_curve
from receipt_dataset import freeze_groups, unique_image_members


def record(name, split, header, dhash):
    return {"document_id": name, "publisher_split": split, "image_sha256": name,
            "ocr_sha256": "ocr-" + name,
            "signature": {"header": [header] if header else [], "dhash": dhash,
                          "width": 100, "height": 200}}


def test_exposed_merchant_and_duplicate_images_never_cross_roles():
    records = [record("a", "test", "merchantalpha", "0" * 16),
               record("b", "train", "merchantalpha", "f" * 16),
               record("c", "train", "merchantbeta", "a" * 16),
               record("c", "train", "merchantbeta", "a" * 16)]
    manifest = freeze_groups(records)
    assert [m["role"] for m in manifest["members"] if m["document_id"] in {"a", "b"}] == ["excluded_exposed"] * 2
    assert len({m["group"] for m in manifest["members"] if m["document_id"] == "c"}) == 1
    assert manifest == freeze_groups(list(reversed(records)))
    assert len(manifest["members"]) == 3
    assert sum(g["documents"] for g in manifest["groups"]) == 3


def test_duplicate_image_aliases_are_canonical_and_cannot_cross_role_or_ocr_identity():
    aliases = [{"image_sha256": "same", "document_id": name, "publisher_split": "train",
                "group": "g", "role": "train", "ocr_sha256": "ocr"} for name in ["z", "a"]]
    unique = unique_image_members(aliases)
    assert len(unique) == 1 and unique[0]["document_id"] == "a"
    assert len(unique[0]["image_aliases"]) == 2
    assert unique == unique_image_members(list(reversed(aliases)))
    for key in ["role", "group", "ocr_sha256"]:
        with pytest.raises(ValueError, match="crosses"):
            unique_image_members([aliases[0], {**aliases[1], key: "other"}])


def test_isotonic_confidence_is_monotone_and_binomial_bound_is_not_zero():
    blocks = isotonic_fit([(.1, False), (.2, True), (.8, False), (.9, True)])
    values = [probability(blocks, score) for score in [.1, .2, .8, .9]]
    assert values == sorted(values)
    assert math.isclose(binomial_upper(0, 100, .05), 1 - .05 ** .01, abs_tol=1e-10)
    assert binomial_upper(0, 2, .05) > .7


def row(group, *, exact=True, iou=1, gold="10"):
    return {"document_id": group, "group": group, "fields": [
        {"field": "total", "value": "10", "gold": gold, "box": [1, 1, 4, 4], "confidence": .9,
         "source_supported": True, "unique_candidate_count": 1,
         "source_iou": iou, "exact": exact}]}


def test_separate_calibration_groups_and_small_sample_review_fallback():
    with pytest.raises(ValueError, match="overlap"):
        fit_policy([row("same")], [row("same")])
    policy = fit_policy([row("fit")], [row("select")])
    assert policy["threshold"] > 1
    assert not risk_curve([row("final")], policy["isotonic"], [policy["threshold"]])[0]["accepted_fields"]


def test_correct_number_wrong_source_and_unlabelled_accepts_count_as_errors():
    curve = risk_curve([row("wrong-box", iou=0), row("unsupported", gold=None)],
                       [{"max_score": 1, "probability": 1}], [.5])[0]
    assert curve["field_errors"] == 2
    assert curve["unsupported_accepts"] == 1
    assert curve["groups_with_any_error"] == 2


def test_unlabelled_subtotal_cannot_inherit_total_only_risk_policy():
    fit = [row(f"fit{i}") for i in range(40)]
    select = [row(f"select{i}") for i in range(100)]
    for record in fit + select:
        record["fields"].append({**record["fields"][0], "field": "subtotal", "gold": None})
    policy = fit_policy(fit, select)
    assert policy["autoaccept_fields"] == ["total"]
    assert policy["threshold"] <= 1
    assert policy["fixed_candidate_validation"]["accepted_fields"] == 100
    result = risk_curve(select, policy["isotonic"], [policy["threshold"]],
                        autoaccept_fields=policy["autoaccept_fields"])[0]
    assert result["unsupported_accepts"] == 0
