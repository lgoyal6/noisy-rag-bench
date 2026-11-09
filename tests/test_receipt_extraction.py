from receipt_extraction import OCRWord, extract_fields, parse_amount, amount_values_equal, joint_assignment
from receipt_service import append_correction, persist_prediction
import sqlite3
import pytest


def test_joint_assignment_reassigns_greedy_choice_to_maximize_supported_fields():
    a, b = ("a",), ("b",)
    # Greedy total->a strands tax. The global optimum serves both fields.
    assert joint_assignment({"total": {a: 1, b: 2}, "tax": {a: 1}}) == {"total": b, "tax": a}
    assert joint_assignment({"total": {a: 10, b: 2}, "tax": {a: 1, b: 3}}) == {"total": b, "tax": a}
    with pytest.raises(ValueError, match="limit"):
        joint_assignment({"total": {(i,): i for i in range(129)}})


def test_shared_source_cannot_be_autoaccepted_as_two_incompatible_fields():
    words = [OCRWord("TOTAL", 1, 1, 20, 10, line_id="1"),
             OCRWord("TAX", 25, 1, 20, 10, line_id="1"),
             OCRWord("RM12.90", 55, 1, 40, 10, line_id="1")]
    fields = extract_fields(words, image_width=100, image_height=30,
                            source_hash="a" * 64)["fields"]
    assert all(fields[name]["status"] == "REVIEW" for name in ["total", "tax"])
    assert all("source_candidate_shared_between_fields" in fields[name]["reason_codes"]
               for name in ["total", "tax"])
    assert sum(fields[name]["joint_assignment_source"] is not None for name in fields) == 1


def test_exact_locale_values_and_ambiguous_separator():
    amount = parse_amount("Rp 1.234.567")
    assert (amount.value, amount.currency, amount.minor_units, amount.exponent) == (
        "1234567", "IDR", 1234567, 0)
    assert parse_amount("Rp 1.234.567,50") is None  # IDR minor units are not accepted
    assert parse_amount("$1,234.56").currency == "USD"
    assert parse_amount("15,00").value == "15.00"
    assert parse_amount("15.0").value == "15.0"
    assert parse_amount("15.000") is None  # locale cannot be inferred safely
    assert parse_amount("Rp 15.000").value == "15000"
    assert parse_amount("RM 12.90").currency == "MYR"
    assert parse_amount("RM12.90").value == "12.90"
    assert amount_values_equal("12.90", "12.9")
    assert not amount_values_equal("12.90", "1290")
    assert not amount_values_equal(None, "0")
    assert not amount_values_equal("NaN", "NaN")
    assert parse_amount("farm 12.90").currency is None
    assert parse_amount("RM 12 99") is None
    assert parse_amount("$12,34.56") is None
    assert parse_amount("RM1 200").value == "1200"


def test_field_requires_same_line_labeled_candidate_and_valid_source_box():
    words = [
        OCRWord("TOTAL", 5, 10, 30, 10, line_id="1"),
        OCRWord("Rp", 60, 10, 12, 10, line_id="1"),
        OCRWord("50000", 80, 10, 35, 10, line_id="1"),
        OCRWord("50000", 80, 40, 35, 10, line_id="2"),
    ]
    result = extract_fields(words, image_width=200, image_height=100, source_hash="a" * 64)
    assert result["fields"]["total"]["value"] == "50000"
    assert result["fields"]["total"]["source_boxes"] == [
        {"left": 80, "top": 10, "width": 35, "height": 10}]
    assert result["fields"]["total"]["status"] == "ACCEPT"
    assert result["fields"]["subtotal"]["status"] == "REVIEW"


def test_nearby_duplicate_amounts_force_review_instead_of_guessing():
    words = [OCRWord("TOTAL", 5, 10, 30, 10, line_id="1"),
             OCRWord("50000", 70, 10, 30, 10, line_id="1"),
             OCRWord("5000", 110, 10, 30, 10, line_id="1")]
    field = extract_fields(words, image_width=200, image_height=100,
                           source_hash="b" * 64)["fields"]["total"]
    assert field["status"] == "REVIEW"
    assert "multiple_amount_candidates" in field["reason_codes"]
    assert field["value"] is None


def test_tax_percentage_is_not_accepted_as_currency_amount():
    words = [OCRWord("Tax", 5, 10, 20, 10, line_id="1"),
             OCRWord("(10.0%)", 40, 10, 45, 10, line_id="1"),
             OCRWord("Rp", 100, 10, 15, 10, line_id="1"),
             OCRWord("3600", 120, 10, 35, 10, line_id="1")]
    field = extract_fields(words, image_width=200, image_height=100,
                           source_hash="f" * 64)["fields"]["tax"]
    assert field["status"] == "ACCEPT"
    assert field["value"] == "3600"


def test_partial_numeric_separator_keeps_line_in_review():
    words = [OCRWord("TAX", 5, 10, 20, 10, line_id="1"),
             OCRWord("Rp20,", 60, 10, 40, 10, line_id="1"),
             OCRWord("446", 102, 10, 25, 10, line_id="1")]
    field = extract_fields(words, image_width=200, image_height=100,
                           source_hash="1" * 64)["fields"]["tax"]
    assert field["status"] == "REVIEW"
    assert "incomplete_numeric_fragment" in field["reason_codes"]


def test_invalid_box_and_unlabeled_or_cross_line_values_cannot_be_accepted():
    words = [OCRWord("TOTAL", 5, 10, 30, 10, line_id="1"),
             OCRWord("42", 50, 10, 15, 10, line_id="2"),
             OCRWord("9", 190, 90, 20, 20, line_id="1")]
    field = extract_fields(words, image_width=200, image_height=100,
                           source_hash="c" * 64)["fields"]["total"]
    assert field["status"] == "REVIEW"
    assert field["value"] is None
    assert "invalid_source_boxes_present" in field["reason_codes"]
    assert "same_line_amount_missing" in field["reason_codes"]


def test_missing_label_is_review_and_result_hash_is_deterministic():
    words = [OCRWord("123", 10, 10, 20, 10, line_id="1")]
    first = extract_fields(words, image_width=100, image_height=100, source_hash="d" * 64)
    second = extract_fields(words, image_width=100, image_height=100, source_hash="d" * 64)
    assert first == second
    assert first["fields"]["total"]["status"] == "REVIEW"
    assert "explicit_field_label_missing" in first["fields"]["total"]["reason_codes"]


def test_predictions_are_immutable_and_corrections_are_append_only(tmp_path):
    result = extract_fields([OCRWord("TOTAL", 0, 0, 30, 10, line_id="1"),
                             OCRWord("Rp", 40, 0, 15, 10, line_id="1"),
                             OCRWord("1000", 60, 0, 25, 10, line_id="1")],
                            image_width=100, image_height=100, source_hash="e" * 64)
    result["document"] = {"source_hash": "e" * 64}
    db = tmp_path / "review.sqlite"
    persist_prediction(db, result)
    persist_prediction(db, result)
    append_correction(db, correction_id="c1", source_hash="e" * 64,
                      field="total", value="1100", currency="IDR")
    append_correction(db, correction_id="c1", source_hash="e" * 64,
                      field="total", value="1100", currency="IDR")
    with sqlite3.connect(db) as conn:
        assert conn.execute("select count(*) from predictions").fetchone()[0] == 1
        assert conn.execute("select count(*) from corrections").fetchone()[0] == 1
        original = conn.execute("select result_json from predictions").fetchone()[0]
        assert '"value": "1000"' in original
    with pytest.raises(ValueError, match="different content"):
        append_correction(db, correction_id="c1", source_hash="e" * 64,
                          field="total", value="1200", currency="IDR")
