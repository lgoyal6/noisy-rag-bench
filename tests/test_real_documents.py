import pytest

from real_document_eval import (band_for, paired_document_interval, reference_receipt,
                                score_ranking, tsv_passages)


def word(text, y=10):
    return {"text": text, "quad": {f"{axis}{i}": (y if axis == "y" else 5)
                                   for axis in "xy" for i in range(1, 5)}}


def test_oracle_value_and_location_do_not_enter_queries():
    gt = {"meta": {"image_size": {"height": 120}},
          "gt_parse": {"total": {"total_price": "987.65"}},
          "valid_line": [{"category": "total.total_price", "words": [word("987.65", 75)]}]}
    texts, qs, _ = reference_receipt(gt)
    assert qs[0]["gold_bands"] == [7]
    assert "987" not in qs[0]["query"]
    assert texts[7] == "987.65"


def test_inconsistent_publisher_label_is_reported_not_silently_scored():
    gt = {"meta": {"image_size": {"height": 100}},
          "gt_parse": {"total": {"total_price": "15"}},
          "valid_line": [{"category": "total.total_price", "words": [word("150")]}]}
    _, qs, excluded = reference_receipt(gt)
    assert not qs
    assert excluded["label_not_in_annotated_line"] == 1


def test_duplicate_fields_are_excluded_explicitly():
    line = {"category": "total.total_price", "words": [word("5")]}
    gt = {"meta": {"image_size": {"height": 100}}, "gt_parse": {"total": {"total_price": "5"}},
          "valid_line": [line, line]}
    assert not reference_receipt(gt)[1]


def test_ocr_bands_use_geometry_not_gold_answers():
    tsv = "level\tleft\ttop\theight\tconf\ttext\n5\t1\t70\t10\t90\tTOTAL\n5\t2\t70\t10\t20\tS0\n"
    texts, conf = tsv_passages(tsv, 120, 12)
    assert texts[7] == "TOTAL S0"
    assert conf == [90, 20]


def test_plain_text_output_is_rejected_instead_of_scored_as_empty_ocr():
    with pytest.raises(ValueError, match="TSV"):
        tsv_passages("TOTAL 60.000\nCASH 70.000\n", 120, 12)


def test_wrong_location_does_not_rescue_corrupted_gold_value():
    q = {"gold_bands": [0], "answer": "5"}
    out = score_ranking([0, 1], q, ["TOTAL 15", "CASH 5"], k=2)
    assert out["recall"] == 1
    assert out["strict_grounded_recovery"] == 0
    assert out["wrong_band_only_support"] == 1
    assert out["strict_gold_region_available"] == 0


def test_ocr_availability_is_separate_from_retrieval_failure():
    out = score_ranking([1, 0], {"gold_bands": [0], "answer": "5"}, ["TOTAL 5", "CASH 8"], k=1)
    assert out["strict_grounded_recovery"] == 0
    assert out["strict_gold_region_available"] == 1


def test_canonical_collision_is_not_exact_financial_correctness():
    out = score_ranking([0], {"gold_bands": [0], "answer": "50"}, ["SO"], k=1)
    assert out["canonical_grounded_recovery"] == 1
    assert out["strict_grounded_recovery"] == 0
    assert out["canonical_only_recovery"] == 1


@pytest.mark.parametrize("text", ["150", "15.0", "-15", "15,000"])
def test_numeric_substrings_do_not_count_as_recovered_values(text):
    assert score_ranking([0], {"gold_bands": [0], "answer": "15"}, [text], k=1)["strict_grounded_recovery"] == 0


def test_document_cluster_interval_does_not_treat_questions_as_independent():
    rows = [{"document_id": doc, "arm": arm, "value": value}
            for doc, repeats in [("a", 1), ("b", 100)]
            for arm, value in [("left", 1 if doc == "a" else 0), ("right", 0)]
            for _ in range(repeats)]
    result = paired_document_interval(rows, "left", "right", "value")
    assert result["document_count"] == 2
    assert result["macro_document_difference"] == .5


def test_unpaired_documents_are_rejected():
    with pytest.raises(ValueError, match="identical documents"):
        paired_document_interval([{"document_id": "a", "arm": "left", "v": 1}], "left", "right", "v")


def test_out_of_bounds_image_centres_are_clamped():
    assert band_for(-1, 100, 12) == 0
    assert band_for(100, 100, 12) == 11


def test_duplicate_rankings_are_rejected():
    with pytest.raises(ValueError):
        score_ranking([0, 0], {"gold_bands": [0], "answer": "1"}, ["1", "2"], k=2)
