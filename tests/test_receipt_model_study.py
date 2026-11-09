from io import BytesIO
import json

from PIL import Image
import pytest

from receipt_model_study import (CORD_REVISION, FIELD_TO_CATEGORY, LABEL_ID,
                                 LAYOUTLM_REVISION, QWEN_REVISION,
                                 _quad_box, sample_from_record, summarize_predictions,
                                 verify_cord, oracle_sample_from_record)
from receipt_model_study import validate_denominator


def test_qwen_summary_keeps_invalid_inputs_in_denominator_without_json_metadata():
    from receipt_model_study import qwen_output_counts
    rows = [{"fields": [{"gold": "1", "value": None, "exact": False,
                         "status": "INVALID", "reason": "over_12_megapixels"}]},
            {"fields": [{"gold": "1", "exact": True, "json_valid": True, "box_valid": True}]},
            {"fields": [{"gold": "1", "exact": False, "json_valid": False, "box_valid": False}]}]
    result = qwen_output_counts(rows)
    assert result == {"documents": 3, "eligible_fields": 3, "exact_fields": 1,
                      "exact_rate": 1 / 3, "validated_box_fields": 1,
                      "invalid_input_documents": 1, "invalid_json_documents": 1}


def test_study_and_serving_share_original_pixels_and_invalid_empty_ocr_semantics():
    from receipt_model_study import scored_prediction_record
    from receipt_model_worker import fields_from_predictions
    image = BytesIO()
    Image.new("RGB", (201, 103), "white").save(image, format="PNG")
    record = {"image": {"bytes": image.getvalue()}, "ground_truth": json.dumps({
        "meta": {"image_id": "doc", "annotated_fields": ["total"]},
        "gt_parse": {"total": {"total_price": "12.90"}}, "valid_line": []})}
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    numeric = "5\t1\t1\t1\t1\t1\t55\t13\t40\t9\t90\tRM12.90\n"
    invalid = "5\t1\t1\t1\t1\t2\t300\t13\t40\t9\t90\twrong\n"
    policy = {"isotonic": [{"max_score": 1, "probability": 1}], "threshold": .5,
              "autoaccept_fields": ["total"], "model_identity": "test"}
    for extra, supported in [("", True), (invalid, False)]:
        sample, _ = sample_from_record(record, ocr_tsv=header + numeric + extra)
        field = scored_prediction_record(sample, {0: ("B-TOTAL", .9)}, source="test")["fields"][0]
        served = fields_from_predictions(sample.ocr_words, {0: ("B-TOTAL", .9)},
            width=201, height=103, policy=policy, policy_hash="test",
            invalid_source_boxes=sample.invalid_source_boxes)["total"]
        assert field["box"] == [55, 13, 95, 22]
        assert field["source_supported"] is served["source_supported"] is supported
        assert field["value"] == served["value"] == "12.90"
    empty, _ = sample_from_record(record, ocr_tsv=header)
    result = scored_prediction_record(empty, {}, source="test")["fields"][0]
    assert result["value"] is None and not result["source_supported"]


def test_manifest_denominator_rejects_missing_duplicate_and_wrong_scope_outputs():
    members = {"a": {"role": "development", "source": "cord", "document_id": "a"},
               "b": {"role": "development", "source": "sroie", "document_id": "b"}}
    rows = [{"image_sha256": key, "document_id": key,
             "fields": [{"field": field} for field in fields]}
            for key, fields in [("a", ["total", "subtotal", "tax"]), ("b", ["total"])]]
    validate_denominator(rows, members, "development")
    with pytest.raises(ValueError, match="missing images"):
        validate_denominator(rows[:1], members, "development")
    with pytest.raises(ValueError, match="duplicate image"):
        validate_denominator(rows + rows[:1], members, "development")
    with pytest.raises(ValueError, match="unsupported field"):
        validate_denominator([rows[0], {**rows[1], "fields": [{"field": "tax"}]}], members, "development")


def test_quadrilateral_is_clipped_and_scaled_to_layoutlm_range():
    box = _quad_box({"quad": {"x1": -4, "y1": 10, "x2": 50, "y2": 10,
                              "x3": 50, "y3": 60, "x4": -4, "y4": 60}},
                    width=100, height=100)
    assert box == [0, 100, 500, 600]
    assert _quad_box({"quad": {}}, width=100, height=100) is None


def test_oracle_ocr_diagnostic_is_separate_from_actual_ocr():
    image = BytesIO()
    Image.new("RGB", (100, 200), "white").save(image, format="PNG")
    def word(text, x0, x1):
        return {"text": text, "quad": {"x1": x0, "y1": 10, "x2": x1, "y2": 10,
                "x3": x1, "y3": 20, "x4": x0, "y4": 20}}
    gt = {"meta": {"split": "valid", "image_id": "doc-1"},
          "gt_parse": {"total": {"total_price": "1000"},
                       "sub_total": {"subtotal_price": "800", "tax_price": "200"}},
          "valid_line": [
              {"category": "total.total_price", "words": [word("TOTAL", 2, 20), word("1000", 50, 80)]},
              {"category": "sub_total.subtotal_price", "words": [word("SUBTOTAL", 2, 30), word("800", 50, 80)]},
              {"category": "sub_total.tax_price", "words": [word("TAX", 2, 15), word("200", 50, 80)]},
          ]}
    record = {"ground_truth": json.dumps(gt), "image": {"bytes": image.getvalue()}}
    sample, parsed = oracle_sample_from_record(record)
    assert parsed["gt_parse"]["total"]["total_price"] == "1000"
    assert sample.words == ["TOTAL", "1000", "SUBTOTAL", "800", "TAX", "200"]
    assert sample.tags == [LABEL_ID["B-TOTAL"], LABEL_ID["I-TOTAL"],
                           LABEL_ID["B-SUBTOTAL"], LABEL_ID["I-SUBTOTAL"],
                           LABEL_ID["B-TAX"], LABEL_ID["I-TAX"]]
    assert sample.gold == {"total": "1000", "subtotal": "800", "tax": "200"}
    assert len(FIELD_TO_CATEGORY) == 3
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    tsv = header + "5\t1\t1\t1\t1\t1\t50\t10\t30\t10\t85\t100O\n"
    actual, _ = sample_from_record(record, ocr_tsv=tsv)
    assert actual.words == ["100O"]
    assert actual.tags == [LABEL_ID["O"]]
    assert actual.boxes == [[500, 50, 800, 100]]
    gt["valid_line"] = []
    gt["gt_parse"] = {"total": {"total_price": "999999"}}
    second, _ = sample_from_record({**record, "ground_truth": json.dumps(gt)}, ocr_tsv=tsv)
    assert (second.words, second.boxes, second.tags) == (actual.words, actual.boxes, actual.tags)
    with pytest.raises(TypeError):
        sample_from_record(record)


def test_train_shard_identity_is_pinned_and_unknown_shards_rejected(tmp_path):
    path = tmp_path / "not-the-publisher-train-shard.parquet"
    path.write_bytes(b"not parquet")
    with pytest.raises(ValueError, match="four pinned"):
        verify_cord(path, "train")


def test_metric_denominator_and_fixed_uncalibrated_threshold_are_explicit():
    rows = [{"document_id": "a", "fields": [
        {"gold": "1", "exact": True, "auto_accept_at_0_8": True},
        {"gold": "2", "exact": False, "auto_accept_at_0_8": False}]},
        {"document_id": "b", "fields": [
            {"gold": None, "exact": False, "auto_accept_at_0_8": False}]}]
    result = summarize_predictions(rows)
    assert result["eligible_fields"] == 2
    assert result["exact_rate_all_fields"] == .5
    assert result["coverage_at_frozen_0_8"] == .5
    assert result["accepted_error_rate_at_frozen_0_8"] == 0
    assert result["confidence_is_uncalibrated"] is True


def test_all_data_and_model_revisions_are_full_immutable_ids():
    for value in (CORD_REVISION, LAYOUTLM_REVISION, QWEN_REVISION):
        assert len(value) == 40
        int(value, 16)
