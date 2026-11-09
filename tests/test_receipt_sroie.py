import json

from receipt_sroie import as_cord
from receipt_dataset import freeze_groups


def example(words, total="12.90"):
    return {"key": "fixture", "image": {"bytes": b"fixture-not-decoded"},
            "entities": {"total": total, "company": "Fixture Company"},
            "words": words, "bboxes": [[i * 10, 0, i * 10 + 9, 9] for i in range(len(words))]}


def test_sroie_supervision_keeps_only_supported_field_and_unique_source():
    gt = json.loads(as_cord(example(["TOTAL", "12.9"]))["ground_truth"])
    assert gt["meta"]["annotated_fields"] == ["total"]
    assert not gt["meta"]["source_label_ambiguous"]
    assert gt["valid_line"][1]["category"] == "total.total_price"
    ambiguous = json.loads(as_cord(example(["12.90", "TOTAL", "12.9"]))["ground_truth"])
    assert ambiguous["meta"]["source_label_ambiguous"]
    assert all(line["category"] == "other" for line in ambiguous["valid_line"])


def test_unseen_public_test_is_distinct_from_previously_inspected_test():
    member = {"document_id": "unseen", "publisher_split": "test", "exposed": False,
              "image_sha256": "a", "ocr_sha256": "o", "company_group": "fixture",
              "signature": {"header": [], "dhash": "0" * 16, "width": 100, "height": 200}}
    manifest = freeze_groups([member])
    assert not manifest["members"][0]["role"].startswith("excluded_")
    exposed = freeze_groups([{**member, "exposed": True}])
    assert exposed["members"][0]["role"] == "excluded_exposed"
