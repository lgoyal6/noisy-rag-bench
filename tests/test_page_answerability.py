import pytest

from heldout_eval import evaluate_pages


def test_retrieval_hit_and_page_answerability_are_separate():
    chunks = [
        {"doc_id": "d1", "section": "p1", "text": "The answer is $4 million."},
        {"doc_id": "d1", "section": "p2", "text": "Unrelated covenant text."},
    ]
    questions = [{"qid": "q1", "gold_chunk_id": 0, "answer": "$4 million"}]
    out = evaluate_pages([[1]], questions, chunks)
    assert out["retrieval_recall"] == 0
    assert out["answer_recoverability"] == 0
    assert out["n_source_locations"] == 1
    assert out["location_field"] == "page_id when present, otherwise section"


def test_answerable_passage_can_be_retrieved_without_gold_chunk_hit():
    chunks = [{"doc_id": "d1", "section": "p1", "text": "Fee is 3.5%"},
              {"doc_id": "d1", "section": "p2", "text": "Fee is 3.5%"}]
    out = evaluate_pages([[1]], [{"qid": "q1", "gold_chunk_id": 0, "answer": "3.5%"}], chunks)
    assert out["retrieval_recall"] == 0
    assert out["answer_recoverability"] == 1


def test_retrieved_gold_location_with_corrupted_value_is_not_answerable():
    chunks = [{"doc_id": "d1", "page_id": 1,
               "text": "The threshold is $15 million after OCR."}]
    out = evaluate_pages([[0]], [{"gold_chunk_id": 0, "answer": "$5 million"}], chunks)
    assert out["retrieval_recall"] == 1
    assert out["answer_recoverability"] == 0


def test_exact_answer_match_rejects_substring_and_unsupported_numeric_normalization():
    chunks = [{"doc_id": "d1", "page_id": 1,
               "text": "The limit is 15.0 million."}]
    for answer in ("5", "15 million"):
        out = evaluate_pages([[0]], [{"gold_chunk_id": 0, "answer": answer}], chunks)
        assert out["answer_recoverability"] == 0


def test_bad_gold_is_not_silently_dropped():
    with pytest.raises(ValueError):
        evaluate_pages([[0]], [{"gold_chunk_id": 2, "answer": "x"}], [{"doc_id": "d", "section": "p", "text": "x"}])


@pytest.mark.parametrize("text", ["The value is 5.0 million.", "The value is 5,000.", "The value is 15.5.", "The value is -5."])
def test_bare_numeric_answer_does_not_match_another_number(text):
    out=evaluate_pages([[0]],[{"gold_chunk_id":0,"answer":"5"}],[{"doc_id":"d","page_id":1,"text":text}])
    assert out["answer_recoverability"]==0


def test_wrong_page_answer_cannot_ground_corrupted_gold_page():
    chunks=[{"doc_id":"d","page_id":1,"text":"Fee is 7%."},
            {"doc_id":"d","page_id":2,"text":"Fee is 5%."}]
    out=evaluate_pages([[0,1]],[{"gold_chunk_id":0,"answer":"5%"}],chunks)
    assert out["retrieval_recall"]==1
    assert out["answer_recoverability"]==1
    assert out["grounded_answer_recoverability"]==0
    assert out["wrong_location_only_support"]==1


def test_exact_numeric_answer_with_sentence_punctuation_is_accepted():
    chunks=[{"doc_id":"d","page_id":1,"text":"The value is 5."}]
    out=evaluate_pages([[0]],[{"gold_chunk_id":0,"answer":"5"}],chunks)
    assert out["answer_recoverability"]==1
    assert out["grounded_answer_recoverability"]==1
