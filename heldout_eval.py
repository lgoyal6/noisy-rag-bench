"""Page-level answerability scoring kept separate from retrieval recall."""
from __future__ import annotations

import re
from collections import defaultdict


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _answer_occurs(answer: str, text: str) -> bool:
    """Require whole-token boundaries so 5 does not match 15 or 5.0."""
    answer = _normalized(answer)
    if not answer:
        return False
    left = r"(?<!\w)"
    right = r"(?!\w)"
    if answer[0].isdigit():
        left += r"(?<![.,+\-])"
    if answer[-1].isdigit():
        right += r"(?![.,]\d)"
    return re.search(left + re.escape(answer) + right, _normalized(text)) is not None


def evaluate_pages(rankings, questions, chunks, *, k: int = 5) -> dict:
    """Report retrieval hit and answer recoverability by document/page.

    Inputs follow corpus.build_corpus's public shape. Each question contributes
    once. Repeated questions on one document are correlated, so aggregate
    intervals must resample documents, not treat these questions as independent.
    Literal recoverability on any page is separate from answer support on the
    labelled document/page; neither metric independently judges answer quality.
    """
    if len(rankings) != len(questions):
        raise ValueError("one ranking is required per question")
    if k < 1:
        raise ValueError("k must be positive")
    groups = defaultdict(lambda: {"n": 0, "retrieved_gold": 0, "answerable": 0,
                                  "retrieved_and_answerable": 0, "grounded_answerable": 0,
                                  "wrong_location_only_support": 0})
    for ranking, q in zip(rankings, questions):
        gold_id = int(q["gold_chunk_id"])
        if not 0 <= gold_id < len(chunks):
            raise ValueError(f"gold_chunk_id out of range for {q.get('qid')}")
        gold = chunks[gold_id]
        key = (gold["doc_id"], gold.get("page_id", gold.get("section", "unknown")))
        top_ids = [int(i) for i in list(ranking)[:k]]
        if any(i < 0 or i >= len(chunks) for i in top_ids):
            raise ValueError("ranking contains an invalid chunk index")
        hit = gold_id in top_ids
        answer = _normalized(q["answer"])
        answerable = any(_answer_occurs(answer, chunks[i]["text"]) for i in top_ids)
        grounded = any(
            (chunks[i]["doc_id"], chunks[i].get("page_id", chunks[i].get("section", "unknown"))) == key
            and _answer_occurs(answer, chunks[i]["text"])
            for i in top_ids
        )
        row = groups[key]
        row["n"] += 1
        row["retrieved_gold"] += int(hit)
        row["answerable"] += int(answerable)
        row["retrieved_and_answerable"] += int(hit and answerable)
        row["grounded_answerable"] += int(grounded)
        row["wrong_location_only_support"] += int(answerable and not grounded)
    per_location = [{"doc_id": doc, "location": page, **counts,
                 "retrieval_recall": counts["retrieved_gold"] / counts["n"],
                 "answer_recoverability": counts["answerable"] / counts["n"],
                 "grounded_answer_recoverability": counts["grounded_answerable"] / counts["n"]}
                for (doc, page), counts in sorted(groups.items())]
    n = sum(r["n"] for r in per_location)
    return {"k": k, "n_questions": n, "n_source_locations": len(per_location),
            "location_field": "page_id when present, otherwise section",
            "retrieval_recall": sum(r["retrieved_gold"] for r in per_location) / n if n else 0.0,
            "answer_recoverability": sum(r["answerable"] for r in per_location) / n if n else 0.0,
            "grounded_answer_recoverability": sum(r["grounded_answerable"] for r in per_location) / n if n else 0.0,
            "wrong_location_only_support": sum(r["wrong_location_only_support"] for r in per_location),
            "retrieval_answerability_gap": (sum(r["retrieved_gold"] for r in per_location) - sum(r["answerable"] for r in per_location)) / n if n else 0.0,
            "per_location": per_location}
