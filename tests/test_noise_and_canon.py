"""Tests for the corruption generators, the canonicalizer and the metrics.

The headline finding is that recall stays flat while answers become
unrecoverable, and that an 11-line canonicalizer wins most of that back. Both
halves rest on code that had nothing verifying it: if `canon` collapsed too
aggressively it would manufacture the recovery, and if the noise generators were
not deterministic the whole degradation curve would be unreproducible.

Pure NumPy and stdlib. No model, no network, no downloads.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from noise import (  # noqa: E402
    CONDITIONS,
    FAMILIES,
    char_error_rate,
    corrupt_chunks,
    header_noise,
    hyphenation_noise,
    ocr_noise,
    redaction_noise,
    whitespace_noise,
)
from retrieval import answer_present, canon, recall_at_k, rr_at_k, rrf  # noqa: E402

import numpy as np  # noqa: E402


def chunks(n=3):
    return [
        {"id": f"c{i}", "doc_id": "d1", "doc_title": "T", "section": "S",
         "text": f"Section {i}. The commitment amount is 4,500,000 dollars and the "
                 f"facility matures in March. Account number 8815-{i}."}
        for i in range(n)
    ]


class TestCharErrorRate:
    def test_identical_text_scores_zero(self):
        assert char_error_rate("hello world", "hello world") == 0.0

    def test_one_substitution_in_ten_chars_is_one_tenth(self):
        assert char_error_rate("abcdefghij", "abcdefghiX") == pytest.approx(0.1)

    def test_empty_reference_returns_zero_rather_than_dividing_by_zero(self):
        assert char_error_rate("", "anything") == 0.0

    def test_insertions_and_deletions_both_count(self):
        assert char_error_rate("abc", "abcd") == pytest.approx(1 / 3)
        assert char_error_rate("abcd", "abc") == pytest.approx(0.25)

    def test_completely_different_text_scores_high(self):
        assert char_error_rate("aaaa", "bbbb") == pytest.approx(1.0)


class TestCanon:
    def test_undoes_the_rn_to_m_confusion(self):
        # The single most common OCR failure on scanned documents.
        assert canon("arnount") == canon("amount")

    def test_undoes_digit_letter_confusions(self):
        assert canon("8815") == canon("881S")
        assert canon("l0O") == canon("100")

    def test_rejoins_a_word_hyphenated_across_a_line_break(self):
        assert canon("commit-\nment") == canon("commitment")

    def test_collapses_runs_of_whitespace(self):
        assert canon("a    b\n\tc") == "a b c"

    def test_is_idempotent(self):
        # It runs over both the index and the query, so applying it twice must
        # not drift, or the two sides stop matching.
        once = canon("The arnount is 4,S00.")
        assert canon(once) == once

    def test_does_not_collapse_genuinely_different_words(self):
        # This is the failure mode that would manufacture the recovery: if canon
        # maps everything onto everything, answer_present starts passing for the
        # wrong reasons and the 0.83 is an artefact.
        assert canon("March") != canon("April")
        assert canon("4500000") != canon("4600000")
        assert canon("facility") != canon("penalty")

    def test_distinct_account_numbers_stay_distinct(self):
        assert canon("8815-0") != canon("8815-1")


class TestAnswerPresent:
    def test_finds_an_exact_answer(self):
        assert answer_present("4,500,000", ["the amount is 4,500,000 dollars"]) == 1.0

    def test_misses_an_ocr_corrupted_answer_without_canonicalization(self):
        assert answer_present("4,500,000", ["the arnount is 4,S00,000 dollars"]) == 0.0

    def test_recovers_that_same_answer_with_canonicalization(self):
        """This one test is the finding.

        Exact matching says the answer is gone. The canonicalizer says it is
        still there. That gap between the two columns, 0.51 against 0.83 on the
        real run, is the entire argument of this repository.
        """
        corrupted = ["the arnount is 4,5OO,OOO dollars"]
        assert answer_present("4,500,000", corrupted) == 0.0
        assert answer_present("4,500,000", corrupted, canonicalize=True) == 1.0

    def test_searches_across_all_retrieved_passages_not_just_the_first(self):
        assert answer_present("March", ["nothing here", "matures in March"]) == 1.0

    def test_absent_answer_stays_absent_under_canonicalization(self):
        # Canonicalization must not invent a hit.
        assert answer_present("September", ["matures in March"], canonicalize=True) == 0.0


class TestRetrievalMetrics:
    def test_recall_is_one_when_gold_is_inside_k(self):
        assert recall_at_k(np.array([3, 7, 1]), 7, 5) == 1.0

    def test_recall_is_zero_when_gold_falls_outside_k(self):
        assert recall_at_k(np.array([3, 7, 1]), 1, 2) == 0.0

    def test_reciprocal_rank_reflects_position(self):
        assert rr_at_k(np.array([5, 9, 2]), 5, 10) == 1.0
        assert rr_at_k(np.array([5, 9, 2]), 9, 10) == 0.5

    def test_reciprocal_rank_is_zero_on_a_miss(self):
        assert rr_at_k(np.array([5, 9, 2]), 42, 10) == 0.0


class TestRRF:
    """rrf returns scores indexed by doc id, so a ranking is argsort descending."""

    @staticmethod
    def _ranking(scores):
        return list(np.argsort(-scores))

    def test_a_document_ranked_first_by_both_lists_wins(self):
        scores = rrf([np.array([0, 1, 2]), np.array([0, 2, 1])])
        assert self._ranking(scores)[0] == 0

    def test_a_split_vote_beats_the_middle_at_any_k(self):
        """Counterintuitive, and worth pinning because it is easy to assume the
        opposite.

        Doc 1 sits second in both lists. Docs 0 and 2 sit first in one and last
        in the other. The split placements win, and they win at every k, because
        1/x is convex: 1/(k+1) + 1/(k+3) > 2/(k+2) always. RRF is not averaging
        positions and it is not rewarding consensus on position.

        Where RRF does reward agreement is in membership. A document that appears
        in both lists at all beats one that appears in only a single list, which
        is the next test.
        """
        for k in (1, 60, 500):
            scores = rrf([np.array([2, 1, 0]), np.array([0, 1, 2])], k=k)
            assert scores[0] == pytest.approx(scores[2]), f"k={k}"
            assert scores[0] > scores[1], f"k={k}"

    def test_appearing_in_both_lists_beats_appearing_in_one(self):
        # This is the property the hybrid retriever actually relies on.
        both = rrf([np.array([0]), np.array([0])])
        one = rrf([np.array([0]), np.array([1])])

        assert both[0] > one[0]

    def test_it_sizes_by_the_largest_id_not_the_list_length(self):
        # A sparse or one-based ranking is valid input per the signature and used
        # to raise IndexError here.
        scores = rrf([np.array([7]), np.array([7])])
        assert len(scores) == 8
        assert scores[7] > 0

    def test_an_empty_input_returns_an_empty_score_array(self):
        assert len(rrf([])) == 0


class TestNoiseFamilies:
    @pytest.mark.parametrize("fn", [ocr_noise, whitespace_noise, hyphenation_noise, redaction_noise])
    def test_a_zero_rate_leaves_text_untouched(self, fn):
        text = "The commitment amount is 4,500,000 dollars."
        assert fn(text, 0.0, random.Random(1)) == text

    @pytest.mark.parametrize("fn", [ocr_noise, whitespace_noise, redaction_noise])
    def test_a_high_rate_actually_changes_the_text(self, fn):
        text = "The commitment amount is 4,500,000 dollars and it matures in March."
        assert fn(text, 0.9, random.Random(1)) != text

    def test_the_same_seed_produces_the_same_corruption(self):
        # Without this the degradation curve is not reproducible and no number
        # in the README can be checked.
        text = "The commitment amount is 4,500,000 dollars."
        assert ocr_noise(text, 0.3, random.Random(7)) == ocr_noise(text, 0.3, random.Random(7))

    def test_different_seeds_produce_different_corruption(self):
        text = "The commitment amount is 4,500,000 dollars and it matures in March."
        a = ocr_noise(text, 0.3, random.Random(1))
        b = ocr_noise(text, 0.3, random.Random(999))
        assert a != b

    def test_redaction_removes_content_rather_than_marking_it(self):
        # A blackout that left a marker would be trivially detectable. Real ones
        # delete the span, which is why they are dangerous.
        text = " ".join(f"word{i}" for i in range(40))
        out = redaction_noise(text, 0.5, random.Random(3))
        assert len(out.split()) < len(text.split())
        assert "[REDACTED]" not in out

    def test_header_noise_injects_a_known_header(self):
        from noise import RUNNING_HEADERS

        out = header_noise("body text " * 40, 1.0, random.Random(2))
        stems = [h.split("{")[0].split(" - ")[0].strip() for h in RUNNING_HEADERS]
        assert any(s and s in out for s in stems)

    def test_header_noise_adds_rather_than_replaces(self):
        body = "the commitment amount is 4,500,000 dollars " * 10
        out = header_noise(body, 1.0, random.Random(2))
        assert "commitment amount" in out
        assert len(out) > len(body)


class TestCorruptChunks:
    def test_it_preserves_ids_and_count(self):
        original = chunks(4)
        out = corrupt_chunks(original, {"ocr": 0.2}, seed=5)
        assert [c["id"] for c in out] == [c["id"] for c in original]

    def test_it_does_not_mutate_the_input(self):
        # The clean corpus is reused for every condition, so mutating it would
        # silently compound noise across the whole ladder.
        original = chunks(2)
        before = [c["text"] for c in original]
        corrupt_chunks(original, {"ocr": 0.9}, seed=5)
        assert [c["text"] for c in original] == before

    def test_an_empty_spec_is_a_no_op(self):
        original = chunks(2)
        out = corrupt_chunks(original, {}, seed=5)
        assert [c["text"] for c in out] == [c["text"] for c in original]

    def test_it_is_deterministic_for_a_given_seed(self):
        a = corrupt_chunks(chunks(3), {"ocr": 0.3}, seed=11)
        b = corrupt_chunks(chunks(3), {"ocr": 0.3}, seed=11)
        assert [c["text"] for c in a] == [c["text"] for c in b]


class TestConditionLadder:
    def test_it_starts_from_a_clean_condition(self):
        assert CONDITIONS[0][0] == "clean"

    def test_the_clean_condition_applies_no_noise(self):
        assert not CONDITIONS[0][1]

    def test_every_condition_names_only_known_families(self):
        for name, spec in CONDITIONS:
            for family in spec:
                assert family in FAMILIES, f"{name} references unknown family {family}"

    def test_condition_names_are_unique(self):
        names = [n for n, _ in CONDITIONS]
        assert len(names) == len(set(names))
