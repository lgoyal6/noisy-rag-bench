"""Conservative, auditable field assignment over word-level OCR geometry.

This CPU reference deliberately never invents a field or repairs OCR text. It
accepts a value only when one same-line numeric candidate is anchored by an
explicit field label, has a valid source box, and has an unambiguous amount
syntax. It is a deterministic geometric baseline, not a learned model.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from collections import defaultdict
from typing import Iterable


FIELDS = {
    "total": ("total", "grand total", "jumlah total", "total bayar", "amount due"),
    "subtotal": ("subtotal", "sub total", "sub-total"),
    "tax": ("tax", "pajak", "ppn", "pb1", "vat"),
}
CURRENCIES = {
    "IDR": ("idr", "rp", "rp."), "USD": ("usd", "$"),
    "EUR": ("eur", "€"), "GBP": ("gbp", "£"),
    "JPY": ("jpy", "¥"),
    "MYR": ("myr", "rm"),
}
_NUMBER = re.compile(r"(?<![\w])[-−]?(?:\d[\d.,'’ ]*\d|\d)(?![\w])")


@dataclass(frozen=True)
class OCRWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    page: int = 0
    line_id: str = ""
    confidence: float | None = None

    def valid_box(self, width: int, height: int) -> bool:
        return (width > 0 and height > 0 and self.left >= 0 and self.top >= 0
                and self.width > 0 and self.height > 0
                and self.left + self.width <= width and self.top + self.height <= height)


@dataclass(frozen=True)
class Amount:
    value: str
    currency: str | None
    minor_units: int | None
    exponent: int | None


def parse_amount(text: str) -> Amount | None:
    """Parse a single conservative localized amount using Decimal semantics.

    Grouping-only inputs with 3-digit groups are accepted as integer amounts;
    a lone separator followed by three digits is considered ambiguous.
    """
    raw = text.strip().replace("−", "-").replace("’", "'")
    currency_hits = _currency_hits(raw)
    if len(currency_hits) > 1:
        return None
    numeric_text = raw
    if currency_hits:
        for token in CURRENCIES[currency_hits[0]]:
            numeric_text = re.sub(r"(?<![a-z])" + re.escape(token) + r"(?![a-z])",
                                  " ", numeric_text, flags=re.IGNORECASE)
    m = _NUMBER.search(numeric_text)
    if not m:
        return None
    token = m.group(0).strip()
    for separator in ("'", " "):
        if separator in token:
            groups = token.lstrip("-").split(separator)
            if not (1 <= len(groups[0]) <= 3 and all(re.fullmatch(r"\d{3}", g) for g in groups[1:])):
                return None
            token = token.replace(separator, "")
    negative = token.startswith("-")
    token = token.lstrip("-")
    comma, dot = token.rfind(","), token.rfind(".")
    decimal_sep = None
    if comma >= 0 and dot >= 0:
        decimal_sep = "," if comma > dot else "."
        group_sep = "." if decimal_sep == "," else ","
        if len(token) - max(comma, dot) - 1 not in (1, 2):
            return None
        groups = token[:max(comma, dot)].split(group_sep)
        if not (groups[0].isdigit() and 1 <= len(groups[0]) <= 3 and
                all(len(g) == 3 and g.isdigit() for g in groups[1:])):
            return None
        token = token.replace(group_sep, "").replace(decimal_sep, ".")
    elif comma >= 0 or dot >= 0:
        sep = "," if comma >= 0 else "."
        groups = token.split(sep)
        if len(groups) == 2 and len(groups[1]) == 3:
            if not currency_hits:
                return None
            token = "".join(groups)
        elif len(groups) > 2 and all(len(g) == 3 for g in groups[1:]):
            token = "".join(groups)
        elif len(groups) == 2 and 1 <= len(groups[1]) <= 2:
            decimal_sep = sep
            token = token.replace(sep, ".")
        else:
            return None
    if not re.fullmatch(r"\d+(?:\.\d{1,2})?", token):
        return None
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    if negative:
        value = -value
    currency = currency_hits[0] if currency_hits else None
    exponent = 0 if currency in ("IDR", "JPY") else 2
    scale = Decimal(10) ** exponent
    if value * scale != (value * scale).to_integral_value():
        return None
    return Amount(format(value, "f"), currency, int(value * scale), exponent)


def _currency_for(text: str) -> str | None:
    found = _currency_hits(text)
    return found[0] if len(found) == 1 else None


def _currency_hits(text: str) -> list[str]:
    return [code for code, tokens in CURRENCIES.items()
            if any(re.search(r"(?<![a-z])" + re.escape(token) + r"(?![a-z])", text.lower())
                   for token in tokens)]


def amount_values_equal(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    try:
        a, b = Decimal(left), Decimal(right)
        return a.is_finite() and b.is_finite() and a == b
    except (InvalidOperation, TypeError):
        return False


def _amount_tokens(words: Iterable[OCRWord]):
    for word in words:
        full = parse_amount(word.text)
        if full and full.currency and "%" not in word.text:
            yield word, word.text, full
            continue
        for m in _NUMBER.finditer(word.text):
            # A percent is a rate, not a monetary amount. Do not extract the
            # numeric substring from strings such as "(10.0%)".
            if word.text[m.end():].lstrip().startswith("%"):
                continue
            amount = parse_amount(m.group(0))
            if amount:
                yield word, m.group(0), amount


def joint_assignment(costs: dict[str, dict[tuple, float]]) -> dict[str, tuple]:
    """Exact maximum-cardinality, minimum-cost bipartite assignment.

    Dummy columns permit unmatched fields. Their penalty dominates the sum
    of real edges, so the Hungarian-family solver maximizes cardinality first.
    Ambiguity is still reviewed by the caller.
    """
    import math
    from scipy.optimize import linear_sum_assignment
    names = list(costs)
    if len(names) > 3 or any(len(edges) > 128 for edges in costs.values()):
        raise ValueError("joint assignment candidate limit exceeded")
    values = [value for edges in costs.values() for value in edges.values()]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("joint assignment costs must be finite and nonnegative")
    candidates = sorted({key for edges in costs.values() for key in edges})
    if not names or not candidates:
        return {}
    unmatched = (max(values) + 1) * (len(names) + 1)
    matrix = [[costs[name].get(key, float("inf")) for key in candidates] +
              [unmatched] * len(names) for name in names]
    rows, columns = linear_sum_assignment(matrix)
    return {names[i]: candidates[j] for i, j in zip(rows, columns) if j < len(candidates)}


def extract_fields(words: list[OCRWord], *, image_width: int, image_height: int,
                   source_hash: str, model_revision: str = "tesseract-tsv-unadapted") -> dict:
    """Assign total/subtotal/tax with explicit review reasons and provenance."""
    if image_width <= 0 or image_height <= 0 or len(words) > 100_000:
        raise ValueError("invalid image dimensions or OCR word limit exceeded")
    good = [w for w in words if w.valid_box(image_width, image_height)]
    invalid_count = len(words) - len(good)
    lines = defaultdict(list)
    for word in good:
        lines[(word.page, word.line_id)].append(word)
    labels: dict[str, list[OCRWord]] = {name: [] for name in FIELDS}
    line_texts = {}
    for line_words in lines.values():
        line_words.sort(key=lambda w: (w.left, w.top))
        line_text = " ".join(re.sub(r"[^a-z]+", " ", w.text.lower()).strip()
                              for w in line_words)
        line_text = re.sub(r"\s+", " ", line_text).strip()
        line_texts[(line_words[0].page, line_words[0].line_id)] = line_text
        for field, names in FIELDS.items():
            matched = [name for name in names if re.search(
                r"\b" + re.escape(re.sub(r"[^a-z]+", " ", name.lower()).strip()) + r"\b",
                line_text)]
            if matched:
                label_tokens = {part for phrase in matched for part in phrase.split()}
                labels[field].extend(w for w in line_words
                    if re.sub(r"[^a-z]+", " ", w.text.lower()).strip() in label_tokens)
    parsed = list(_amount_tokens(good))
    fragment_lines = {(w.page, w.line_id) for w in good
                      if re.search(r"\d[.,]\s*$", w.text)}
    field_options = {}
    for field in FIELDS:
        anchors = labels[field]
        proposals = []
        for anchor in anchors:
            same_line = [(w, txt, amount) for w, txt, amount in parsed
                         if w.page == anchor.page and w.line_id == anchor.line_id
                         and w is not anchor]
            explicit_currencies = {_currency_for(w.text) for w in good
                                    if w.page == anchor.page and w.line_id == anchor.line_id
                                    and _currency_for(w.text)}
            # Number in the label itself is only a candidate for total if there
            # is no distinct amount after it; this rule never overrides review.
            for word, text, amount in same_line:
                dx = (word.left + word.width / 2) - (anchor.left + anchor.width / 2)
                dy = abs((word.top + word.height / 2) - (anchor.top + anchor.height / 2))
                if dx >= -2 and dy <= max(anchor.height, word.height) * 1.5:
                    if amount.currency is None and len(explicit_currencies) == 1:
                        currency = next(iter(explicit_currencies))
                        exponent = 0 if currency in ("IDR", "JPY") else 2
                        scaled = Decimal(amount.value) * (Decimal(10) ** exponent)
                        if scaled != scaled.to_integral_value():
                            continue
                        amount = Amount(amount.value, currency, int(scaled), exponent)
                    confidence_penalty = (100 - max(0, min(100, word.confidence))) / 10 if word.confidence is not None else 10
                    proposals.append((word, text, amount, dx + dy * 2 + confidence_penalty))
        # Deduplicate a candidate shared by accidentally repeated label anchors.
        unique = {}
        for proposal in proposals:
            w, text, amount, distance = proposal
            key = (w.page, w.left, w.top, text)
            if key not in unique or distance < unique[key][3]:
                unique[key] = proposal
        options = sorted(unique.values(), key=lambda p: p[3])
        field_options[field] = options
    def candidate_key(proposal):
        word, text, _, _ = proposal
        return (word.page, word.left, word.top, word.width, word.height, text)
    overflow = any(len(options) > 128 for options in field_options.values())
    owners = defaultdict(set)
    for name, options in field_options.items():
        for proposal in options:
            owners[candidate_key(proposal)].add(name)
    assignment = {} if overflow else joint_assignment({
        name: {candidate_key(p): p[3] for p in options}
        for name, options in field_options.items()})
    results = {}
    for field, options in field_options.items():
        anchors = labels[field]
        reasons = []
        if overflow:
            reasons.append("joint_assignment_candidate_limit")
        if any(len(owners[candidate_key(p)]) > 1 for p in options):
            reasons.append("source_candidate_shared_between_fields")
        if any("tax" in line_texts.get((a.page, a.line_id), "")
               and "included" in line_texts.get((a.page, a.line_id), "") for a in anchors):
            reasons.append("tax_included_context_ambiguous")
        if invalid_count:
            reasons.append("invalid_source_boxes_present")
        if not anchors:
            reasons.append("explicit_field_label_missing")
        if not options:
            reasons.append("same_line_amount_missing")
        if len(options) > 1:
            reasons.append("multiple_amount_candidates")
        if len(options) == 1 and options[0][2].currency is None:
            reasons.append("explicit_currency_missing")
        if options and any((p[0].page, p[0].line_id) in fragment_lines for p in options):
            reasons.append("incomplete_numeric_fragment")
        selected = (options[0] if len(options) == 1 and
                    assignment.get(field) == candidate_key(options[0]) else None)
        if options and field not in assignment:
            reasons.append("joint_assignment_unmatched")
        status = "REVIEW" if reasons else "ACCEPT"
        if selected:
            word, original, amount, _ = selected
            if not word.valid_box(image_width, image_height):
                status = "REVIEW"
                reasons.append("selected_box_invalid")
            result = {"value": amount.value, "currency": amount.currency,
                      "minor_units": amount.minor_units, "exponent": amount.exponent,
                      "original_text": original,
                      "source_boxes": [{"left": word.left, "top": word.top,
                                        "width": word.width, "height": word.height}],
                      "page": word.page, "candidate_score": round(1 / (1 + options[0][3]), 6),
                      "status": status,
                      "reason_codes": sorted(set(reasons))}
        else:
            result = {"value": None, "currency": None, "minor_units": None,
                      "exponent": None, "original_text": None, "source_boxes": [],
                      "page": None, "candidate_score": None, "status": status,
                      "reason_codes": sorted(set(reasons))}
        result.update({"source_hash": source_hash, "model_revision": model_revision})
        result["joint_assignment_source"] = list(assignment[field]) if field in assignment else None
        results[field] = result
    payload = {"schema_version": 1, "fields": results}
    payload["result_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def load_tesseract_tsv(tsv: str) -> tuple[list[OCRWord], int, int]:
    """Read word-level TSV; image dimensions are supplied by caller separately."""
    import csv
    from io import StringIO
    reader = csv.DictReader(StringIO(tsv), delimiter="\t")
    required = {"level", "left", "top", "width", "height", "conf", "text", "page_num", "block_num", "par_num", "line_num"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Tesseract TSV missing required columns")
    words = []
    for row in reader:
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        words.append(OCRWord(text=text, left=int(row["left"]), top=int(row["top"]),
                             width=int(row["width"]), height=int(row["height"]),
                             page=int(row["page_num"]),
                             line_id="/".join(row[k] for k in ("block_num", "par_num", "line_num")),
                             confidence=float(row["conf"])))
    return words, len(words), 0
