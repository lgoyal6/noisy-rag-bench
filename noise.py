"""Document-noise transforms: what a scanned, redacted, page-numbered PDF does to text.

Five families, applied to the *corpus* only. Queries stay clean, because the lawyer
types the question correctly; it is the archive that is damaged. That asymmetry is
the realistic one and it is what makes the curves interpretable.
"""
import random
import re

# OCR confusion pairs seen in real Tesseract output on 300dpi scans of legal paper.
OCR_PAIRS = [
    ("rn", "m"), ("m", "rn"), ("cl", "d"), ("d", "cl"), ("vv", "w"), ("w", "vv"),
    ("0", "O"), ("O", "0"), ("l", "1"), ("1", "l"), ("I", "l"), ("S", "5"),
    ("5", "S"), ("B", "8"), ("8", "B"), ("G", "6"), ("6", "G"), ("ll", "II"),
    ("fi", "n"), ("ti", "ti"), (".", ","), (",", "."), ("g", "q"), ("q", "g"),
]
_MULTI = sorted({a for a, _ in OCR_PAIRS if len(a) > 1}, key=len, reverse=True)
_SINGLE = {a: b for a, b in OCR_PAIRS if len(a) == 1}
_MULTIMAP = {a: b for a, b in OCR_PAIRS if len(a) > 1}

RUNNING_HEADERS = [
    "CONFIDENTIAL - ATTORNEY WORK PRODUCT",
    "EXECUTION VERSION",
    "Page {p} of {n}",
    "PRIVILEGED AND CONFIDENTIAL",
    "DRAFT - SUBJECT TO REVIEW - DO NOT CIRCULATE",
    "{n}-{p} Rev. 4",
    "Doc ID: 8841-{p}{n} / Bates SRN{n}{p}",
    "[CONTINUED ON FOLLOWING PAGE]",
]


def ocr_noise(text, rate, rng):
    """Per-character-position substitution using OCR confusion pairs."""
    out, i = [], 0
    while i < len(text):
        hit = False
        if rng.random() < rate:
            for m in _MULTI:
                if text.startswith(m, i):
                    out.append(_MULTIMAP[m])
                    i += len(m)
                    hit = True
                    break
            if not hit and text[i] in _SINGLE:
                out.append(_SINGLE[text[i]])
                i += 1
                hit = True
        if not hit:
            out.append(text[i])
            i += 1
    return "".join(out)


def whitespace_noise(text, rate, rng):
    """Drop spaces (words fuse) and double them (words split). Both happen on scans."""
    out = []
    for ch in text:
        if ch == " " and rng.random() < rate:
            if rng.random() < 0.5:
                continue                      # dropped -> "thefollowing"
            out.append("  ")                  # doubled
            continue
        if ch != " " and rng.random() < rate * 0.35:
            out.append(ch)
            out.append(" ")                   # spurious split inside a word
            continue
        out.append(ch)
    return "".join(out)


def hyphenation_noise(text, rate, rng):
    """Break words across line ends with a soft hyphen that never got rejoined."""
    words = text.split(" ")
    out = []
    for w in words:
        if len(w) >= 7 and rng.random() < rate * 6:
            cut = rng.randint(3, len(w) - 3)
            out.append(w[:cut] + "-\n" + w[cut:])
        else:
            out.append(w)
    return " ".join(out)


def header_noise(text, rate, rng):
    """Inject running headers / footers / Bates stamps into the chunk body.

    `rate` here is the expected number of injections per chunk.
    """
    sents = re.split(r"(?<=\. )", text)
    n_inject = int(rate) + (1 if rng.random() < (rate - int(rate)) else 0)
    for _ in range(n_inject):
        hdr = rng.choice(RUNNING_HEADERS).format(p=rng.randint(1, 400),
                                                 n=rng.randint(1, 900))
        pos = rng.randint(0, len(sents))
        sents.insert(pos, "\n" + hdr + "\n")
    return "".join(sents)


def redaction_noise(text, rate, rng):
    """Blackouts that remove spans outright. Nothing marks where the hole was."""
    words = text.split(" ")
    out, i = [], 0
    while i < len(words):
        if rng.random() < rate:
            span = rng.randint(1, 4)
            i += span                          # span vanishes
            continue
        out.append(words[i])
        i += 1
    return " ".join(out)


FAMILIES = {
    "ocr": ocr_noise,
    "whitespace": whitespace_noise,
    "hyphenation": hyphenation_noise,
    "header": header_noise,
    "redaction": redaction_noise,
}

# (condition label, {family: rate}). Knob rates are calibrated so that the *measured*
# character error rate lands near the label, since CER is the number an OCR vendor
# actually reports and the only x-axis that transfers to somebody else's documents.
CONDITIONS = [
    ("clean", {}),
    ("ocr cer~1%", {"ocr": 0.035}),
    ("ocr cer~2%", {"ocr": 0.075}),
    ("ocr cer~5%", {"ocr": 0.19}),
    ("ocr cer~10%", {"ocr": 0.40}),
    ("whitespace cer~1%", {"whitespace": 0.015}),
    ("whitespace cer~3%", {"whitespace": 0.055}),
    ("whitespace cer~6%", {"whitespace": 0.11}),
    ("hyphen cer~1%", {"hyphenation": 0.017}),
    ("hyphen cer~3%", {"hyphenation": 0.05}),
    ("hyphen cer~6%", {"hyphenation": 0.10}),
    ("header x1", {"header": 1.0}),
    ("header x3", {"header": 3.0}),
    ("redaction 2% words", {"redaction": 0.02}),
    ("redaction 5% words", {"redaction": 0.05}),
    ("redaction 10% words", {"redaction": 0.10}),
    # what a real scan of 1990s paper, run through a redaction pass, looks like
    ("scan-realistic", {"ocr": 0.075, "whitespace": 0.03, "hyphenation": 0.04,
                        "header": 1.0, "redaction": 0.02}),
    ("scan-degraded", {"ocr": 0.19, "whitespace": 0.06, "hyphenation": 0.08,
                       "header": 2.0, "redaction": 0.05}),
]


def corrupt_chunks(chunks, spec, seed=7):
    """Apply a noise spec to every indexed field of every chunk."""
    rng = random.Random(seed)
    out = []
    for c in chunks:
        rec = dict(c)
        for field in ("doc_title", "section", "text"):
            t = c[field]
            for fam in ("redaction", "hyphenation", "whitespace", "ocr", "header"):
                if fam in spec:
                    # headers only contaminate the body, not the title metadata
                    if fam == "header" and field != "text":
                        continue
                    t = FAMILIES[fam](t, spec[fam], rng)
            rec[field] = t
        rec["indexed"] = f"{rec['doc_title']} {rec['section']} {rec['text']}"
        out.append(rec)
    return out


def char_error_rate(clean, noisy):
    """Levenshtein / len(clean), the standard OCR quality number, so the x-axis is
    comparable to whatever a customer's own OCR vendor reports."""
    if not clean:
        return 0.0
    prev = list(range(len(noisy) + 1))
    for i, cc in enumerate(clean, 1):
        cur = [i]
        for j, nc in enumerate(noisy, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cc != nc)))
        prev = cur
    return prev[-1] / len(clean)


if __name__ == "__main__":
    from corpus import build_corpus
    ch, _ = build_corpus()
    for label, spec in CONDITIONS:
        n = corrupt_chunks(ch[:3], spec)
        cer = sum(char_error_rate(a["text"], b["text"]) for a, b in zip(ch[:3], n)) / 3
        print(f"{label:18s} CER={cer:6.3f}  {n[0]['text'][:90]!r}")
