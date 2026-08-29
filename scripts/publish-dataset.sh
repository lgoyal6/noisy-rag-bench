#!/usr/bin/env bash
# Upload the corpus, the QA set and the benchmark run to Hugging Face.
#
#   hf auth login                          # once
#   scripts/publish-dataset.sh [repo]
#
# The corpus is generated rather than stored, so it is materialised here from the
# same seeded builder the benchmark used. Publishing the generator alone would
# make the dataset unusable to anyone who does not want to run it.
set -euo pipefail

repo="${1:-lgoyal/noisy-rag-bench}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hf="${HF:-hf}"

command -v "$hf" >/dev/null || { echo "install the CLI: pip install -U huggingface_hub" >&2; exit 1; }
"$hf" auth whoami >/dev/null 2>&1 || { echo "log in first: hf auth login" >&2; exit 1; }

staging="$(mktemp -d)"

echo "materialising the corpus from corpus.build_corpus()"
python3 - "$staging" "$here" <<'PY'
import json, sys, pathlib

# Reading from stdin means there is no __file__, so the repository root arrives
# as an argument rather than being derived from the script's own location.
sys.path.insert(0, sys.argv[2])
from corpus import build_corpus

out = pathlib.Path(sys.argv[1])
chunks, questions = build_corpus()
for name, rows in (("corpus.jsonl", chunks), ("questions.jsonl", questions)):
    with open(out / name, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(rows)} rows")
PY

echo "generating the card from results/noise_bench.json"
python3 "$here/scripts/dataset-card.py" > "$staging/README.md"
cp "$here/results/noise_bench.json" "$staging/noise_bench.json"
cp "$here/LICENSE" "$staging/LICENSE"

echo
sed -n '1,44p' "$staging/README.md"
echo
echo "files:"
ls -1 "$staging"
echo
read -r -p "publish this to $repo? [y/N] " reply
[ "$reply" = "y" ] || { echo "stopped"; exit 0; }

"$hf" repo create "$repo" --repo-type dataset --exist-ok
"$hf" upload "$repo" "$staging" . --repo-type dataset \
  --commit-message "Corpus, QA set, and the run the card was generated from"

echo
echo "https://huggingface.co/datasets/$repo"
