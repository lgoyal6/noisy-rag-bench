"""Copy the modules the results page runs into docs/.

The page applies the noise itself, on text a reader types, so the functions
doing the damage have to be the ones the benchmark used. Copying them verbatim
rather than porting is what stops the demo and the measurement drifting apart.

    python3 scripts/make_page_data.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data"

# noise.py is pure stdlib. retrieval.py imports numpy for its dense retriever,
# which pyodide carries, and the page only uses its canon/answer_present pair.
MODULES = ("noise.py", "retrieval.py")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        (OUT / name).write_text((ROOT / name).read_text())
        print(f"docs/data/{name}")


if __name__ == "__main__":
    main()
