"""
Regex / rule-based baseline (related work #1: keyword-driven extraction).

For each note it looks for a number anchored to a label-specific cue:
  AHI         - "AHI", "apnea-hypopnea index", "AH index"
  SPO2_MEAN   - "mean/average ... saturation/SpO2", "saturation averaged"
  SPO2_NADIR  - "nadir", "lowest oxygen saturation", "down to", "low of", "dipping to"

It takes the first match per label and records the character span of the number,
which is exactly the prediction format the evaluation harness expects.

Usage:
    python src/baseline_regex.py                 # score on the test split
    python src/baseline_regex.py --split val      # or train / val
    python src/baseline_regex.py --save preds.jsonl
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_jsonl, evaluate, print_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Each label: ordered list of patterns; group(1) is the value. Case-insensitive.
PATTERNS = {
    "AHI": [
        r"apnea[-/]hypopnea index\D{0,8}(\d+(?:\.\d+)?)",
        r"\bAH index\b\D{0,6}(\d+(?:\.\d+)?)",
        r"\bAHI\b\D{0,6}(\d+(?:\.\d+)?)",
    ],
    "SPO2_MEAN": [
        r"mean (?:oxygen )?(?:saturation|SpO2)\D{0,6}(\d+)",
        r"average (?:oxygen )?(?:saturation|SpO2)\D{0,6}(\d+)",
        r"(?:oxygen )?saturation averaged\D{0,4}(\d+)",
        r"mean (?:saturation )?of\D{0,4}(\d+)\s*%",
    ],
    "SPO2_NADIR": [
        r"nadir(?:\s+of)?\D{0,6}(\d+)",
        r"lowest (?:oxygen )?(?:saturation|oxygen level)\D{0,14}(\d+)",
        r"desaturation\b.{0,18}?down to\D{0,4}(\d+)",
        r"down to\D{0,4}(\d+)\s*%",
        r"(?:dropped to a )?low of\D{0,4}(\d+)\s*%",
        r"dipping to\D{0,4}(\d+)\s*%",
    ],
}
COMPILED = {lab: [re.compile(p, re.IGNORECASE) for p in pats] for lab, pats in PATTERNS.items()}


def extract(text: str) -> list[dict]:
    """Return entities [{start,end,label}] for one note (first match per label)."""
    ents = []
    for label, regexes in COMPILED.items():
        best = None  # earliest match across this label's patterns
        for rx in regexes:
            m = rx.search(text)
            if m and (best is None or m.start(1) < best.start(1)):
                best = m
        if best is not None:
            ents.append({"start": best.start(1), "end": best.end(1), "label": label})
    return ents


def predict(records: list[dict]) -> list[dict]:
    return [{"id": r["id"], "entities": extract(r["text"])} for r in records]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--gold", help="evaluate on this JSONL instead of a standard split")
    ap.add_argument("--save", help="optional path to write predictions JSONL")
    args = ap.parse_args()

    gold_path = args.gold or ROOT / "data" / f"notes_{args.split}.jsonl"
    gold = load_jsonl(gold_path)
    preds = predict(gold)
    if args.save:
        import json
        Path(args.save).write_text("\n".join(json.dumps(p) for p in preds))
        print(f"wrote {len(preds)} predictions to {args.save}")
    print_report(evaluate(gold, preds), f"Regex baseline — {args.gold or args.split}")


if __name__ == "__main__":
    main()
