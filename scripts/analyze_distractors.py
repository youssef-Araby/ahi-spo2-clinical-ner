#!/usr/bin/env python3
"""Per-distractor confusion analysis for NER predictions on the v2 test set.

For each predicted entity, parse its numeric value and classify it as:
  correct      - matches the gold target value for that label
  distractor:X - matches a known distractor value X from the record's meta
  other        - neither (wrong span on a non-distractor number)
Quantifies how often the engineered distractors lure each model.
"""
import json
import re
from pathlib import Path
from collections import Counter

V2 = Path(__file__).resolve().parents[1] / "data" / "v2"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
gold = {r["id"]: r for r in (json.loads(l) for l in open(V2 / "notes_v2_test.jsonl"))}

# distractor columns that share units / look-alikes per target label
DISTR = {
    "AHI": ["odi", "rdi", "rem_ahi", "supine_ahi", "arousal_index"],
    "SPO2_MEAN": ["spo2_nadir", "t90", "heart_rate"],
    "SPO2_NADIR": ["spo2_mean", "t90", "heart_rate"],
}
TOL = {"AHI": 0.06, "SPO2_MEAN": 0.6, "SPO2_NADIR": 0.6}


def val(text):
    m = NUM.search(text or "")
    return float(m.group()) if m else None


def classify(pred_v, label, meta):
    if pred_v is None:
        return "other"
    tol = TOL[label]
    tgt = {"AHI": "ahi", "SPO2_MEAN": "spo2_mean", "SPO2_NADIR": "spo2_nadir"}[label]
    if meta.get(tgt) is not None and abs(pred_v - float(meta[tgt])) <= tol:
        return "correct"
    for d in DISTR[label]:
        if meta.get(d) is not None and meta[d] != "" and abs(pred_v - float(meta[d])) <= tol:
            return f"distractor:{d}"
    return "other"


for model in ["regex", "crf", "clinicalbert"]:
    preds = {p["id"]: p for p in (json.loads(l) for l in open(V2 / f"preds_{model}.jsonl"))}
    cls = Counter()
    distr_hits = Counter()
    by_label = {l: Counter() for l in DISTR}
    for pid, g in gold.items():
        meta = g["meta"]
        for e in preds.get(pid, {}).get("entities", []):
            text = g["text"][e["start"]:e["end"]]
            c = classify(val(text), e["label"], meta)
            cls[c.split(":")[0]] += 1
            by_label[e["label"]][c.split(":")[0]] += 1
            if c.startswith("distractor"):
                distr_hits[f'{e["label"]}<-{c.split(":")[1]}'] += 1
    total = sum(cls.values())
    print(f"\n=== {model} ===  ({total} predicted entities)")
    print(f"  correct={cls['correct']}  distractor-confusions={cls['distractor']}  other-errors={cls['other']}")
    for l in DISTR:
        b = by_label[l]
        print(f"    {l:11} correct={b['correct']:3} distractor={b['distractor']:3} other={b['other']:3}")
    if distr_hits:
        print("  distractor confusions (target <- mistaken-for):")
        for k, v in distr_hits.most_common():
            print(f"    {k}: {v}")
