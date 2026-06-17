#!/usr/bin/env python3
"""Stratified 70/15/15 split of the v2 dataset by severity.

Writes notes_v2_{train,val,test}.jsonl and splits.json ({id: split}) under data/v2/.
Round-robin assignment within each severity class spreads the difficulty types
(which rotate with id) evenly across splits. Deterministic (sorted by id).
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
notes = [json.loads(l) for l in open(V2 / "notes_v2.jsonl") if l.strip()]

by_sev = defaultdict(list)
for n in sorted(notes, key=lambda r: r["id"]):
    by_sev[n["meta"]["severity"]].append(n)

# round-robin buckets out of 20: 0-13 train, 14-16 val, 17-19 test
def bucket(i):
    m = i % 20
    return "train" if m < 14 else ("val" if m < 17 else "test")

splits = {"train": [], "val": [], "test": []}
assign = {}
for sev, recs in by_sev.items():
    for i, n in enumerate(recs):
        s = bucket(i)
        splits[s].append(n)
        assign[n["id"]] = s

for s, recs in splits.items():
    with open(V2 / f"notes_v2_{s}.jsonl", "w") as f:
        for n in recs:
            f.write(json.dumps(n) + "\n")
json.dump(assign, open(V2 / "splits.json", "w"), indent=0)

print(f"total {len(notes)} -> train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")
for s in ("train", "val", "test"):
    c = Counter(n["meta"]["severity"] for n in splits[s])
    print(f"  {s:6} severity: {dict(c)}")
print("difficulty (all):", dict(Counter(
    (n["meta"].get("difficulty_flag") or "clean").split("+")[0] for n in notes)))
