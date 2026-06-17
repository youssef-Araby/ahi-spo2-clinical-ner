#!/usr/bin/env python3
"""End-to-end pipeline eval: report -> NER-extracted AHI -> AASM severity.

Severity is defined as AASM(AHI), so the severity step is deterministic AASM
thresholding on whatever AHI the NER model extracted. Compares each NER model's
end-to-end severity accuracy against the ceiling (gold AHI -> severity = 100%).
"""
import json
import re
from pathlib import Path
from sklearn.metrics import f1_score

V2 = Path(__file__).resolve().parents[1] / "data" / "v2"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
ORDER = ["None", "Mild", "Moderate", "Severe"]
OIDX = {s: i for i, s in enumerate(ORDER)}
gold = {r["id"]: r for r in (json.loads(l) for l in open(V2 / "notes_v2_test.jsonl"))}


def aasm(a):
    return "None" if a < 5 else "Mild" if a < 15 else "Moderate" if a <= 30 else "Severe"


def first_ahi(entities, text):
    for e in entities:
        if e["label"] == "AHI":
            m = NUM.search(text[e["start"]:e["end"]])
            if m:
                return float(m.group())
    return None


print(f"test notes: {len(gold)}\n")
print(f"{'pipeline':28}{'cover%':>8}{'sevAcc(all)':>12}{'sevAcc(cov)':>12}{'macroF1(cov)':>13}")

# ceiling: gold AHI -> severity (definitional 100%)
yt = [OIDX[gold[i]["meta"]["severity"]] for i in gold]
print(f"{'CEILING: gold AHI->AASM':28}{100.0:>8.1f}{1.000:>12.3f}{1.000:>12.3f}{1.000:>13.3f}")

for model in ["regex", "crf", "clinicalbert"]:
    preds = {p["id"]: p for p in (json.loads(l) for l in open(V2 / f"preds_{model}.jsonl"))}
    covered_t, covered_p, n_cover, n_correct_all = [], [], 0, 0
    for pid, g in gold.items():
        gsev = g["meta"]["severity"]
        pa = first_ahi(preds.get(pid, {}).get("entities", []), g["text"])
        if pa is None:
            continue  # miss: counts wrong in all-notes accuracy
        n_cover += 1
        psev = aasm(pa)
        covered_t.append(OIDX[gsev]); covered_p.append(OIDX[psev])
        n_correct_all += int(psev == gsev)
    n = len(gold)
    acc_all = n_correct_all / n
    acc_cov = sum(int(a == b) for a, b in zip(covered_t, covered_p)) / max(1, n_cover)
    mf1 = f1_score(covered_t, covered_p, average="macro", labels=[0, 1, 2, 3], zero_division=0)
    print(f"{('END2END: '+model+'->AASM'):28}{100*n_cover/n:>8.1f}{acc_all:>12.3f}{acc_cov:>12.3f}{mf1:>13.3f}")
