#!/usr/bin/env python3
"""Gold-anchored auto-labeling: add NER labels for all in-text features.

Starts from the 3 generation-tagged targets (AHI/SPO2_MEAN/SPO2_NADIR, kept as-is)
and adds spans for the remaining columnar features by locating, for each feature, a
number that BOTH equals the gold value AND sits next to that feature's cue phrase.
Booleans/categoricals (gender, snoring, cpap) are tagged with polarity-encoding labels
derived from the gold value. High precision by construction; we report recall (coverage).

Writes data/v2/notes_v2_multi.jsonl. Does not train anything.
"""
import json
import re
from pathlib import Path
from collections import Counter

V2 = Path(__file__).resolve().parents[1] / "data" / "v2"
notes = [json.loads(l) for l in open(V2 / "notes_v2.jsonl") if l.strip()]
NUM = re.compile(r"-?\d+(?:\.\d+)?")

# numeric features: label -> (gold_key, cue_regex, tolerance)
NUMERIC = {
    "ODI":            ("odi",           r"oxygen desaturation index|\bODI\b|desaturation index", 0.051),
    "RDI":            ("rdi",           r"respiratory disturbance index|\bRDI\b",                  0.051),
    "REM_AHI":        ("rem_ahi",       r"REM[- ]?(?:related )?(?:AHI|apnea[- ]hypopnea index)",   0.051),
    "SUPINE_AHI":     ("supine_ahi",    r"supine[- ]?(?:AHI|apnea[- ]hypopnea index)",             0.051),
    "AROUSAL_INDEX":  ("arousal_index", r"arousal index",                                          0.051),
    "T90":            ("t90",           r"T90|below 90%|<\s?90%|under 90%",                         0.051),
    "AGE":            ("age",           r"(\d+)[- ]year[- ]old",                                    0.0),
    "BMI":            ("bmi",           r"\bBMI\b|body[- ]mass index",                              0.051),
    "ESS":            ("ess",           r"Epworth|\bESS\b",                                         0.0),
    "HEART_RATE":     ("heart_rate",    r"heart rate|\bHR\b",                                       0.0),
}


def find_near(text, cue_m, gold, tol):
    """Find a number == gold within a window around the cue; return (start,end) of the number."""
    a = max(0, cue_m.start() - 45)
    b = min(len(text), cue_m.end() + 70)
    best = None
    for nm in NUM.finditer(text, a, b):
        try:
            v = float(nm.group())
        except ValueError:
            continue
        if abs(v - gold) <= tol:
            # prefer the number closest after the cue
            dist = abs(nm.start() - cue_m.end())
            if best is None or dist < best[0]:
                best = (dist, nm.start(), nm.end())
    return (best[1], best[2]) if best else None


def overlaps(s, e, ents):
    return any(not (e <= x["start"] or s >= x["end"]) for x in ents)


cover = Counter()
total = len(notes)
for r in notes:
    text = r["text"]
    meta = r["meta"]
    ents = list(r["entities"])  # keep the 3 gold targets

    # numeric features
    for label, (key, cue, tol) in NUMERIC.items():
        gold = meta.get(key)
        if gold in (None, ""):
            continue
        gold = float(gold)
        hit = None
        for cm in re.finditer(cue, text, re.I):
            span = find_near(text, cm, gold, tol)
            if span and not overlaps(span[0], span[1], ents):
                hit = span
                break
        if hit:
            ents.append({"start": hit[0], "end": hit[1], "label": label,
                         "text": text[hit[0]:hit[1]]})
            cover[label] += 1

    # gender: tag the first word whose value matches the gold gender
    gmap = {"male": "male", "man": "male", "female": "female", "woman": "female"}
    gold_g = str(meta.get("gender", "")).lower()
    for gm in re.finditer(r"\b(male|female|man|woman)\b", text, re.I):
        if gmap[gm.group(1).lower()] == gold_g:
            ents.append({"start": gm.start(), "end": gm.end(), "label": "GENDER",
                         "text": text[gm.start():gm.end()]})
            cover["GENDER"] += 1
            break

    # snoring polarity (use gold to pick label; tag the cue word)
    sm = re.search(r"snor\w*", text, re.I)
    if sm:
        win = text[max(0, sm.start() - 35):sm.start()].lower()
        neg = any(w in win for w in ["denie", "deny", "no ", "not ", "without", "absent"])
        gold_snore = str(meta.get("snoring", "")).lower() == "true"
        label = "SNORING_PRESENT" if gold_snore else "SNORING_ABSENT"
        ents.append({"start": sm.start(), "end": sm.end(), "label": label,
                     "text": text[sm.start():sm.end()]})
        cover[label] += 1

    # cpap polarity (abbreviation OR spelled-out phrase)
    cm = re.search(r"\bC?PAP\b|positive[- ]airway[- ]pressure", text, re.I)
    if cm:
        gold_cpap = str(meta.get("cpap_recommended", "")).lower() == "true"
        label = "CPAP_YES" if gold_cpap else "CPAP_NO"
        ents.append({"start": cm.start(), "end": cm.end(), "label": label,
                     "text": text[cm.start():cm.end()]})
        cover[label] += 1

    # verify all spans round-trip
    for e in ents:
        assert text[e["start"]:e["end"]] == e["text"], (r["id"], e)
    r["entities"] = sorted(ents, key=lambda x: x["start"])

with open(V2 / "notes_v2_multi.jsonl", "w") as f:
    for r in notes:
        f.write(json.dumps(r) + "\n")

print(f"records: {total}\n")
print(f"{'label':16}{'coverage':>10}{'%':>7}")
for label in list(NUMERIC) + ["GENDER"]:
    print(f"  {label:14}{cover[label]:>10}{100*cover[label]//total:>6}%")
sp = cover["SNORING_PRESENT"] + cover["SNORING_ABSENT"]
cp = cover["CPAP_YES"] + cover["CPAP_NO"]
print(f"  {'SNORING':14}{sp:>10}{100*sp//total:>6}%   (present {cover['SNORING_PRESENT']} / absent {cover['SNORING_ABSENT']})")
print(f"  {'CPAP':14}{cp:>10}{100*cp//total:>6}%   (yes {cover['CPAP_YES']} / no {cover['CPAP_NO']})")
avg_ent = sum(len(r["entities"]) for r in notes) / total
print(f"\navg entities/record: {avg_ent:.1f}  (was 3.0)")
