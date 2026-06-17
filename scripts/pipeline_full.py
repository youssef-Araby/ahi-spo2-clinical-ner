#!/usr/bin/env python3
"""Full-feature classifier pipeline.

Ceiling: RandomForest on ALL gold columnar features -> severity.
End-to-end: the 3 NER-extractable measurements (AHI, mean, nadir) are replaced by
what each NER model extracted from the report; the remaining structured features
stay gold. Also reports a 3-feature-only ablation (AHI/mean/nadir alone), where NER
error actually shows, since there are no gold proxies to carry the label.
"""
import json
import re
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

V2 = Path(__file__).resolve().parents[1] / "data" / "v2"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
ORDER = ["None", "Mild", "Moderate", "Severe"]
OIDX = {s: i for i, s in enumerate(ORDER)}
NUMF = ["ahi", "odi", "spo2_mean", "spo2_nadir", "t90", "age", "bmi", "ess",
        "arousal_index", "rdi", "rem_ahi", "supine_ahi", "heart_rate"]
NER_FEATS = ["ahi", "spo2_mean", "spo2_nadir"]

import csv
splits = json.load(open(V2 / "splits.json"))
rows = {r["patient_id"]: r for r in csv.DictReader(open(V2 / "columnar_v2.csv"))}
gold_notes = {r["id"]: r for r in (json.loads(l) for l in open(V2 / "notes_v2_test.jsonl"))}
train_ids = [i for i, s in splits.items() if s == "train"]
test_ids = [i for i, s in splits.items() if s == "test"]


def f(r, k):
    try:
        return float(r[k])
    except (ValueError, TypeError, KeyError):
        return np.nan


def featrow(r, feats, override=None):
    vals = []
    for k in feats:
        if override and k in override:
            vals.append(override[k] if override[k] is not None else np.nan)
        else:
            vals.append(f(r, k))
    if feats is NUMF:
        vals.append(1.0 if r["gender"] == "male" else 0.0)
        vals.append(1.0 if str(r["snoring"]).lower() == "true" else 0.0)
    return vals


def matrix(ids, feats, overrides=None):
    X = np.array([featrow(rows[i], feats, (overrides or {}).get(i)) for i in ids], float)
    y = np.array([OIDX[rows[i]["severity"]] for i in ids])
    return X, y


def med_impute(Xtr, X):
    med = np.nanmedian(Xtr, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    return X


def ner_ahi_mean_nadir(model):
    preds = {p["id"]: p for p in (json.loads(l) for l in open(V2 / f"preds_{model}.jsonl"))}
    ext = {}
    cover = 0
    for pid in test_ids:
        g = gold_notes[pid]
        got = {}
        for lbl, key in [("AHI", "ahi"), ("SPO2_MEAN", "spo2_mean"), ("SPO2_NADIR", "spo2_nadir")]:
            v = None
            for e in preds.get(pid, {}).get("entities", []):
                if e["label"] == lbl:
                    m = NUM.search(g["text"][e["start"]:e["end"]])
                    if m:
                        v = float(m.group()); break
            got[key] = v
        if got["ahi"] is not None:
            cover += 1
        ext[pid] = got
    return ext, cover


def report(name, feats):
    Xtr, ytr = matrix(train_ids, feats)
    Xtr = med_impute(Xtr.copy(), Xtr)
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42).fit(Xtr, ytr)
    Xte, yte = matrix(test_ids, feats)
    Xte = med_impute(Xtr, Xte.copy())
    yp = clf.predict(Xte)
    print(f"  CEILING (gold {name:13}) acc={accuracy_score(yte, yp):.3f}  macroF1={f1_score(yte, yp, average='macro', labels=[0,1,2,3], zero_division=0):.3f}")
    for model in ["regex", "crf", "clinicalbert"]:
        ext, cover = ner_ahi_mean_nadir(model)
        Xp, _ = matrix(test_ids, feats, overrides=ext)
        Xp = med_impute(Xtr, Xp.copy())
        yp = clf.predict(Xp)
        print(f"    {model:12} acc={accuracy_score(yte, yp):.3f}  macroF1={f1_score(yte, yp, average='macro', labels=[0,1,2,3], zero_division=0):.3f}  (AHI coverage {100*cover/len(test_ids):.0f}%)")


print(f"train {len(train_ids)}  test {len(test_ids)}\n")
print("=== FULL feature set (3 from NER, rest gold/structured) ===")
report("ALL 15 feat", NUMF)
print("\n=== 3-feature ablation (AHI+mean+nadir ONLY: no gold proxies) ===")
report("AHI/mean/nadir", NER_FEATS)
