#!/usr/bin/env python3
"""Severity classifier on the v2 columnar data.

Three feature conditions, because many columns are near-deterministic functions
of AHI (the label source) and would leak the label:
  1. with_ahi        - includes AHI itself (tautology sanity ceiling)
  2. proxies_no_ahi  - drops AHI, keeps /hr respiratory proxies (ODI/RDI/REM/supine/arousal)
  3. physiology_only - drops AHI and ALL /hr proxies; oximetry + demographics only (hardest)

Reports macro-F1, balanced accuracy, quadratic-weighted kappa (ordinal), per-class
F1, confusion matrix, and stratified accuracy on the hidden difficulty subsets.
"""
import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
ORDER = ["None", "Mild", "Moderate", "Severe"]
OIDX = {s: i for i, s in enumerate(ORDER)}

ALL_NUM = ["ahi", "odi", "spo2_mean", "spo2_nadir", "t90", "age", "bmi", "ess",
           "arousal_index", "rdi", "rem_ahi", "supine_ahi", "heart_rate"]
PROXIES = ["odi", "rdi", "rem_ahi", "supine_ahi", "arousal_index"]
CONDITIONS = {
    "with_ahi": ALL_NUM,
    "proxies_no_ahi": [c for c in ALL_NUM if c != "ahi"],
    "physiology_only": [c for c in ALL_NUM if c not in (["ahi"] + PROXIES)],
}


def load():
    splits = json.load(open(V2 / "splits.json"))
    rows = list(csv.DictReader(open(V2 / "columnar_v2.csv")))
    data = {"train": [], "val": [], "test": []}
    for r in rows:
        s = splits.get(r["patient_id"])
        if s:
            data[s].append(r)
    return data


def to_float(r, k):
    v = r.get(k, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def build(rows, feats):
    X = np.array([[to_float(r, f) for f in feats]
                  + [1.0 if r["gender"] == "male" else 0.0,
                     1.0 if str(r["snoring"]).lower() == "true" else 0.0] for r in rows])
    y = np.array([OIDX[r["severity"]] for r in rows])
    return X, y


def fillmed(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    for X in (Xtr, Xte):
        idx = np.where(np.isnan(X))
        X[idx] = np.take(med, idx[1])
    return Xtr, Xte


def evaluate(ytrue, ypred):
    return {
        "macro_f1": f1_score(ytrue, ypred, average="macro", labels=[0, 1, 2, 3], zero_division=0),
        "bal_acc": balanced_accuracy_score(ytrue, ypred),
        "qwk": cohen_kappa_score(ytrue, ypred, weights="quadratic", labels=[0, 1, 2, 3]),
        "per_class_f1": f1_score(ytrue, ypred, average=None, labels=[0, 1, 2, 3], zero_division=0),
        "cm": confusion_matrix(ytrue, ypred, labels=[0, 1, 2, 3]),
    }


def main():
    data = load()
    test_rows = data["test"]
    print(f"train {len(data['train'])}  test {len(test_rows)}\n")
    out = []
    best = {}
    for cond, feats in CONDITIONS.items():
        Xtr, ytr = build(data["train"], feats)
        Xte, yte = build(test_rows, feats)
        Xtr, Xte = fillmed(Xtr, Xte)
        line = f"### Condition: {cond}  ({len(feats)+2} features)\n"
        print(line.rstrip())
        for name, clf in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                          ("rforest", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42))]:
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)
            m = evaluate(yte, yp)
            pcf = " ".join(f"{ORDER[i][:4]}={m['per_class_f1'][i]:.2f}" for i in range(4))
            print(f"  {name:8} macroF1={m['macro_f1']:.3f}  balAcc={m['bal_acc']:.3f}  QWK={m['qwk']:.3f}  | {pcf}")
            out.append((cond, name, m))
            if name == "rforest":
                best[cond] = (clf, feats, yp)
        print()

    # stratified accuracy by difficulty subset for the two no-AHI conditions (rforest)
    diff_of = {r["patient_id"]: (r.get("difficulty_flag") or "clean").split("+")[0] for r in test_rows}
    print("### Stratified test accuracy by difficulty subset (rforest)\n")
    for cond in ("proxies_no_ahi", "physiology_only"):
        clf, feats, _ = best[cond]
        Xte, yte = build(test_rows, feats)
        _, Xte = fillmed(build(data["train"], feats)[0], Xte)
        yp = clf.predict(Xte)
        groups = defaultdict(lambda: [0, 0])
        for i, r in enumerate(test_rows):
            g = diff_of[r["patient_id"]]
            groups[g][1] += 1
            groups[g][0] += int(yp[i] == yte[i])
        print(f"  [{cond}]")
        for g, (ok, n) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
            print(f"    {g:22} acc={ok/n:.2f}  (n={n})")
        print()

    # confusion matrix for the hardest condition
    cm = [m for c, nm, m in out if c == "physiology_only" and nm == "rforest"][0]["cm"]
    print("### Confusion matrix - physiology_only / rforest (rows=true, cols=pred; None/Mild/Mod/Sev)")
    for i, row in enumerate(cm):
        print(f"  {ORDER[i][:4]:5} " + " ".join(f"{v:3d}" for v in row))


if __name__ == "__main__":
    main()
