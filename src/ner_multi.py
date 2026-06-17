#!/usr/bin/env python3
"""Multi-entity NER + true end-to-end pipeline on the v2 dataset.

Trains a windowed Bio_ClinicalBERT token classifier on the full auto-labeled entity
set (16 types), reports per-entity strict span F1, then runs the end-to-end pipeline:
every classifier input feature comes from what NER extracted (nothing gold at
inference) -> severity. Compares against the gold-feature ceiling.

Self-contained (does not touch the v1 3-label harness).
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, set_seed)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAXLEN, STRIDE = 384, 128
NUM = re.compile(r"-?\d+(?:\.\d+)?")
set_seed(42)

splits = json.load(open(V2 / "splits.json"))
notes = {r["id"]: r for r in (json.loads(l) for l in open(V2 / "notes_v2_multi.jsonl"))}
train = [notes[i] for i in splits if splits[i] == "train"]
test = [notes[i] for i in splits if splits[i] == "test"]

# dynamic label set from training data
ENT = sorted({e["label"] for r in train for e in r["entities"]})
LABELS = ["O"] + [f"{p}-{l}" for l in ENT for p in ("B", "I")]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}
print(f"train {len(train)}  test {len(test)}  |  {len(ENT)} entity types: {ENT}\n")

tok = AutoTokenizer.from_pretrained(MODEL)


def windows(text):
    enc = tok(text, truncation=True, max_length=MAXLEN, stride=STRIDE,
              return_overflowing_tokens=True, return_offsets_mapping=True)
    for w in range(len(enc["input_ids"])):
        yield enc["input_ids"][w], enc["attention_mask"][w], enc["offset_mapping"][w]


def labels_for(offsets, ents):
    out = []
    for (s, e) in offsets:
        if s == e:
            out.append(-100); continue
        lab = "O"
        for ent in ents:
            if s >= ent["start"] and e <= ent["end"] and e > ent["start"]:
                lab = ("B-" if s == ent["start"] else "I-") + ent["label"]
                break
        out.append(L2I.get(lab, 0))
    return out


class DS(torch.utils.data.Dataset):
    def __init__(self, recs):
        self.items = []
        for r in recs:
            for ids, mask, offs in windows(r["text"]):
                self.items.append({"input_ids": ids, "attention_mask": mask,
                                   "labels": labels_for(offs, r["entities"])})

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def predict_spans(recs, model, device):
    model.eval()
    out = {}
    for r in recs:
        found = set()
        for ids, mask, off in windows(r["text"]):
            ti = torch.tensor([ids], device=device)
            tm = torch.tensor([mask], device=device)
            with torch.no_grad():
                tags = model(input_ids=ti, attention_mask=tm).logits[0].argmax(-1).tolist()
            labs = [I2L[t] for t in tags]
            i = 0
            while i < len(labs):
                if labs[i].startswith("B-") and off[i] != [0, 0]:
                    lab = labs[i][2:]
                    s, e, j = off[i][0], off[i][1], i + 1
                    while j < len(labs) and labs[j] == "I-" + lab:
                        e = off[j][1]; j += 1
                    found.add((s, e, lab)); i = j
                else:
                    i += 1
        out[r["id"]] = found
    return out


# ---- train ----
model = AutoModelForTokenClassification.from_pretrained(
    MODEL, num_labels=len(LABELS), id2label=I2L, label2id=L2I)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
targs = TrainingArguments(output_dir=str(ROOT / "outputs" / "ner_multi"),
                          num_train_epochs=8, per_device_train_batch_size=16,
                          learning_rate=3e-5, weight_decay=0.01, save_strategy="no",
                          logging_steps=100, report_to=[], disable_tqdm=True, seed=42)
Trainer(model=model, args=targs, train_dataset=DS(train),
        data_collator=DataCollatorForTokenClassification(tok)).train()

preds = predict_spans(test, model, device)

# ---- per-entity strict span F1 ----
tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
for r in test:
    g = {(e["start"], e["end"], e["label"]) for e in r["entities"]}
    p = preds[r["id"]]
    for lab in ENT:
        gl = {x for x in g if x[2] == lab}; pl = {x for x in p if x[2] == lab}
        tp[lab] += len(gl & pl); fp[lab] += len(pl - gl); fn[lab] += len(gl - pl)

print("\n### Per-entity strict span F1 (v2 multi-label NER, test)")
print(f"  {'entity':16}{'P':>7}{'R':>7}{'F1':>7}{'supp':>7}")
f1s = []
for lab in ENT:
    P = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else 0
    R = tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else 0
    F = 2 * P * R / (P + R) if P + R else 0
    f1s.append(F)
    print(f"  {lab:16}{P:>7.3f}{R:>7.3f}{F:>7.3f}{tp[lab]+fn[lab]:>7}")
TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
mp, mr = TP / (TP + FP), TP / (TP + FN)
print(f"  {'micro':16}{mp:>7.3f}{mr:>7.3f}{2*mp*mr/(mp+mr):>7.3f}")
print(f"  macro-F1: {np.mean(f1s):.3f}")

# ---- note-level exact match (strict: ALL targets correct in a note) ----
NUMTOL = {"AHI": 0.06, "SPO2_MEAN": 0.6, "SPO2_NADIR": 0.6}


def note_exact(label_subset=None):
    span_ok = val_ok = 0
    for r in test:
        G = [e for e in r["entities"] if (label_subset is None or e["label"] in label_subset)]
        P = preds[r["id"]]
        gset = {(e["start"], e["end"], e["label"]) for e in G}
        span_ok += int(gset.issubset(P))
        predval = {}
        for (s, e, lab) in sorted(P):
            if lab not in predval:
                mm = NUM.search(r["text"][s:e])
                predval[lab] = mm.group() if mm else r["text"][s:e]
        ok = True
        for e in G:
            lab = e["label"]
            if lab not in predval:
                ok = False; break
            gm = NUM.search(e["text"])
            if gm:  # numeric label: compare values within tolerance
                try:
                    if abs(float(predval[lab]) - float(gm.group())) > NUMTOL.get(lab, 0.051):
                        ok = False; break
                except ValueError:
                    ok = False; break
            # categorical (GENDER/SNORING_*/CPAP_*): same-label presence already = value match
        val_ok += int(ok)
    return span_ok / len(test), val_ok / len(test)


core3 = {"AHI", "SPO2_MEAN", "SPO2_NADIR"}
s_c3, v_c3 = note_exact(core3)
s_all, v_all = note_exact()
print("\n### Note-level exact match (the HARD metric: every target in the note correct)")
print(f"  core 3 (AHI/mean/nadir): span-exact={s_c3:.3f}  value-exact={v_c3:.3f}")
print(f"  ALL 16 entity types:     span-exact={s_all:.3f}  value-exact={v_all:.3f}")

# ---- end-to-end: extract every feature from NER, classify severity ----
ORDER = ["None", "Mild", "Moderate", "Severe"]
OIDX = {s: i for i, s in enumerate(ORDER)}
FEATS = ["ahi", "odi", "spo2_mean", "spo2_nadir", "t90", "age", "bmi", "ess",
         "arousal_index", "rdi", "rem_ahi", "supine_ahi", "heart_rate", "gender", "snoring", "cpap"]
LAB2FEAT = {"AHI": "ahi", "ODI": "odi", "SPO2_MEAN": "spo2_mean", "SPO2_NADIR": "spo2_nadir",
            "T90": "t90", "AGE": "age", "BMI": "bmi", "ESS": "ess", "AROUSAL_INDEX": "arousal_index",
            "RDI": "rdi", "REM_AHI": "rem_ahi", "SUPINE_AHI": "supine_ahi", "HEART_RATE": "heart_rate"}


def gold_vec(r):
    m = r["meta"]
    v = []
    for fcol in FEATS:
        if fcol == "gender":
            v.append(1.0 if m["gender"] == "male" else 0.0)
        elif fcol == "snoring":
            v.append(1.0 if str(m["snoring"]).lower() == "true" else 0.0)
        elif fcol == "cpap":
            v.append(1.0 if str(m["cpap_recommended"]).lower() == "true" else 0.0)
        else:
            v.append(float(m[fcol]))
    return v


def extracted_vec(r, spans):
    text = r["text"]
    got = {}
    for (s, e, lab) in sorted(spans):
        if lab in LAB2FEAT and LAB2FEAT[lab] not in got:
            m = NUM.search(text[s:e])
            if m: got[LAB2FEAT[lab]] = float(m.group())
        elif lab == "GENDER" and "gender" not in got:
            got["gender"] = 1.0 if text[s:e].lower() in ("male", "man") else 0.0
        elif lab.startswith("SNORING") and "snoring" not in got:
            got["snoring"] = 1.0 if lab == "SNORING_PRESENT" else 0.0
        elif lab.startswith("CPAP") and "cpap" not in got:
            got["cpap"] = 1.0 if lab == "CPAP_YES" else 0.0
    return [got.get(fcol, np.nan) for fcol in FEATS]


Xtr = np.array([gold_vec(r) for r in train])
ytr = np.array([OIDX[r["meta"]["severity"]] for r in train])
yte = np.array([OIDX[r["meta"]["severity"]] for r in test])
med = np.nanmedian(Xtr, axis=0)


def impute(X):
    X = X.copy(); idx = np.where(np.isnan(X)); X[idx] = np.take(med, idx[1]); return X


def run(feat_idx, tag):
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42)
    clf.fit(impute(Xtr)[:, feat_idx], ytr)
    Xg = impute(np.array([gold_vec(r) for r in test]))[:, feat_idx]
    Xe = impute(np.array([extracted_vec(r, preds[r["id"]]) for r in test]))[:, feat_idx]
    ag = accuracy_score(yte, clf.predict(Xg))
    ae = accuracy_score(yte, clf.predict(Xe))
    fe = f1_score(yte, clf.predict(Xe), average="macro", labels=[0, 1, 2, 3], zero_division=0)
    print(f"  {tag:28} ceiling(gold)={ag:.3f}   end2end(NER)={ae:.3f}  macroF1={fe:.3f}")


print("\n### True end-to-end: reports -> NER (all features) -> classifier -> severity")
allidx = list(range(len(FEATS)))
run(allidx, "full feature set")
run([i for i in allidx if FEATS[i] != "cpap"], "full set minus cpap (leak)")
