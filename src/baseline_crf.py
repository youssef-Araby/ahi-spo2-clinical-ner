"""
CRF baseline (related work #2: classic sequence labelling, pre-transformer).

Notes are tokenized with character offsets and converted to BIO tags (the targets
are single numeric tokens, so each entity is one B- tag). A linear-chain CRF
(sklearn-crfsuite) is trained on the train split with word/number/context features,
then decoded back to character spans and scored with the shared harness.

Usage:
    python src/baseline_crf.py                 # train on train, score on test
    python src/baseline_crf.py --split val
    python src/baseline_crf.py --save preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_jsonl, evaluate, print_report  # noqa: E402

from sklearn_crfsuite import CRF  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["AHI", "SPO2_MEAN", "SPO2_NADIR"]
TOK = re.compile(r"\d+\.\d+|\d+|[A-Za-z]+|[^\sA-Za-z0-9]")


def tokenize(text: str):
    return [(m.group(), m.start(), m.end()) for m in TOK.finditer(text)]


def is_num(t: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", t))


def shape(t: str) -> str:
    return re.sub(r"\d", "d", re.sub(r"[a-z]", "x", re.sub(r"[A-Z]", "X", t)))


def bio(tokens, entities):
    labels = ["O"] * len(tokens)
    for e in entities:
        for i, (_, s, en) in enumerate(tokens):
            if s >= e["start"] and en <= e["end"] and en > e["start"]:
                labels[i] = ("B-" if s == e["start"] else "I-") + e["label"]
    return labels


def tok_feats(tokens, i):
    t = tokens[i][0]
    f = {
        "bias": 1.0, "low": t.lower(), "is_num": is_num(t), "is_digit": t.isdigit(),
        "is_alpha": t.isalpha(), "shape": shape(t), "suf3": t[-3:].lower(), "len": len(t),
    }
    for d in (-2, -1, 1, 2):
        j = i + d
        if 0 <= j < len(tokens):
            f[f"low{d:+d}"] = tokens[j][0].lower()
            f[f"isnum{d:+d}"] = is_num(tokens[j][0])
        else:
            f[f"edge{d:+d}"] = True
    return f


def featurize(records):
    X, Y, toks = [], [], []
    for r in records:
        t = tokenize(r["text"])
        toks.append(t)
        X.append([tok_feats(t, i) for i in range(len(t))])
        Y.append(bio(t, r["entities"]))
    return X, Y, toks


def decode(tokens, labels):
    ents, i = [], 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            lab = labels[i][2:]
            s, e, j = tokens[i][1], tokens[i][2], i + 1
            while j < len(labels) and labels[j] == "I-" + lab:
                e = tokens[j][2]
                j += 1
            ents.append({"start": s, "end": e, "label": lab})
            i = j
        else:
            i += 1
    return ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--gold", help="evaluate on this JSONL instead of a standard split")
    ap.add_argument("--train", help="train on this JSONL instead of data/notes_train.jsonl")
    ap.add_argument("--save")
    args = ap.parse_args()

    train = load_jsonl(args.train or ROOT / "data" / "notes_train.jsonl")
    eval_set = load_jsonl(args.gold or ROOT / "data" / f"notes_{args.split}.jsonl")

    Xtr, Ytr, _ = featurize(train)
    Xev, _, toks_ev = featurize(eval_set)

    crf = CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100,
              all_possible_transitions=True)
    crf.fit(Xtr, Ytr)
    pred_labels = crf.predict(Xev)

    preds = [{"id": eval_set[k]["id"], "entities": decode(toks_ev[k], pred_labels[k])}
             for k in range(len(eval_set))]
    if args.save:
        Path(args.save).write_text("\n".join(json.dumps(p) for p in preds))
        print(f"wrote {len(preds)} predictions to {args.save}")
    tag = args.gold or args.split
    print(f"trained CRF on {len(train)} notes; evaluating on {len(eval_set)} ({tag})")
    print_report(evaluate(eval_set, preds), f"CRF baseline — {tag}")


if __name__ == "__main__":
    main()
