"""
ClinicalBERT baseline (related work #3: transformer token classification).

Fine-tunes Bio_ClinicalBERT for BIO token classification. Character-level entity
spans are aligned to subword tokens via the fast tokenizer's offset mapping; at
inference the predicted BIO tags are decoded back to character spans and scored
with the shared harness.

Usage:
    python src/model_clinicalbert.py                 # train on train, score on test
    python src/model_clinicalbert.py --epochs 8 --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_jsonl, evaluate, print_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "emilyalsentzer/Bio_ClinicalBERT"
LABELS = ["O", "B-AHI", "I-AHI", "B-SPO2_MEAN", "I-SPO2_MEAN", "B-SPO2_NADIR", "I-SPO2_NADIR"]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}
# Clinical PSG reports run long (median ~970 tokens). We tokenize each document
# into overlapping windows so targets in late sections are never truncated away.
MAXLEN = 384
STRIDE = 128


def _windows(text, tok):
    """Yield (input_ids, attention_mask, offset_mapping) for each overlapping window."""
    enc = tok(text, truncation=True, max_length=MAXLEN, stride=STRIDE,
              return_overflowing_tokens=True, return_offsets_mapping=True)
    for w in range(len(enc["input_ids"])):
        yield enc["input_ids"][w], enc["attention_mask"][w], enc["offset_mapping"][w]


def _labels_for(offsets, ents):
    labels = []
    for (s, e) in offsets:
        if s == e:  # special token
            labels.append(-100)
            continue
        lab = "O"
        for ent in ents:
            if s >= ent["start"] and e <= ent["end"] and e > ent["start"]:
                lab = ("B-" if s == ent["start"] else "I-") + ent["label"]
                break
        labels.append(L2I[lab])
    return labels


class DS(torch.utils.data.Dataset):
    def __init__(self, records, tok):
        self.items = []
        for r in records:
            for ids, mask, offs in _windows(r["text"], tok):
                self.items.append({"input_ids": ids, "attention_mask": mask,
                                   "labels": _labels_for(offs, r["entities"])})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def predict(records, model, tok, device):
    model.eval()
    out = []
    for r in records:
        found = set()  # dedupe spans detected in overlapping windows
        for ids, mask, off in _windows(r["text"], tok):
            t_ids = torch.tensor([ids], device=device)
            t_mask = torch.tensor([mask], device=device)
            with torch.no_grad():
                tags = model(input_ids=t_ids, attention_mask=t_mask).logits[0].argmax(-1).tolist()
            labs = [I2L[t] for t in tags]
            i = 0
            while i < len(labs):
                if labs[i].startswith("B-") and off[i] != [0, 0]:
                    lab = labs[i][2:]
                    s, e, j = off[i][0], off[i][1], i + 1
                    while j < len(labs) and labs[j] == "I-" + lab:
                        e = off[j][1]
                        j += 1
                    found.add((s, e, lab))
                    i = j
                else:
                    i += 1
        ents = [{"start": s, "end": e, "label": lab} for (s, e, lab) in sorted(found)]
        out.append({"id": r["id"], "entities": ents})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--gold", help="evaluate on this JSONL instead of a standard split")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--train", help="train on this JSONL instead of data/notes_train.jsonl")
    ap.add_argument("--save", help="optional path to write predictions JSONL")
    args = ap.parse_args()
    set_seed(42)

    train = load_jsonl(args.train or ROOT / "data" / "notes_train.jsonl")
    eval_set = load_jsonl(args.gold or ROOT / "data" / f"notes_{args.split}.jsonl")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), id2label=I2L, label2id=L2I)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    targs = TrainingArguments(
        output_dir=str(ROOT / "outputs" / "clinicalbert"),
        num_train_epochs=args.epochs, per_device_train_batch_size=16,
        learning_rate=3e-5, weight_decay=0.01, save_strategy="no",
        logging_steps=50, report_to=[], disable_tqdm=True, seed=42)
    trainer = Trainer(model=model, args=targs, train_dataset=DS(train, tok),
                      data_collator=DataCollatorForTokenClassification(tok))
    print(f"training Bio_ClinicalBERT on {len(train)} notes ({device}, {args.epochs} epochs)...")
    trainer.train()

    preds = predict(eval_set, model, tok, device)
    if args.save:
        import json as _json
        Path(args.save).write_text("\n".join(_json.dumps(p) for p in preds))
        print(f"wrote {len(preds)} predictions to {args.save}")
    print_report(evaluate(eval_set, preds), f"ClinicalBERT — {args.gold or args.split}")


if __name__ == "__main__":
    main()
