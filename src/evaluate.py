"""
Evaluation harness for the AHI / SpO2 NER task.

Every method (regex, CRF, ClinicalBERT, the proposed model) is scored here so the
comparison is apples-to-apples on the same test notes.

Prediction format (one JSON object per line, same ids as the gold file):
    {"id": "P1234", "entities": [{"start": 12, "end": 14, "label": "AHI"}, ...]}

Metrics produced:
  1. Span-level precision / recall / F1, strict (exact start, end, AND label),
     per label + micro + macro.
  2. Value-level accuracy: correct numeric value per entity (float tolerance),
     which tolerates off-by-a-character spans and "43" vs "43.0".
  3. Note-level exact match: all three target values correct in a note.

Usage:
    from evaluate import load_jsonl, evaluate, print_report
    metrics = evaluate(gold_records, predictions)   # predictions: list or {id: [ents]}
    print_report(metrics)

CLI:
    python src/evaluate.py --pred preds.jsonl           # score a prediction file
    python src/evaluate.py --selftest                   # sanity checks
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD_TEST = ROOT / "data" / "notes_test.jsonl"
LABELS = ["AHI", "SPO2_MEAN", "SPO2_NADIR"]
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def _value(s: str | None):
    """Extract the first numeric value from a surface string, as float."""
    if not s:
        return None
    m = _NUM.search(s)
    return float(m.group()) if m else None


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(gold: list[dict], preds, value_tol: float = 0.05) -> dict:
    """gold: list of records {id, text, entities:[{start,end,label,text}]}.
    preds: list of records {id, entities:[...]}  OR  dict {id: [entities]}."""
    if isinstance(preds, list):
        preds = {p["id"]: p.get("entities", []) for p in preds}

    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    v_ok, v_tot = defaultdict(int), defaultdict(int)
    note_exact = note_total = 0

    for g in gold:
        text = g["text"]
        gold_spans = {(e["start"], e["end"], e["label"]) for e in g["entities"]}
        pred_ents = preds.get(g["id"], [])
        pred_spans = {(e["start"], e["end"], e["label"]) for e in pred_ents}

        # span-level (strict)
        for lab in LABELS:
            gl = {s for s in gold_spans if s[2] == lab}
            pl = {s for s in pred_spans if s[2] == lab}
            tp[lab] += len(gl & pl)
            fp[lab] += len(pl - gl)
            fn[lab] += len(gl - pl)

        # value-level: gold value vs first predicted entity of the same label
        gold_val = {e["label"]: _value(e.get("text") or text[e["start"]:e["end"]])
                    for e in g["entities"]}
        pred_val = {}
        for e in pred_ents:
            surface = text[e["start"]:e["end"]] if 0 <= e["start"] <= e["end"] <= len(text) else e.get("text")
            pred_val.setdefault(e["label"], _value(surface))

        note_ok = True
        for lab in LABELS:
            if lab in gold_val:
                v_tot[lab] += 1
                pv, gv = pred_val.get(lab), gold_val[lab]
                ok = pv is not None and gv is not None and abs(pv - gv) <= value_tol
                v_ok[lab] += int(ok)
                note_ok = note_ok and ok
        note_total += 1
        note_exact += int(note_ok)

    # assemble
    per_label = {}
    for lab in LABELS:
        p, r, f = _prf(tp[lab], fp[lab], fn[lab])
        per_label[lab] = {"P": p, "R": r, "F1": f, "support": tp[lab] + fn[lab],
                          "value_acc": v_ok[lab] / v_tot[lab] if v_tot[lab] else 0.0}
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p, micro_r, micro_f = _prf(TP, FP, FN)
    macro_f = sum(per_label[l]["F1"] for l in LABELS) / len(LABELS)
    return {
        "per_label": per_label,
        "micro": {"P": micro_p, "R": micro_r, "F1": micro_f},
        "macro_F1": macro_f,
        "value_acc_overall": sum(v_ok.values()) / sum(v_tot.values()) if sum(v_tot.values()) else 0.0,
        "note_exact_match": note_exact / note_total if note_total else 0.0,
        "n_notes": note_total,
    }


def print_report(m: dict, title: str = "") -> None:
    if title:
        print(f"\n### {title}")
    print("Span-level (strict: exact span + label)")
    print(f"  {'label':<12}{'P':>7}{'R':>7}{'F1':>7}{'supp':>7}{'valAcc':>9}")
    for lab in LABELS:
        d = m["per_label"][lab]
        print(f"  {lab:<12}{d['P']:>7.3f}{d['R']:>7.3f}{d['F1']:>7.3f}{d['support']:>7}{d['value_acc']:>9.3f}")
    mi = m["micro"]
    print(f"  {'micro':<12}{mi['P']:>7.3f}{mi['R']:>7.3f}{mi['F1']:>7.3f}")
    print(f"  macro-F1: {m['macro_F1']:.3f}")
    print(f"Value-level accuracy (overall): {m['value_acc_overall']:.3f}")
    print(f"Note-level exact match (all 3 values): {m['note_exact_match']:.3f}  (n={m['n_notes']})")


def _selftest():
    gold = load_jsonl(GOLD_TEST)
    print(f"loaded {len(gold)} gold test notes")
    # perfect predictions == gold entities -> all 1.0
    perfect = [{"id": g["id"], "entities": g["entities"]} for g in gold]
    print_report(evaluate(gold, perfect), "sanity: gold vs gold (expect 1.000)")
    # empty predictions -> all 0.0
    empty = [{"id": g["id"], "entities": []} for g in gold]
    print_report(evaluate(gold, empty), "sanity: gold vs empty (expect 0.000)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="prediction JSONL to score against the gold test set")
    ap.add_argument("--gold", default=str(GOLD_TEST))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest or not args.pred:
        _selftest()
        return
    gold = load_jsonl(args.gold)
    preds = load_jsonl(args.pred)
    print_report(evaluate(gold, preds), f"predictions: {args.pred}")


if __name__ == "__main__":
    main()
