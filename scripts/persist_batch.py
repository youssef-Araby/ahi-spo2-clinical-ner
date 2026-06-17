#!/usr/bin/env python3
"""Persist a generation-batch workflow output into the v2 dataset.

Usage: python scripts/persist_batch.py <task_output.json>

- Parses each accepted record's tagged report into character-offset NER labels
  (<AHI>/<MEAN>/<NADIR> -> AHI/SPO2_MEAN/SPO2_NADIR), verifying every tagged
  value equals its columnar value. Records that fail verification are dropped.
- Appends NER records to data/v2/notes_v2.jsonl and columnar rows to
  data/v2/columnar_v2.csv. Tracks processed ids in data/v2/manifest.json so
  re-running a batch never duplicates.
"""
import json, re, sys, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
NOTES = OUT / "notes_v2.jsonl"
CSV = OUT / "columnar_v2.csv"
MAN = OUT / "manifest.json"

TAGMAP = {"AHI": "AHI", "MEAN": "SPO2_MEAN", "NADIR": "SPO2_NADIR"}
TAGRE = re.compile(r"<(AHI|MEAN|NADIR)>(.*?)</\1>", re.S)
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")
COLCHECK = {"AHI": ("ahi", 0.06), "SPO2_MEAN": ("spo2_mean", 0.6), "SPO2_NADIR": ("spo2_nadir", 0.6)}

CSV_COLS = ["patient_id", "ahi", "severity", "odi", "spo2_mean", "spo2_nadir", "t90",
            "age", "gender", "bmi", "ess", "snoring", "arousal_index", "rdi",
            "rem_ahi", "supine_ahi", "heart_rate", "scoring_rule", "cpap_recommended",
            "difficulty_flag"]


def parse_tagged(tagged):
    """Strip tags, return (text, entities). One entity per tag occurrence."""
    parts, ents, cur, last = [], [], 0, 0
    for m in TAGRE.finditer(tagged):
        pre = tagged[last:m.start()]
        parts.append(pre); cur += len(pre)
        inner, label = m.group(2), TAGMAP[m.group(1)]
        nm = NUMRE.search(inner)
        if nm:
            ents.append({"start": cur + nm.start(), "end": cur + nm.end(),
                         "label": label, "text": nm.group(0)})
        parts.append(inner); cur += len(inner)
        last = m.end()
    parts.append(tagged[last:])
    return "".join(parts), ents


def verify(ents, col):
    by = {}
    for e in ents:
        by.setdefault(e["label"], []).append(e)
    for label, (key, tol) in COLCHECK.items():
        if label not in by:
            return f"missing {label}"
        for e in by[label]:
            if abs(float(e["text"]) - float(col[key])) > tol:
                return f"{label} span {e['text']} != col {col[key]}"
        # offset sanity: text[start:end] must equal recorded text -- checked by caller
    return None


def main():
    src = json.load(open(sys.argv[1]))
    res = src["result"]
    records = res.get("records", [])

    man = json.load(open(MAN)) if MAN.exists() else {"done_ids": [], "batches": [], "dropped": []}
    done = set(man["done_ids"])

    new_notes, new_rows, dropped = [], [], []
    for r in records:
        pid = r["id"]
        if pid in done:
            continue
        col = r["columnar"]
        text, ents = parse_tagged(r["report_tagged"])
        # offset integrity
        bad = next((e for e in ents if text[e["start"]:e["end"]] != e["text"]), None)
        if bad:
            dropped.append((pid, "offset mismatch")); continue
        err = verify(ents, col)
        if err:
            dropped.append((pid, err)); continue
        meta = dict(col); meta["difficulty_flag"] = r.get("difficulty_flag", "")
        new_notes.append({"id": pid, "text": text, "entities": ents, "meta": meta})
        row = {k: (col.get(k) if k in col else "") for k in CSV_COLS}
        row["patient_id"] = pid
        row["difficulty_flag"] = r.get("difficulty_flag", "")
        new_rows.append(row)
        done.add(pid)

    # append notes
    with open(NOTES, "a") as f:
        for n in new_notes:
            f.write(json.dumps(n) + "\n")
    # append csv (write header if new)
    write_header = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if write_header:
            w.writeheader()
        for row in new_rows:
            w.writerow(row)

    man["done_ids"] = sorted(done)
    man["batches"].append({"src": sys.argv[1], "added": len(new_notes),
                           "dropped": len(dropped), "batch": res.get("batch")})
    man["dropped"].extend([{"id": d[0], "why": d[1]} for d in dropped])
    json.dump(man, open(MAN, "w"), indent=2)

    print(f"added {len(new_notes)} notes (total now {len(done)}); dropped {len(dropped)}: {dropped[:5]}")
    print(f"workflow reported acceptedN={res.get('acceptedN')} avg_hardness={res.get('avg_hardness'):.2f} avg_realism={res.get('avg_realism'):.2f}")


if __name__ == "__main__":
    main()
