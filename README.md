# AHI / SpO₂ Clinical NER + Severity Pipeline

A research-backed **synthetic** dataset of obstructive-sleep-apnea (OSA)
polysomnography (PSG) reports, and a two-part modelling pipeline built on it:

1. **NER** — extract clinical values from the free-text reports.
2. **Severity classification** — map a patient to an AASM severity class
   (None / Mild / Moderate / Severe), end-to-end from the report.

## Why synthetic

The originally-assigned dataset was unusable for this task: its note fields
contained no numbers, the notes were ~10 recycled templates, and the severity
labels agreed with the AHI in only ~36 % of rows. With the supervisor's approval
we built a synthetic replacement, on the condition that it be principled and
literature-grounded. See [report/methodology.md](report/methodology.md).

## What's in the data

Each record has a realistic PSG **report** (free text) plus the **columnar**
values it was built from. 500 records, split 361 / 70 / 69 (train / val / test),
stratified by severity.

- **Severity** (classifier target) is defined deterministically from AHI
  (AASM 1999): None <5, Mild 5–<15, Moderate 15–30, Severe >30.
- **Difficulty is engineered in** — distractor indices (ODI, RDI, REM/supine-AHI…),
  phrasing variety, and medically-valid discordant cases (e.g. severe AHI with a
  benign desaturation profile).

## Repository layout

```
src/
  evaluate.py            shared strict-span NER evaluator (3-target)
  baseline_regex.py      NER baseline 1 — rules
  baseline_crf.py        NER baseline 2 — CRF        (--train / --gold)
  model_clinicalbert.py  NER baseline 3 — Bio_ClinicalBERT, sliding-window
  classifier_severity.py severity classifier on columnar features (3 conditions)
  ner_multi.py           16-entity NER + true end-to-end pipeline
scripts/
  gen_batch.workflow.js  agent-orchestrated data generation (batched)
  persist_batch.py       tag→offset conversion + verification, writes data/v2/
  build_splits.py        stratified 70/15/15 split
  autolabel_multi.py     gold-anchored auto-labelling of all in-text features
  analyze_distractors.py per-distractor confusion analysis
  end_to_end.py          report→NER-AHI→AASM severity pipeline
  pipeline_full.py       full-feature classifier pipeline + 3-feature ablation
report/
  methodology.md         how the data was generated and validated
  results.md             all model + pipeline results
data/v2/                 the dataset (kept local, git-ignored)
archive/                 intermediate dumps + superseded docs (kept local)
```

## Reproduce

```bash
# 1. generate (agent-orchestrated; see methodology.md) -> data/v2/notes_v2.jsonl
#    scripts/gen_batch.workflow.js  +  python scripts/persist_batch.py <out.json>
python scripts/build_splits.py          # stratified split
python scripts/autolabel_multi.py       # add all-feature NER labels

# NER (3-target)
python src/baseline_regex.py     --gold data/v2/notes_v2_test.jsonl
python src/baseline_crf.py       --train data/v2/notes_v2_train.jsonl --gold data/v2/notes_v2_test.jsonl
python src/model_clinicalbert.py --train data/v2/notes_v2_train.jsonl --gold data/v2/notes_v2_test.jsonl

# severity classifier + pipelines
python src/classifier_severity.py
python scripts/end_to_end.py
python src/ner_multi.py                 # 16-entity NER + true end-to-end
```

## Headline results

- **NER** is genuinely hard: rules collapse (micro-F1 0.49), CRF 0.89,
  ClinicalBERT 0.94. Labelling the distractors as their own entities lifts
  note-level exact-match on the 3 core targets to **1.00** (from 0.87 with only the
  3 targets labelled).
- **End-to-end** severity (report → NER → class) reaches **0.99** with ClinicalBERT
  against a 1.00 gold ceiling — AHI bucketing absorbs small extraction errors.

Full numbers and caveats: [report/results.md](report/results.md).
