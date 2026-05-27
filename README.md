# Sleep-Apnea Clinical NER: AHI & Oxygen-Saturation Extraction

A named-entity-recognition (NER) dataset for extracting the **Apnea–Hypopnea Index (AHI)**
and **oxygen-saturation metrics** from free-text sleep-study notes. 500 clinical notes,
each annotated with character-level spans for three entity types:

| Label | Meaning |
|-------|---------|
| `AHI` | apnea–hypopnea index (events/hour) |
| `SPO2_MEAN` | mean / average oxygen saturation (%) |
| `SPO2_NADIR` | lowest / nadir oxygen saturation (%) |

> 📦 **Kaggle:** https://www.kaggle.com/datasets/youssefarabyyoussef/ahi-spo2-clinical-ner

---

## Why this dataset exists

The task — *extract AHI and oxygen-saturation values from clinical notes via NER* — was
assigned against the Kaggle dataset
[`ziya07/sleep-disordered-breathing-detection`](https://www.kaggle.com/datasets/ziya07/sleep-disordered-breathing-detection)
(CC0). On inspection, that dataset **cannot support the task as distributed**:

- **The notes contain no numbers.** 0 of 500 `Physician_Notes` (and 0 `Patient_Symptoms`)
  contain a single digit — the AHI/SpO₂ values live only in structured columns. There is
  literally nothing in the text for an NER model to extract.
- **Only 10 template sentences** are recycled across all 500 rows.
- **Labels are near-random.** `Severity` matches the AHI-derived severity in only 36% of
  rows (chance ≈ 25%); e.g. a row with AHI 43 (severe) is labelled "Mild".

So we kept the dataset's real measured values and **rebuilt the text** so that the metrics
actually appear in clinically realistic notes. This README documents exactly what we changed.

## What we did to the data

### 1. Cleaned the numeric values
We preserved each patient's real values and corrected the internal inconsistencies:

| Field | Problem | Fix |
|-------|---------|-----|
| `AHI` | 14-digit spurious precision (`42.99606872…`) | rounded to one decimal |
| two oxygen columns | uncorrelated (r = −0.06), neither ordered as mean/nadir | per row: `SPO2_MEAN = max(both)`, `SPO2_NADIR = min(both)` → enforces mean ≥ nadir |
| `Severity` | uncorrelated with AHI, 131 missing | recomputed from AHI (Mild 5–15, Moderate 15–30, Severe ≥30) |
| diagnosis | redundant + 119 missing | derived `SDB_present = AHI ≥ 5` |

Result: [`data/sdb_clean.csv`](data/sdb_clean.csv) — clean, consistent structured values.

### 2. Wrote realistic notes carrying those values
The original notes were unusable, so we **authored a new note for each patient**, conditioned
on that patient's real `AHI`, `SPO2_MEAN`, and `SPO2_NADIR`. The notes were written by a large
language model (not fixed templates — an early template version read too formulaic and was
discarded) in the style of real de-identified polysomnography reports, which we reviewed for
structure and phrasing (see [`data/reference/real_psg_reports.md`](data/reference/real_psg_reports.md)).
The notes vary in document type — full polysomnography report, sleep-clinic consultation,
attending interpretation, dictated summary — and in length, register, and section ordering.

Each note also embeds many **distractor numbers** that must *not* be extracted — study dates,
BMI, heart rate, sleep-stage percentages, arousal and limb-movement indices, respiratory
disturbance index, ICD codes — so the extraction task is genuinely hard rather than "grab the
only number".

### 3. Guaranteed-correct labels
While writing, the three target values were marked inline; we then recorded their exact
character spans and **verified every label against the cleaned dataset value** (AHI within
±0.05, SpO₂ exact). Any note whose values didn't match was discarded. Final result:
**500 notes, 1,500 entities, 0 span errors.**

Full write-up for the report: [`report/methodology_dataset.md`](report/methodology_dataset.md).

## Files

```
data/
  notes.jsonl          # all 500 annotated notes
  notes_train.jsonl    # 350  (70%)
  notes_val.jsonl      #  75  (15%)
  notes_test.jsonl     #  75  (15%)
  sdb_clean.csv        # cleaned structured values the notes were built from
  original/            # the original Kaggle CSVs, kept for provenance
  reference/           # real PSG report structure & phrasing we modelled on
report/
  methodology_dataset.md
```

## Format

Each line of `notes*.jsonl` is one record:

```json
{
  "id": "P1001",
  "text": "PULMONARY / SLEEP CONSULTATION\n\nMr. ___ is a 69-year-old gentleman ... an AHI of 9.4 per hour ... Oxygen saturation averaged 91% during sleep and dropped to a low of 90%. ...",
  "entities": [
    {"start": 302, "end": 305, "label": "AHI",        "text": "9.4"},
    {"start": 387, "end": 389, "label": "SPO2_MEAN",   "text": "91"},
    {"start": 421, "end": 423, "label": "SPO2_NADIR",  "text": "90"}
  ],
  "meta": {"ahi": 9.4, "spo2_mean": 91, "spo2_nadir": 90, "severity": "Mild"}
}
```

`start`/`end` are character offsets into `text`; `text[start:end]` equals the entity surface.

Loading example:

```python
import json
data = [json.loads(l) for l in open("data/notes_train.jsonl")]
```

## Statistics

- **500** notes · **1,500** entities (500 each of `AHI`, `SPO2_MEAN`, `SPO2_NADIR`)
- Severity mix: Severe 236 · Moderate 155 · Mild 109
- Note length: 186–757 characters
- Split: 350 / 75 / 75 (train / val / test)

## Limitations

- The notes are **synthetic**. They are grounded in real measured values and modelled on real
  report language, but do not reproduce the full noise of authentic clinical documentation.
- In the source data the **oxygen values do not correlate with AHI**, so a minority of notes
  describe clinically atypical combinations (e.g. severe AHI with a near-normal nadir). We kept
  the dataset's true numbers rather than fabricate correlations.

## License & attribution

Built on `ziya07/sleep-disordered-breathing-detection` (CC0-1.0). The structured values are
reused from that dataset; the note text was generated for this project. Released under the same
permissive terms — please credit both the original dataset and this derived corpus.
