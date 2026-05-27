# AHI and SpO2 Clinical NER

500 sleep-study notes labelled for pulling three values out of free text:

- `AHI` — apnea-hypopnea index (events per hour)
- `SPO2_MEAN` — mean oxygen saturation (%)
- `SPO2_NADIR` — lowest (nadir) oxygen saturation (%)

Each note has character-level spans for these three entities. The notes are split
into train (350), validation (75), and test (75).

- Kaggle dataset: https://www.kaggle.com/datasets/youssefarabyyoussef/ahi-spo2-clinical-ner
- Code and methodology: https://github.com/youssef-Araby/ahi-spo2-clinical-ner
- Source data we built on: https://www.kaggle.com/datasets/ziya07/sleep-disordered-breathing-detection

## Why we built it

The task is NER: find the AHI and oxygen-saturation values inside clinical notes. The
source dataset can't be used for that as it ships, for three reasons:

1. Its note fields contain no numbers. The AHI and SpO2 values sit only in spreadsheet
   columns, never in the text, so there is nothing in the notes for a model to extract.
2. The notes are the same 10 template sentences repeated across all 500 rows.
3. The severity labels mostly disagree with the AHI (they match in about a third of rows).

So we kept the dataset's real measured values and rewrote the notes around them.

## How we built it

1. **Cleaned the values** (`sdb_clean.csv`). Rounded AHI to one decimal. Used the higher of
   the two oxygen columns as mean SpO2 and the lower as the nadir. Recomputed severity from
   AHI using the standard AASM cut-offs — mild 5–15, moderate 15–30, severe ≥30 (American
   Academy of Sleep Medicine).
2. **Wrote one note per patient.** Each note reads like a real polysomnography report (we
   based the structure and wording on de-identified examples from MTSamples) and states that
   patient's AHI, mean SpO2, and nadir SpO2 in ordinary clinical language. Each note also
   carries the other numbers a real report has — dates, BMI, heart rate, sleep-stage
   percentages, arousal index — so the three targets are not the only numbers on the page.
3. **Checked every label.** The three values were marked as the note was written, then
   compared against the cleaned values; any note that didn't match was dropped. The released
   set has 1,500 entities and no span errors.

The full write-up is in [`report/methodology_dataset.md`](report/methodology_dataset.md).

## Files (on the Kaggle dataset page)

- `notes.jsonl` — all 500 notes
- `notes_train.jsonl`, `notes_val.jsonl`, `notes_test.jsonl` — the 350 / 75 / 75 split
- `sdb_clean.csv` — the cleaned values the notes were built from

## Record format

One JSON object per line:

```json
{
  "id": "P1001",
  "text": "... an AHI of 9.4 per hour ... saturation averaged 91% during sleep and dropped to a low of 90% ...",
  "entities": [
    {"start": 302, "end": 305, "label": "AHI",        "text": "9.4"},
    {"start": 387, "end": 389, "label": "SPO2_MEAN",   "text": "91"},
    {"start": 421, "end": 423, "label": "SPO2_NADIR",  "text": "90"}
  ],
  "meta": {"ahi": 9.4, "spo2_mean": 91, "spo2_nadir": 90, "severity": "Mild"}
}
```

`start` and `end` are character offsets, so `text[start:end]` is the entity.

## Limitations

- The notes are synthetic. They use real values and real report style, but they are not real
  patient records.
- In the source data the oxygen values don't track AHI, so a few notes describe unusual
  combinations (for example a high AHI with a near-normal nadir). We kept the real numbers
  rather than invent correlations.

## License

CC0-1.0, the same licence as the source dataset. Please credit both this dataset and
`ziya07/sleep-disordered-breathing-detection`.
