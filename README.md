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

## Before and after (patient P1000)

Here is the **complete original row** for patient P1000:

| Column | Value |
|--------|-------|
| Age | 56 |
| Gender | Female |
| BMI | 27.2 |
| Snoring | False |
| Oxygen_Saturation | 96.96 |
| AHI | 42.996 |
| ECG_Heart_Rate | 90 |
| SpO2 | 96.27 |
| Nasal_Airflow | 0.51 |
| Chest_Movement | 0.12 |
| Diagnosis_of_SDB | Severe |
| Severity | Mild |
| Treatment_Required | True |
| CPAP | True |
| Surgery | False |
| Physician_Notes | "Patient reports difficulty breathing during sleep." |
| Patient_Symptoms | "No issues related to sleep, feels rested." |

Two problems are visible. The note fields contain no numbers, so there is nothing to
extract. And the labels contradict each other — `Severity` says "Mild" while
`Diagnosis_of_SDB` says "Severe" for the same patient — and by the AASM cut-offs an AHI of
43 is severe, not mild (AASM Task Force, *Sleep* 1999):

| AHI (events/hr) | Severity |
|-----------------|----------|
| < 5             | None     |
| 5 – 15          | Mild     |
| 15 – 30         | Moderate |
| ≥ 30            | Severe   |

Our note for the same patient, with the three target spans shown in **bold**:

> SLEEP STUDY INTERPRETATION
>
> 56-year-old woman referred for evaluation of unrefreshing sleep and witnessed pauses in
> breathing. She denies habitual snoring. Epworth Sleepiness Scale 14/24. Diagnostic
> polysomnography was performed on 02/14/2023 using a 16-channel montage.
>
> The study demonstrated frequent obstructive respiratory events, most pronounced in REM
> and the supine position, yielding an apnea-hypopnea index of **43** events per hour.
> Despite the elevated event frequency, oxygenation was relatively preserved: mean oxygen
> saturation **97**% with a nadir of **96**%. Arousal index 39/hr.
>
> IMPRESSION: **Severe** obstructive sleep apnea. Recommend CPAP titration.

**Every clinical fact in the note is taken from the row above — none of it is invented:**

| In the note | Comes from the row |
|-------------|--------------------|
| "56-year-old" | `Age` = 56 |
| "woman" | `Gender` = Female |
| "denies habitual snoring" | `Snoring` = False |
| "apnea-hypopnea index of **43**" (`AHI` span) | `AHI` = 42.996 → 43.0 |
| "mean oxygen saturation **97**%" (`SPO2_MEAN` span) | max(`Oxygen_Saturation` 96.96, `SpO2` 96.27) |
| "nadir of **96**%" (`SPO2_NADIR` span) | min(`Oxygen_Saturation` 96.96, `SpO2` 96.27) |
| "**Severe** obstructive sleep apnea" | severity recomputed from `AHI` (the row's "Mild" was wrong) |
| "Recommend CPAP titration" | `CPAP` = True |

(`BMI` and `ECG_Heart_Rate` from the row are written into many other patients' notes.) The
only invented parts are realistic report filler — the study date, the Epworth score,
"16-channel montage", and the arousal index — which are deliberately **not** labelled, so
they act as distractors the model must learn to ignore.

## How we built it

1. **Cleaned the values** (`sdb_clean.csv`). Rounded AHI to one decimal. Used the higher of
   the two oxygen columns as mean SpO2 and the lower as the nadir. Recomputed severity from
   AHI using the AASM cut-offs shown in the table above.
2. **Wrote one note per patient with Claude Opus 4.7.** Using each patient's cleaned values
   (AHI, mean SpO2, nadir SpO2, severity, age, sex, BMI) as input, we generated the note text
   with the large language model Claude Opus 4.7 (Anthropic). The notes follow the section
   structure and wording of real de-identified polysomnography reports from MTSamples [1]
   (Clinical Information → Study Protocol → Respiratory Measurements → Impression) and state
   the patient's AHI, mean SpO2, and nadir SpO2 in ordinary clinical language. Each note also
   carries the other numbers a real report has — dates, BMI, heart rate, sleep-stage
   percentages, arousal index — so the three targets are not the only numbers on the page.
   (We first tried fixed templates, but the text was too formulaic, so we used the model.)
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

## Sources

- **[1] Note structure — MTSamples, Sleep Medicine specialty.** De-identified transcribed
  reports we used as the template for section structure and phrasing. Examples:
  - Overnight Polysomnogram — https://mtsamples.com/site/pages/sample.asp?Type=78-Sleep+Medicine&Sample=1163-Overnight+Polysomnogram
  - Polysomnography — https://mtsamples.com/site/pages/sample.asp?Type=78-Sleep+Medicine&Sample=1483-Polysomnography
  - Sleep Study Interpretation — https://mtsamples.com/site/pages/sample.asp?Type=78-Sleep+Medicine&Sample=668-Sleep+Study+Interpretation
- **[2] Severity cut-offs — AASM.** American Academy of Sleep Medicine Task Force.
  *Sleep-related breathing disorders in adults: recommendations for syndrome definition and
  measurement techniques in clinical research.* Sleep. 1999;22(5):667–689.
- **Source dataset.** `ziya07/sleep-disordered-breathing-detection`, Kaggle (CC0):
  https://www.kaggle.com/datasets/ziya07/sleep-disordered-breathing-detection
- **Note-generation model.** Claude Opus 4.7 (Anthropic), used to write the note text from
  each patient's cleaned values. All generated values were verified against the source data.

## License

CC0-1.0, the same licence as the source dataset. Please credit both this dataset and
`ziya07/sleep-disordered-breathing-detection`.
