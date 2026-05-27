# Methodology — Dataset Construction

## A. Source dataset and its limitation for NER

We build on the publicly available *Sleep Disordered Breathing Detection* dataset
(Kaggle, `ziya07`, CC0 license), which the project brief designates as the working
corpus. It comprises 500 patient records with 18 fields: structured clinical
variables (age, gender, BMI, snoring, oxygen saturation, AHI, ECG heart rate, SpO2,
nasal airflow, chest movement), categorical labels (diagnosis, severity, treatment
flags), and two free-text fields (`Physician_Notes`, `Patient_Symptoms`).

The designated task — *named-entity recognition of the apnea–hypopnea index (AHI)
and oxygen-saturation metrics from clinical notes* — cannot be performed on the
free-text fields as distributed. On inspection we found that (i) **none of the 500
notes contain any numeric value** (0/500 records include a digit in either text
field); (ii) the notes are drawn from only **10 recycled template sentences**; and
(iii) the `Severity` label is statistically independent of the AHI it should reflect
(it matches the clinically derived severity in only 36% of records, versus ~25%
expected by chance). The target metrics therefore exist only in the structured
columns and never in the text, leaving nothing for a sequence-labelling model to
extract.

## B. Numeric cleaning

We retain the dataset's real measured values and correct the internal
inconsistencies before any text is produced:

| Field | Issue | Correction |
|-------|-------|-----------|
| `AHI` | 14-digit spurious precision | round to one decimal |
| oxygen columns | two uncorrelated SpO2 columns (r = −0.06), neither ordered | per record, `SpO2_mean = max(·)`, `SpO2_nadir = min(·)`, enforcing mean ≥ nadir |
| `Severity` | uncorrelated with AHI, 131 missing | recompute from AHI (Mild 5–15, Moderate 15–30, Severe ≥30) |
| diagnosis | redundant, 119 missing | derive `SDB_present = AHI ≥ 5` |

This yields a clean table of physiologically ordered, internally consistent values
(`data/sdb_clean.csv`).

## C. Note synthesis with verifiable labels

Because no usable free text exists, we **generate** clinical notes ourselves,
conditioned on each patient's cleaned values. To obtain natural clinical language we
authored the notes with a large language model rather than fixed templates: an
initial template-based generator produced text that was fluent but visibly
formulaic across records, so it was discarded in favour of LLM-authored prose that
varies in document type (full polysomnography report, sleep-clinic consultation,
attending interpretation, dictated summary, follow-up note), register, sentence
structure, and section ordering. The notes are modelled on the structure and
phrasing of real de-identified polysomnography reports (MTSamples, Sleep Medicine
specialty), which we reviewed to fix section conventions
(`CLINICAL INFORMATION → STUDY PROTOCOL → RESPIRATORY MEASUREMENTS → IMPRESSION`)
and the natural surface forms of each metric (e.g. "apnea-hypopnea index of 43
events per hour", "lowest oxygen saturation 88%", "oxygen desaturation down to …").

**Verifiable labelling via inline tagging.** Free LLM generation normally
sacrifices label certainty, because the model may paraphrase or relocate values. We
avoid this by requiring the model to wrap *only* the three target numbers in
sentinel tags inline (`<AHI>`, `<MEAN>`, `<NADIR>`) while writing all surrounding
content, including distractor numbers, untagged. A deterministic parser then (1)
strips the tags and records the exact character offsets of each target, (2) verifies
that each tagged value equals the cleaned dataset value (AHI within ±0.05; SpO2
exact), and (3) requires exactly one of each label and no malformed tags. **Any note
failing verification is discarded**, guaranteeing that every retained label is
correct by construction.

**Realistic distractors.** Each note embeds numerous untagged numerals — study
dates, BMI, heart rate, sleep-stage percentages, arousal and periodic-limb-movement
indices, respiratory disturbance index, event durations, ICD codes — so that the
target values are not trivially the only numbers present. This reproduces the
discrimination problem faced on genuine reports.

The three entity types — `AHI`, `SPO2_MEAN`, `SPO2_NADIR` — are exported as
character spans and, for the sequence models, converted to a BIO token scheme. The
corpus is split 70/15/15 into train/validation/test.

## D. Limitations

The notes are synthetic; although grounded in real measured values and modelled on
real report language, they may not capture the full noise of authentic clinical
documentation. The source dataset's oxygen values are uncorrelated with AHI, so a
minority of records describe clinically atypical combinations (e.g. severe AHI with
a near-normal nadir); we preserve the dataset's true numbers rather than fabricate
correlations, and flag this explicitly. As an external check, a small set of real
MTSamples polysomnography reports can be hand-annotated to test cross-domain
generalisation.
