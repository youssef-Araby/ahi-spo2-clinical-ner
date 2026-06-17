# Methodology — synthetic dataset generation & validation

## 1. Motivation

The task is NER — find the AHI and oxygen-saturation values inside clinical notes —
followed by severity classification. The assigned **source dataset**
([`ziya07/sleep-disordered-breathing-detection`](https://www.kaggle.com/datasets/ziya07/sleep-disordered-breathing-detection),
Kaggle, CC0) cannot support this as it ships, for three reasons:

1. **No numbers in the text.** The AHI/SpO₂ values live only in spreadsheet columns;
   the only free-text fields contain none — there is nothing for a model to extract.
2. **Recycled notes.** The note fields are ~10 template sentences repeated across all
   500 rows.
3. **Labels disagree with the AHI.** The provided `Severity` matches the clinically-
   derived (AASM) severity in only ~36 % of rows (≈ chance for the class distribution).

**Example — the complete row for patient `P1000`** (Kaggle source):

| AHI | Severity | Diagnosis_of_SDB | Physician_Notes | Patient_Symptoms |
|---|---|---|---|---|
| 42.996 | Mild | Severe | "Patient reports difficulty breathing during sleep." | "No issues related to sleep, feels rested." |

(Remaining columns: Age 56, Gender Female, BMI 27.2, SpO₂ 96.27, Oxygen_Saturation 96.96,
ECG_Heart_Rate 90, …). Two problems are visible at once: the only free-text fields
(`Physician_Notes`, `Patient_Symptoms`) contain **no numbers**, so there is nothing for
NER to find; and the labels are **internally contradictory** — `Severity` says *Mild*
while `Diagnosis_of_SDB` says *Severe* for the same patient, and by the AASM cut-offs
(§2) an AHI of 43 is *Severe*, not *Mild*.

We therefore kept the dataset's real measured values and, with the supervisor's
approval, **rewrote the notes around them** — building a synthetic dataset required to
be **as hard as possible while remaining medically valid and literature-grounded**.

## 2. Severity definition

Severity is the classifier target and is defined **deterministically from AHI**,
per the AASM 1999 Task Force cut-offs:

| Class | AHI (events/hr) |
|---|---|
| None | < 5 |
| Mild | 5 – <15 |
| Moderate | 15 – 30 |
| Severe | > 30 |

Difficulty is **not** created by corrupting this rule — it is created by
*decoupling the other variables from the AHI class* within medically valid bounds,
exploiting documented real-world weak/conflicting relationships (AHI vs nadir SpO₂,
ODI in the moderate band, AHI vs daytime symptoms, the AHI–CVD-mortality null).

## 3. Sources imitated

Every quantitative choice is anchored to a published source:

| Element | Source |
|---|---|
| Severity cut-offs | AASM Task Force, *Sleep* 1999;22:667–689 |
| Hypopnea scoring rule (1A: ≥30 % airflow ↓ + ≥3 % desat/arousal) | AASM 2012 scoring manual |
| Report structure, section names, phrasing, distractor indices | MTSamples sleep-medicine transcriptions |
| Cohort mix, age, ESS distributions | sleep-clinic cohort (PMC9722997) |
| AHI↔ODI relationship (r≈0.92; moderate-band discordance) | PMC8889990 |
| Mean / nadir SpO₂ and T90 by severity | PMC9719713 |
| AHI↔nadir-SpO₂ correlation (r≈−0.56) | PMC11128192 |
| Hypoxic burden; AHI's null link to CVD mortality | Azarbarzin et al., *Eur Heart J* 2019 |
| Sex ratio rising with severity | O'Connor et al. 2000 (PMID 10806140) |

## 4. Generation — agent orchestration

The data is **authored by LLM agents**, not a numeric simulator. A deterministic
orchestration script only assigns each agent a *stratified target profile*, so 500
independent authors still sum to a literature-realistic cohort. Three phases:

1. **Literature research (11 agents).** Five agents each researched one angle
   (severity definition, epidemiology, physiologic correlations, report realism,
   comorbidities) with live web search (~237 queries); five adversarial verifier
   agents fact-checked every quantitative claim (and corrected several — e.g. a
   wrong AHI–nadir correlation and a mis-attributed citation); one synthesiser
   produced the generation spec.
2. **Pilot (40 agents, 20 records).** Authored + adversarially validated a
   stratified pilot to prove the pipeline; surfaced and fixed one systematic flaw
   (count-vs-rate arithmetic in distractor fields).
3. **Full generation (batched).** Per record: an **author** agent writes the
   columnar row + the tagged report; a **deterministic checker** verifies labels
   and physiologic invariants; an **adversarial validator** agent rates medical
   validity, tag correctness, hardness and realism; a **repair** agent fixes any
   flagged record. Only records passing the deterministic checks are accepted.
   Run in batches to survive API rate limits (`scripts/gen_batch.workflow.js`).

## 5. Validation metrics

- **Label correctness by construction.** Authors wrap the three targets in inline
  `<AHI>/<MEAN>/<NADIR>` tags. A deterministic parser strips the tags, records
  character offsets, and asserts each tagged value equals the columnar value.
  **Result: 0 round-trip failures across all 500 records** (`scripts/persist_batch.py`).
- **Physiologic invariants** enforced on every record: severity == AASM(AHI);
  mean SpO₂ ≥ nadir SpO₂; ODI ≤ AHI; RDI ≥ AHI; REM-AHI & supine-AHI ≥ overall AHI;
  per-class nadir floors; stage percentages sum to 100.
- **Adversarial validation:** medical validity, tags-wrap-the-genuine-target,
  hardness (1–5), realism (1–5). Mean realism ≈ **4.7–4.8 / 5**.
- **Stratification:** severity mix (sleep-lab referral: None .25 / Mild .30 /
  Moderate .22 / Severe .23) with deliberate boundary over-sampling and engineered
  discordant subsets.

## 6. The dataset

500 records, split **361 / 70 / 69** stratified by severity.

**Difficulty composition** (whole set): boundary 195, clean 83, heavy_tail 46,
magnitude_trap 38, spo2_distractor_rich 37, spo2_discordant 26, ess_discordant 26,
odi_discordant 26, duplicate 12, artifact 11.

**Columnar schema** — target `severity`; 17 features:

```
patient_id, ahi, severity, odi, spo2_mean, spo2_nadir, t90, age, gender, bmi,
ess, snoring, arousal_index, rdi, rem_ahi, supine_ahi, heart_rate, scoring_rule,
cpap_recommended            (+ difficulty_flag, metadata only)
```

```
P3000,4.6,None,3.7,96,89,0,47,male,29.4,8,True,4.1,5.3,8.2,7.5,64,AASM 1A (3% hypopnea),False,boundary
P3001,3.2,None,2.6,96,91,0.2,47,female,28.4,6,False,8.5,4.1,5.4,6.1,64,AASM 2012 (...1A),False,clean
```

## 7. Expanding the NER labels (all in-text features)

The 3 targets are labelled by construction (generation tags). To support a
*full-feature*, genuinely text-driven pipeline, the remaining in-text values are
labelled by **gold-anchored auto-labelling** (`scripts/autolabel_multi.py`): for
each feature, a number is tagged only when it both equals the known gold value and
sits next to that feature's cue phrase (e.g. tag "12.7" as ODI only if it ≈ gold
ODI *and* is adjacent to "oxygen desaturation index"). Booleans/categoricals
(gender, snoring, CPAP) are tagged with polarity-encoding labels.

This is weak supervision with ground-truth values, so precision is high; we report
**coverage (recall)**:

| | cov | | cov | | cov |
|---|---:|---|---:|---|---:|
| HEART_RATE | 100 % | AROUSAL_INDEX | 99 % | AGE | 97 % |
| SNORING | 100 % | BMI | 99 % | REM_AHI | 96 % |
| CPAP | 100 % | ESS | 99 % | RDI | 89 % |
| ODI | 99 % | T90 | 99 % | | |
| GENDER | 99 % | SUPINE_AHI | 98 % | | |

Sub-100 % cases are records that genuinely omit that field in the text (correct
behaviour, not error). Records go from **3 → 18.7 entities each** (16 entity types).

## 8. Reproduce

`scripts/gen_batch.workflow.js` (generate) → `scripts/persist_batch.py`
(tag→offset + verify) → `scripts/build_splits.py` (split) →
`scripts/autolabel_multi.py` (expanded labels). Models in `src/`; see
[results.md](results.md).
