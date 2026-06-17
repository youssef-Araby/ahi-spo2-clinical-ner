# Results

All metrics on the held-out **test split (69 reports)** of the 500-record dataset
(train 361 / val 70 / test 69). NER uses strict span scoring (exact span + label)
via `src/evaluate.py`; the classifier uses the columnar features.

## 1. NER — 3 core targets (AHI / SpO₂-mean / SpO₂-nadir)

"v1" = the earlier easy template data; **v2 = this dataset**.

| Method | v1 micro-F1 | **v2 micro-F1** | v2 macro-F1 | v2 value-acc | v2 note-exact |
|---|---:|---:|---:|---:|---:|
| Regex (rules) | 0.823 | **0.491** | 0.483 | 0.744 | 0.319 |
| CRF | 0.940 | **0.892** | 0.893 | 0.947 | 0.841 |
| Bio_ClinicalBERT | 0.991 | **0.942** | 0.942 | 0.957 | 0.870 |

Per-label F1 (v2): regex AHI 0.562 / mean 0.333 / nadir 0.554; CRF 0.838 / 0.894 /
0.948; ClinicalBERT 0.933 / 0.951 / 0.943.

The dataset is markedly harder than v1: the rule baseline **collapses** (recall 0.343 —
distractors and phrasing variety defeat hand-written patterns), CRF drops ~5 points, and
ClinicalBERT falls from a near-ceiling 0.991 to 0.942.

> **Note.** ClinicalBERT required sliding-window tokenisation — reports are ~970 tokens
> (median), and the original 256-token truncation removed every target (the model scored
> 0.000 until fixed). `src/model_clinicalbert.py` now windows (384 tokens, stride 128)
> and stitches the predictions.

### Per-distractor confusion (does the trap bite?)

Predicted entities classified as correct / matched-a-known-distractor / other:

| Method | predicted | correct | distractor-confusions | other |
|---|---:|---:|---:|---:|
| Regex | 170 | 155 | 5 | 10 |
| CRF | 441 | 421 | 13 | 7 |
| ClinicalBERT | 454 | 440 | 0 | 14 |

- **CRF** is lured by the AHI look-alikes: AHI←RDI ×4, AHI←ODI ×4, AHI←arousal ×4.
- **Regex** confuses AHI with REM-AHI ×5 and otherwise fails by *missing* (low recall).
- **ClinicalBERT** makes **0** columnar-distractor confusions; its 14 "other" errors are
  wrong spans / in-text non-columnar values, spread across labels (6 AHI, 2 mean, 6 nadir).

## 2. NER — full 16-entity set (Bio_ClinicalBERT)

Labelling all in-text features (methodology §7) gives a richer benchmark.

| Entity | F1 | | Entity | F1 |
|---|---:|---|---|---:|
| AROUSAL_INDEX | 1.000 | | T90 | 0.945 |
| BMI | 1.000 | | ODI | 0.936 |
| ESS | 1.000 | | AHI | 0.929 |
| GENDER | 1.000 | | REM_AHI | 0.917 |
| AGE | 0.985 | | RDI | 0.832 |
| SPO2_NADIR | 0.959 | | CPAP_YES | 0.719 |
| SUPINE_AHI | 0.958 | | CPAP_NO | 0.646 |
| SPO2_MEAN | 0.955 | | SNORING_PRESENT | 0.938 |
| HEART_RATE | 0.978 | | **SNORING_ABSENT** | **0.000** |
| | | | **micro 0.936 / macro 0.872** | |

### Note-level exact match (the hard metric: every target in the note correct)

| Scope | value-exact | span-exact |
|---|---:|---:|
| Core 3 targets (AHI/mean/nadir) | **1.000** | 0.884 |
| All 16 entity types | **0.594** | 0.493 |

**Key finding — multi-task labelling helps the core task.** With the *same metric and
test set*, core-3 note-exact rises from **0.870** (3-label model) to **1.000** (16-label
model). Teaching the model that ODI/RDI/REM-AHI are their *own* entities stops it
stealing them for AHI, so the original targets get cleaner. The effect is consistent
across both data versions (497-split: 0.696 → 0.986; 500-split: 0.870 → 1.000); with only
69 test notes the absolute values are noisy, but the direction is stable.

The all-16 note-exact (0.594) is the demanding number, held back almost entirely by the
**booleans**: `SNORING_ABSENT` (F1 0.000 — the rare negative class, 8 test cases, with
polarity carried by negation elsewhere in the sentence) and CPAP polarity (~0.65).
Encoding an assertion as a token label is the wrong tool; this is a known limitation (§5).

## 3. Severity classifier (columnar → AASM 4-class)

Trained on gold columnar features. Three conditions, because several features are
near-deterministic functions of AHI and leak the label.

| Condition | features | logreg macro-F1 | RandomForest macro-F1 |
|---|---|---:|---:|
| with_ahi (tautology ceiling) | 15 | 0.986 | **1.000** |
| proxies_no_ahi (ODI/RDI/REM/supine/arousal) | 14 | 0.986 | **1.000** |
| physiology_only (oximetry + demographics) | 9 | 0.899 | **0.971** |

RandomForest, physiology-only, by difficulty subset: **boundary 0.91**, all other subsets
1.00. Confusion: only 2 errors, both **Moderate→Mild at the boundary** (None, Mild, Severe
all perfect).

**Interpretation.** The classifier is *not* hard in aggregate — even physiology-only
reaches 0.97, because the data faithfully reproduces the real per-class oximetry
separation, so oxygen features genuinely track severity. Difficulty concentrates in the
boundary subset. The classifier's role here is the pipeline endpoint and ceiling, not a
hard standalone task.

## 4. End-to-end pipeline (report → NER → severity)

### 4a. AHI-only (severity = AASM(extracted AHI))

| Pipeline | AHI coverage | severity acc | macro-F1 (covered) |
|---|---:|---:|---:|
| **Ceiling** — gold AHI → severity | 100 % | 1.000 | 1.000 |
| regex → severity | 100 % | 0.957 | 0.954 |
| CRF → severity | 98.6 % | 0.942 | 0.961 |
| **ClinicalBERT → severity** | 98.6 % | **0.986** | 1.000 |

ClinicalBERT loses only ~1.4 points vs the ceiling — severity bucketing absorbs small AHI
extraction errors.

### 4b. Full-feature (every feature from the 16-entity NER, nothing gold)

| Classifier inputs | ceiling (gold) | end-to-end (NER) |
|---|---:|---:|
| Full feature set | 1.000 | **1.000** |
| Full set minus CPAP (leakage control) | 1.000 | **1.000** |

A genuinely text-driven pipeline — every classifier input is extracted from the report —
still hits the ceiling, because of **feature redundancy**: even when AHI extraction slips,
the extracted ODI/RDI/REM-AHI reinforce the right severity band.

## 5. Limitations

- **Severity is defined from AHI**, so the classifier is near-trivial in aggregate; the
  honest contribution is the physiology-only condition + the difficulty-stratified
  breakdown, and the NER stage that feeds it.
- **Boolean extraction is weak** (`SNORING_ABSENT` 0.0, CPAP ~0.65). Polarity-as-label is
  the wrong design; proper negation/assertion detection (or report-level binary
  classification) is the fix, and would raise the all-16 note-exact.
- **Small test set (69 notes)** — single-split point estimates are noisy (e.g. the 3-label
  core-3 note-exact moved 0.70→0.87 between splits); trends are reported across both splits.
- **Clinician sign-off** still pending on the per-class SpO₂/ESS plausibility floors.
- `cpap_recommended` **leaks** the label (it is a consequence of severity); reported with
  and without it — it does not change the headline since accuracy is at ceiling.
