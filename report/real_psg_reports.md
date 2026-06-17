# Reference: how real polysomnography (PSG) reports are written

Source: de-identified transcribed reports from MTSamples (Sleep Medicine specialty).
These are used as the structural + phrasing basis for our note generator so the
synthetic notes read like real transcriptions, not AI prose.

## 1. Canonical section structure (order varies slightly)

```
PROCEDURE:                 e.g. "Sleep study." / "Overnight polysomnogram."
CLINICAL INFORMATION:      age, sex, presenting symptoms, comorbidities, study date, height, weight
SLEEP QUESTIONNAIRE:       subjective sleep estimate (optional)
STUDY PROTOCOL:            equipment + electrode montage (near-boilerplate, reused across reports)
TECHNICAL QUALITY:         "Good." / "Fair." / "Adequate."
ELECTROPHYSIOLOGIC / SLEEP ARCHITECTURE:
                           total recording time, total sleep time, sleep latency, REM latency,
                           sleep efficiency, stage I/II/III/REM percentages
RESPIRATORY MEASUREMENTS:  apnea + hypopnea counts, AHI, REM AHI, RDI, arousal index,
                           oxygen desaturation / nadir, longest event
ELECTROCARDIOGRAPHIC:      heart rate asleep / awake
CONCLUSIONS / DIAGNOSES:   severity statement, ICD code (e.g. 780.53-0)
RECOMMENDATIONS:           CPAP titration, weight loss, ENT eval, avoid alcohol/sedatives
```

Style notes for realism:
- Telegraphic fragments, not flowing prose: "Total recording time 406 minutes, total sleep time 365 minutes, sleep latency 25.5 minutes."
- ALL-CAPS section headers followed by a colon.
- Reports are FULL of numbers besides AHI/SpO2 (distractors): dates, weight, height,
  heart rate, arousal index, PLM index, stage %, ICD codes, electrode labels (C4-A1).
- Occasional transcription artifacts: blanks `____`, mis-transcriptions like "age index" (= AH index).

## 2. Verbatim phrasing banks for the TARGET metrics
(observed forms, lightly templated with {v} = value)

### AHI (apnea-hypopnea index)
- "an overall apnea-hypopnea index of {v} events per hour"
- "apnea/hypopnea index {v}"
- "with apnea/hypopnea index {v}"
- "Severe obstructive sleep apnea with apnea/hypopnea index {v}"
- "Total apnea/hypopnea {n}, AH index {v} per hour"
- "AHI {v}" / "AHI of {v}/hour" / "AHI was {v} events/hr"
- (rare transcription noise) "age index {v} per hour"
- REM-specific: "REM AHI {v} per hour"

### Mean / average oxygen saturation
- "Mean oxygen saturation {v}%"
- "mean SpO2 {v}%"
- "average oxygen saturation was {v}%"
- "Oxygen saturation during awake {v}%"   (awake baseline variant)

### Lowest / nadir oxygen saturation (desaturation)
- "lowest oxygen saturation {v}%"
- "The lowest oxygen level reached was {v}%"
- "Oxygen desaturation was down to {v}%"
- "oxygen saturation nadir of {v}%"
- "SpO2 nadir {v}%"

## 3. Distractor phrasings (NOT labeled — present to make NER hard)
- "respiratory disturbance index {v}"            (RDI)
- "arousal index {v} per hour" / "{n} arousals with index {v}"
- "Periodic limb movements ... overall index of {v} events per hour"
- "Stage I {a}%, stage II {b}%, stage III {c}%, and REM stage {d}%"
- "Total sleep period {x} minutes, total sleep time {y} minutes" ; "sleep efficiency {e}%"
- "Heart rate while asleep {lo} to {hi} per minute, while awake {lo2} to {hi2} per minute"
- "{w} pounds, {ft} feet {inch} inches tall"
- ICD: "(780.53-0)" etc.
- study date "MM/DD/YY"

## 4. Real example excerpts (for tone)

> "Obstructive apneas and hypopneas were identified with an overall apnea-hypopnea
> index of 15.2 events per hour ... The lowest oxygen level reached was 88%."

> "306 apneas and hypopnea with apnea/hypopnea index 76 ... Mean oxygen saturation
> 91% with lowest oxygen saturation 70%. A 19% of sleep time was spent with oxygen
> saturation less than 90% and 1% with less than 80%."

> "Total apnea/hypopnea 75, age index 12.3 per hour. REM age index 15 per hour.
> Total arousal 101, arousal index 15.6 per hour. Oxygen desaturation was down to 88%."
