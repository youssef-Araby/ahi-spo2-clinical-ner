export const meta = {
  name: 'osa-full-generation-batch',
  description: 'Generate a batch of the v2 OSA dataset: stratified agent-authored columnar + tagged PSG report, deterministic + adversarial validation, one-shot repair, return accepted records',
  phases: [
    { title: 'Author', detail: 'author each stratified record' },
    { title: 'Validate', detail: 'deterministic arithmetic + terse adversarial validator' },
    { title: 'Repair', detail: 'one-shot fix for flagged records' },
  ],
}

// ---- batch window: generation ran in batches to survive API rate limits. ----
// Edit per batch (args do not pass through scriptPath). ids = P(IDOFF + plan_index).
const START = 0      // first plan index for this batch
const COUNT = 100    // records in this batch
const IDOFF = 3000
// optional gap-fill: explicit plan indices to (re)generate; empty -> contiguous [START, START+COUNT)
const ONLY = []

// ---- build the deterministic 500-record stratification plan ----
function buildPlan() {
  const plan = []
  const classes = [['None',125],['Mild',150],['Moderate',110],['Severe',115]]
  let gi = 0
  for (const cls of classes) {
    const sev = cls[0], n = cls[1]
    for (let k = 0; k < n; k++) {
      const i = gi++
      const lm = (i * 5 + k * 3) % 20
      const layout = lm < 8 ? 'prose narrative' : (lm < 15 ? 'mixed prose + key:value table' : 'key:value TABLE layout')
      const dm = (i * 7 + k) % 20
      let diff = 'clean', extra = '', ahiBand
      if (sev === 'None') {
        const b = i % 3
        if (b === 0) { ahiBand = 'AHI in [4.0,4.9] (just under the Mild cutoff)'; diff = 'boundary'; extra = 'borderline-normal; include an RDI slightly above 5 as a near-threshold distractor' }
        else if (b === 1) { ahiBand = 'AHI in [2.5,4.0]'; extra = 'phrase as a normal/negative study, OSA not present' }
        else { ahiBand = 'AHI in [0.5,2.5]'; extra = 'phrase as a normal/negative study, OSA not present' }
      } else if (sev === 'Mild') {
        const b = i % 4
        if (b === 0) { ahiBand = 'AHI in [5.0,6.5] (just over None)'; diff = 'boundary' }
        else if (b === 1) { ahiBand = 'AHI in [13.0,14.9] (just under Moderate)'; diff = 'boundary' }
        else { ahiBand = 'AHI in [7,13]' }
      } else if (sev === 'Moderate') {
        const b = i % 4
        if (b === 0) { ahiBand = 'AHI in [15.0,16.5] (just over Mild)'; diff = 'boundary' }
        else if (b === 1) { ahiBand = 'AHI in [28.0,30.0] (just under Severe)'; diff = 'boundary' }
        else { ahiBand = 'AHI in [18,27]' }
      } else {
        const b = i % 5
        if (b === 0) { ahiBand = 'AHI in [30.1,33.0] (just over Moderate)'; diff = 'boundary' }
        else if (b === 1) { ahiBand = 'AHI in [60,85] (very severe, 2 digits)'; diff = 'heavy_tail' }
        else if (b === 2) { ahiBand = 'AHI in [85,115] (extreme tail, 2-3 digits)'; diff = 'heavy_tail' }
        else { ahiBand = 'AHI in [34,55]' }
      }
      if (dm === 1 || dm === 11) { diff = (diff === 'clean' ? '' : diff + '+') + 'spo2_discordant'; extra += (sev === 'Moderate' || sev === 'Severe') ? ' ; benign nadir (88-92) despite high AHI (stay within floor)' : ' ; surprisingly low nadir near the per-class floor' }
      else if (dm === 3 || dm === 13) { diff = (diff === 'clean' ? '' : diff + '+') + 'ess_discordant'; extra += sev === 'Severe' ? ' ; low ESS (~3-5) despite Severe AHI' : ((sev === 'Mild' || sev === 'None') ? ' ; high ESS (~15-18) despite low AHI' : ' ; ESS discordant with AHI') }
      else if (dm === 5 || dm === 15) { diff = (diff === 'clean' ? '' : diff + '+') + 'odi_discordant'; extra += ' ; ODI sits a class lower than AHI (well below AHI)' }
      else if (dm === 7) { diff = (diff === 'clean' ? '' : diff + '+') + 'magnitude_trap'; extra += ' ; make REM-AHI and supine-AHI markedly LARGER than overall AHI to trap max-grabbers' }
      else if (dm === 9) { diff = (diff === 'clean' ? '' : diff + '+') + 'duplicate'; extra += ' ; state the mean SpO2 (or AHI) twice -- prose AND table -- and tag BOTH occurrences' }
      else if (dm === 17) { diff = (diff === 'clean' ? '' : diff + '+') + 'artifact'; extra += " ; include one realistic transcription artifact (e.g. 'age index' for AHI, or a blank/underscore for a missing distractor field)" }
      else if (dm === 19) { diff = (diff === 'clean' ? '' : diff + '+') + 'spo2_distractor_rich'; extra += ' ; include awake, NREM and REM mean SpO2 plus %TST<90 as % distractors near the targets' }
      const pmaleBySev = { None: 2, Mild: 5, Moderate: 13, Severe: 20 }
      const sexRoll = (i * 9 + k * 4) % 25
      const gender = sexRoll < pmaleBySev[sev] ? 'male' : 'female'
      plan.push({ plan_index: i, id: 'P' + (IDOFF + i), sev: sev, ahiBand: ahiBand, diff: diff || 'clean', extra: extra.trim(), layout: layout, gender: gender })
    }
  }
  return plan
}

const PLAN = buildPlan()
const SLICE = (typeof ONLY !== 'undefined' && ONLY.length) ? ONLY.map(i => PLAN[i]) : PLAN.slice(START, START + COUNT)
log(`Generating records ${START}..${START + SLICE.length - 1} (ids ${SLICE[0] ? SLICE[0].id : '?'}..${SLICE[SLICE.length-1] ? SLICE[SLICE.length-1].id : '?'}) of 500-plan`)

const BRIEF = `You author ONE synthetic obstructive-sleep-apnea patient record: realistic columnar values AND a free-text polysomnography (PSG) report, for a research-backed NER + severity-classification dataset. Follow these rules EXACTLY.

SEVERITY (AASM-1999, deterministic from AHI): None AHI<5; Mild 5<=AHI<15; Moderate 15<=AHI<=30; Severe AHI>30. The 'severity' field MUST equal this function of your chosen AHI.

PHYSIOLOGIC INVARIANTS (never violate):
- spo2_mean >= spo2_nadir ALWAYS.
- odi <= ahi (near-proxy, ~0.84*ahi; lower when discordance is requested).
- rdi >= ahi (RDI = AHI + RERA index).
- rem_ahi >= ahi AND supine_ahi >= ahi (stage/position AHI usually HIGHER than overall) -- DISTRACTORS, never the target.
- Per-class nadir floors so 'hard' stays valid: None nadir>=80, Mild>=78, Moderate>=75, Severe may reach 55-65.
- spo2_mean compressed: None/Mild ~96, Moderate ~95, Severe ~93 (+/-2), floor ~90.

*** ARITHMETIC VALIDITY (the #1 past failure -- obey strictly) ***
- DO NOT state explicit raw COUNTS of arousals or desaturation events (e.g. NOT '71 desaturations', NOT '142 discrete desaturations'). Report these ONLY as rates per hour (arousal index X/hr, ODI X/hr). A raw count that does not equal rate*(TST in hours) is the main validity error -- so omit counts entirely.
- If you state sleep efficiency, it MUST equal round(total_sleep_time / total_recording_time * 100, 1), with TST < TRT. Keep architecture numbers mutually consistent or omit the riskier ones.
- Stage percentages (N1+N2+N3+REM) must sum to 100.

LITERATURE RANGES: nadir medians by class 90/87/84/72; t90 (%TST<90) by class ~0/0.1/0.7/11.8 (severe long tail); BMI mean ~32; age mean ~49.6; ess 0-24 mean ~10 and NEAR-INDEPENDENT of AHI; arousal_index ~0.9*ahi; heart_rate ~60-80 bpm.

THE REPORT (realistic, clinically worded PSG report):
- Sections like Clinical History, Procedure/Protocol, Sleep Architecture, Arousal Summary, Respiratory Events, Oximetry, Impression, Recommendations, signature. Vary order/presence to fit the requested layout.
- Wrap ONLY the three NER targets in inline tags, wrapping ONLY the numeric value (optionally a unit):
    <AHI>{value}</AHI>  = the OVERALL apnea-hypopnea index only
    <MEAN>{value}</MEAN> = the overall/asleep MEAN oxygen saturation only
    <NADIR>{value}</NADIR> = the lowest/nadir oxygen saturation only
  Tagged values MUST equal columnar ahi / spo2_mean / spo2_nadir. If a target is stated twice (prose + table), tag BOTH.
- EMBED as PLAIN UNTAGGED numbers (do NOT tag): odi (/hr), rdi (/hr), arousal_index (/hr), rem_ahi & supine_ahi (/hr, larger than overall AHI), t90 (%TST<90), an awake or stage mean SpO2 (%), heart_rate (bpm). At least 5 distinct distractors must appear as numbers in the text.
- Vary AHI phrasing ('apnea-hypopnea index of X events per hour','AHI X/hr','AH index X'); vary SpO2 phrasing ('mean oxygen saturation X%','Mean SaO2: X%','nadir of X%','Lowest SaO2: X%','bottomed out at X%','trough saturation of X%'). Mix SpO2/SaO2/O2 spellings.
- Single hypopnea convention (3% AASM 1A); may footnote; emit only ONE AHI target number.

Return the columnar row and the tagged report. Medically coherent, and as HARD for NER/classification as the profile demands, while staying valid.`

const AUTHOR_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['patient_id','columnar','report_tagged','difficulty_flag'],
  properties: {
    patient_id: { type: 'string' },
    columnar: { type: 'object', additionalProperties: false,
      required: ['ahi','severity','odi','spo2_mean','spo2_nadir','t90','age','gender','bmi','ess','snoring','arousal_index','rdi','rem_ahi','supine_ahi','heart_rate','scoring_rule','cpap_recommended'],
      properties: {
        ahi: { type: 'number' }, severity: { type: 'string', enum: ['None','Mild','Moderate','Severe'] },
        odi: { type: 'number' }, spo2_mean: { type: 'number' }, spo2_nadir: { type: 'number' }, t90: { type: 'number' },
        age: { type: 'integer' }, gender: { type: 'string', enum: ['male','female'] }, bmi: { type: 'number' },
        ess: { type: 'integer' }, snoring: { type: 'boolean' }, arousal_index: { type: 'number' }, rdi: { type: 'number' },
        rem_ahi: { type: 'number' }, supine_ahi: { type: 'number' }, heart_rate: { type: 'integer' },
        scoring_rule: { type: 'string' }, cpap_recommended: { type: 'boolean' },
      } },
    report_tagged: { type: 'string' },
    difficulty_flag: { type: 'string' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['medically_valid','tags_wrap_correct_target','hardness','realism','verdict','blocking_issues'],
  properties: {
    medically_valid: { type: 'boolean' },
    tags_wrap_correct_target: { type: 'boolean' },
    hardness: { type: 'integer' },
    realism: { type: 'integer' },
    verdict: { type: 'string', enum: ['pass','revise','reject'] },
    blocking_issues: { type: 'array', items: { type: 'string' } },
  },
}

function aasm(a) { return a < 5 ? 'None' : a < 15 ? 'Mild' : a <= 30 ? 'Moderate' : 'Severe' }
function nums(s) { const m = (s || '').match(/-?\d+(?:\.\d+)?/g); return m ? m.map(Number) : [] }
function tagVals(t, tag) { const o = []; const re = new RegExp('<' + tag + '>(.*?)<\\/' + tag + '>', 'gs'); let m; while ((m = re.exec(t)) !== null) { const n = nums(m[1]); if (n.length) o.push(n[0]) } return o }
function near(a, b, tol) { return a != null && b != null && Math.abs(a - b) <= tol }
function grab(t, re) { const m = t.match(re); return m ? Number(m[1]) : null }

function detCheck(a) {
  const issues = []; const c = a.columnar; const t = a.report_tagged || ''
  if (aasm(c.ahi) !== c.severity) issues.push(`severity ${c.severity}!=AASM(${c.ahi})=${aasm(c.ahi)}`)
  if (!(c.spo2_mean >= c.spo2_nadir)) issues.push(`mean ${c.spo2_mean}<nadir ${c.spo2_nadir}`)
  if (c.odi > c.ahi * 1.05) issues.push(`odi ${c.odi}>ahi ${c.ahi}`)
  if (c.rdi < c.ahi - 0.05) issues.push(`rdi ${c.rdi}<ahi ${c.ahi}`)
  if (c.rem_ahi < c.ahi - 0.05) issues.push('rem_ahi<ahi')
  if (c.supine_ahi < c.ahi - 0.05) issues.push('supine_ahi<ahi')
  const floors = { None: 80, Mild: 78, Moderate: 75, Severe: 50 }
  if (c.spo2_nadir < floors[c.severity]) issues.push(`nadir ${c.spo2_nadir} below floor for ${c.severity}`)
  const A = tagVals(t, 'AHI'), M = tagVals(t, 'MEAN'), N = tagVals(t, 'NADIR')
  if (!A.length) issues.push('no <AHI> tag'); else if (!A.some(v => near(v, c.ahi, 0.06))) issues.push(`<AHI>=${A}!=${c.ahi}`)
  if (!M.length) issues.push('no <MEAN> tag'); else if (!M.some(v => near(v, c.spo2_mean, 0.6))) issues.push(`<MEAN>=${M}!=${c.spo2_mean}`)
  if (!N.length) issues.push('no <NADIR> tag'); else if (!N.some(v => near(v, c.spo2_nadir, 0.6))) issues.push(`<NADIR>=${N}!=${c.spo2_nadir}`)
  for (const d of ['odi','rdi','rem_ahi','supine_ahi','arousal_index']) if (A.some(v => near(v, c[d], 0.06)) && !near(c.ahi, c[d], 0.06)) issues.push(`<AHI> equals distractor ${d}`)
  const body = t.replace(/<[^>]+>/g, ' ')
  const present = ['odi','rdi','arousal_index','rem_ahi','supine_ahi','t90','heart_rate'].filter(d => { const v = c[d]; return v != null && nums(body).some(n => near(n, v, Math.max(0.6, Math.abs(v) * 0.01))) })
  if (present.length < 4) issues.push(`only ${present.length} distractor numbers in text (need>=4)`)
  if (/(\d+)\s+(?:discrete\s+)?desaturation/i.test(body)) issues.push('explicit desaturation COUNT present (forbidden -- use rate only)')
  const se = grab(body, /sleep efficiency[^0-9]*(\d+(?:\.\d+)?)/i)
  const tst = grab(body, /total sleep time[^0-9]*(\d+)/i)
  const trt = grab(body, /total recording time[^0-9]*(\d+)/i) || grab(body, /time in bed[^0-9]*(\d+)/i)
  if (se != null && tst != null && trt != null) { const calc = tst / trt * 100; if (Math.abs(calc - se) > 1.6) issues.push(`sleep efficiency ${se}% != TST/TRT ${calc.toFixed(1)}%`) }
  return { pass: issues.length === 0, issues }
}

phase('Author')

const processed = await pipeline(
  SLICE,
  (p) => agent(`${BRIEF}\n\n=== ASSIGNED PROFILE ===\npatient_id: ${p.id}\nTarget severity: ${p.sev}\n${p.ahiBand}\nSex: ${p.gender}\nLayout: ${p.layout}\nDifficulty design: ${p.diff}\n${p.extra ? ('Extra: ' + p.extra) : ''}\n\nAuthor now. Set patient_id="${p.id}". Set difficulty_flag="${p.diff}".`,
    { label: `author:${p.id}`, phase: 'Author', schema: AUTHOR_SCHEMA }),
  async (authored, p) => {
    if (!authored) return { id: p.id, profile: p, accepted: false, reason: 'author null' }
    let rec = authored
    let det = detCheck(rec)
    const vp = `Adversarial reviewer (sleep physician + NLP annotator). Be TERSE: list only BLOCKING issues (max 3), empty if none. Do not praise correct things.\n\nCOLUMNAR: ${JSON.stringify(rec.columnar)}\n\nREPORT:\n"""\n${rec.report_tagged}\n"""\n\nDeterministic checker flagged: ${det.issues.length ? det.issues.join('; ') : 'none'}.\nJudge: medically_valid (values consistent + in real PSG ranges, no count-vs-rate or efficiency arithmetic contradiction); tags_wrap_correct_target (the 3 tags wrap the genuine overall AHI / overall-mean SpO2 / nadir SpO2, never a distractor); hardness 1-5; realism 1-5; verdict pass/revise/reject.`
    let v = await agent(vp, { label: `validate:${p.id}`, phase: 'Validate', schema: VERDICT_SCHEMA })
    const needsRepair = !det.pass || !v || v.verdict !== 'pass' || !v.medically_valid || !v.tags_wrap_correct_target
    if (needsRepair) {
      const allIssues = [...det.issues, ...((v && v.blocking_issues) || [])].slice(0, 6)
      const rp = `Repair this synthetic PSG record. Fix ONLY these issues, preserve everything else (especially the <AHI>/<MEAN>/<NADIR> tags and their values, the columnar targets, and all untagged distractors). Re-emit the FULL corrected record in the same schema.\n\nISSUES:\n${allIssues.map((x, i) => `${i + 1}. ${x}`).join('\n')}\n\nRULES REMINDER: ${BRIEF}\n\nCURRENT COLUMNAR: ${JSON.stringify(rec.columnar)}\nCURRENT REPORT:\n"""\n${rec.report_tagged}\n"""\nSet patient_id="${p.id}", difficulty_flag="${p.diff}".`
      const fixed = await agent(rp, { label: `repair:${p.id}`, phase: 'Repair', schema: AUTHOR_SCHEMA })
      if (fixed) { rec = fixed; det = detCheck(rec) }
    }
    const accepted = det.pass
    return {
      id: p.id, profile: { sev: p.sev, diff: p.diff, layout: p.layout },
      accepted: accepted, repaired: needsRepair, det_issues: det.issues,
      hardness: v ? v.hardness : null, realism: v ? v.realism : null,
      columnar: rec.columnar, report_tagged: rec.report_tagged, difficulty_flag: rec.difficulty_flag,
    }
  }
)

const clean = processed.filter(Boolean)
const accepted = clean.filter(r => r.accepted)
const withH = clean.filter(r => r.hardness)
const withR = clean.filter(r => r.realism)
log(`Batch done: ${accepted.length}/${clean.length} accepted after repair`)
return {
  batch: { start: START, count: SLICE.length, idOffset: IDOFF },
  acceptedN: accepted.length, total: clean.length,
  avg_hardness: withH.reduce((s, r) => s + r.hardness, 0) / Math.max(1, withH.length),
  avg_realism: withR.reduce((s, r) => s + r.realism, 0) / Math.max(1, withR.length),
  records: accepted,
  rejects: clean.filter(r => !r.accepted).map(r => ({ id: r.id, issues: r.det_issues })),
}
