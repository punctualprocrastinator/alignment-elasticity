# Two-box, 12h-each experiment plan (2026-08-29 → 1 Sep deadline)

**Budget:** 2 × RTX PRO 6000 Blackwell, ~12 GPU-h each (24 total). **Two days to
submission**, so every experiment is gated: each outcome must either strengthen a
specific claim or be reportable as a limitation. No open-ended exploration.

**Paper's spine to defend:** steering *efficacy* (displacement per unit dose) is
invariant across the OLMo-3 flow (CV 0.13) while the *margin* grows 9.5×; fixed-dose
audits misrank aligned models for that reason alone.

**The four reviewer attacks we must answer:**
- (i) efficacy-invariance is a diff-in-means artifact → E1
- (ii) efficacy is measured on the logit gap the direction was fit to — circular → E2
- (iii) only layer 20 → E3
- (iv) only refusal, only 7B, only OLMo → E4, E5

**Robustness protocol (every run, non-negotiable — 5 boxes have died this month):**
pull each artifact the instant it exists; poll INLINE (background tasks are killed at
turn boundaries here); probe HTTP 410 before any retry; pin every commit; mirror cell
bodies to `scripts/notebook_cells/`; restore `pipeline.py`+`gate_a*.py` byte-identical;
reproduce fingerprint `99a7ac88` on refusal runs or STOP.

---

## BOX A — "core robustness" (defends the spine on OLMo refusal)

### E1 — Supervised causal direction (~1 h) — answers attack (i)
Fit the L20 refusal direction three ways on base: diff-in-means (have it), logistic
(have it), and a **gradient-trained steering vector** (optimise a single L20 vector to
minimise refusal-logit-gap on harmful prompts, held-out eval). Recompute efficacy +
margin for all three across base / SFT-1k / DPO / Instruct.
- **Decision:** if efficacy stays invariant (CV < 0.25) for the causal direction too →
  the invariance is not a diff-in-means artifact; report all three. If the causal
  direction is far more efficient on base only → base "inefficiency" was diff-in-means
  laziness; reframe as "even an optimised base lever gains nothing per-dose from
  alignment." Either is a clean paragraph.

### E2 — Behavioural dose–response (~4 h) — answers attack (ii), the deepest
Efficacy so far is displacement of the refusal *logit gap*; a skeptic calls that
near-circular. Route it through the full model instead: **generation refusal rate vs
dose**. For 5 checkpoints (base, SFT-1k, DPO, RLVR-last, Instruct) × 7 coefficients ×
80 held-out harmful prompts, greedy 256 tok (512 for Think), classify refusal onset,
HarmBench-cls spot-check on a subset.
- **Metric:** behavioural efficacy = slope of refusal-rate vs achieved displacement,
  and c-to-halve-refusal, each vs the model's margin.
- **Decision:** if behavioural efficacy is ~invariant while margin grows → the core
  claim holds on real generations, non-circular — this becomes Figure 2b and the
  paper's strongest evidence. If behavioural efficacy *falls* with alignment → the
  logit-gap efficacy was misleading; the paper narrows to "logit-level lever constant,
  behavioural lever decays," still publishable and honest. Highest-value single run.

### E3 — Layer robustness (~3 h) — answers attack (iii)
Extract at layers {8,12,16,20,24,28}, refit the base refusal direction per layer, run
the efficacy/margin sweep for base / SFT-1k / Instruct at each.
- **Decision:** efficacy-invariance (per layer, across checkpoints) should hold at most
  mid–late layers. Report the layer profile; if L20 is special, say so. A heatmap of
  efficacy(layer × checkpoint) is a strong appendix figure.

*(Box A stretch if time: reproduce the three prior instruments' native metrics for the
reconciliation figure — but this is mostly zero-GPU, see "local tasks".)*

---

## BOX B — "generalization" (defends breadth)

### E4 — Second concept: honesty (~2.5 h) — answers attack (iv, concept)
Full efficacy/margin trajectory for a **honesty** direction (Azaria–Mitchell true/false,
L20) across base / SFT-1k / SFT-43k / DPO / RLVR-last / Instruct, same protocol.
- **Decision:** does "efficacy invariant, margin grows" replicate for a second safety
  property? If yes → the phenomenon is not refusal-specific, broadening the audit
  warning. If honesty's margin does *not* grow (models don't become more confidently
  honest) → contrast is itself informative: the artifact tracks *behavioural
  confidence growth*, which is exactly the point. Either sharpens the claim.

### E5 — Second model family (~3 h) — answers attack (iv, lineage/family)
Does "margin grows, efficacy constant" hold OFF OLMo? Take a base→instruct pair from
another open family with both public (e.g. Qwen3-8B base vs instruct, or Llama-3.1-8B
base vs instruct). Fit the refusal direction on that base, measure efficacy + margin on
base and instruct. Only 2 points, but a different tokenizer, data, and RLHF recipe.
- **Decision:** if margin grows and efficacy holds on a second family → the audit
  implication is general, not an OLMo quirk (big for the "so what"). If it does not
  replicate → scope the claim to OLMo and report the difference honestly. Risk:
  chat-template/tokenizer friction; box B budgets this as the uncertain one.

### E6 — 32B scale (STRETCH, ~4 h, download-heavy) — answers attack (iv, scale)
Only if E4+E5 land with time left: efficacy/margin on OLMo-3 **32B** base / SFT / DPO /
RLVR-last (recompute the R^5120 nulls). 4 points showing the same flat-efficacy /
growing-margin shape kills the "one scale" objection. Drop first if B runs behind.

---

## LOCAL (zero-GPU, I run these in parallel while boxes work)
- **E7 — Reconciliation figure:** reproduce each prior instrument's *native* metric
  from existing sweeps (fixed-ablation crossing, fixed-c input-norm crossing, refit-at-
  checkpoint) and show all three give their published answers, then efficacy resolves
  them. Pure re-analysis of `results/gateA_*`.
- Re-render Figure 1/2 to final quality; fold E-results into the draft as they land.
- Excess-d50 appendix (the negative result) — already drafted in `redteam.md`.

---

## Sequencing & integration
- **Both boxes start immediately on E1/E2 (A) and E4/E5 (B)** — the four attack-answers.
- Pull results continuously; I integrate each into `paper/paper.md` the moment it lands,
  so the paper is always in a submittable state (no big-bang merge at hour 24).
- **Hard stop for new experiments: 31 Aug 12:00**, leaving a full day for writing,
  final figures, and the reproducibility appendix.
- Priority if compute runs short: E2 > E1 > E4 > E3 > E5 > E6. E2 is the one that
  makes the core claim non-circular; protect it.

---

## Tool ideas assessed 2026-08-30: SAEs and J-lens — HOLD for the full paper

**Pretrained SAEs (`decoderesearch/olmo-3-saes`): not for the workshop paper.**
Base-only (L4/16/28), so cannot track a direction across the flow — orthogonal to the
efficacy/margin claim. Good ingredient for a separate frozen-dictionary-survival paper.

**J-lens: the natural MECHANISM for E2's dissociation — but hold for the full version.**
Hypothesis worth testing later: the refusal direction retains its grip on the
verbalizable/ONSET workspace across the whole flow (-> invariant onset lever, our E1/E2/E3)
but loses its grip on the sustained GENERATION TRAJECTORY (-> the HarmBench harm decay,
E2). J-lens measures exactly "what token is this activation disposed to make the model say,"
so it can adjudicate this. Doing it right needs (a) a chat-fit lens (the pre-fitted ones are
wikitext, a confound for chat-tuned models) and (b) a non-safety control direction — the two
controls that demoted the earlier E8 J-space result to inconclusive. That is a maximal-version
section, not a 2-day bolt-on; adding it now would reintroduce cleaned-out confounds and break
the one-idea focus.
- **Cheap low-risk exception if E4/E5 finish early:** a WITHIN-checkpoint illustration on
  Instruct alone — decode the refusal direction through Instruct's own lens (refusal
  vocabulary = onset grip intact) beside its actual safe generations (trajectory grip lost).
  One figure, no cross-checkpoint lens confound, ~30 min. Illustrative, explicitly subordinate.
