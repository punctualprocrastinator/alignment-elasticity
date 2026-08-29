# Red-team of the Gate A paper (2026-08-29)

All checks below are **zero-GPU re-analyses of the existing per-prompt sweep JSONs**
(`gaps_massmean` is [n_c x 200] per checkpoint).

## The decisive problem: excess-d50 is largely a gap-distribution-shape statistic

`excess = d50 - median(unsteered gap)`. Across all 6 refusal checkpoints:

    corr(excess, std of unsteered gap)  = +0.927
    corr(excess, skew of unsteered gap) = +0.830

| model | excess | std_g0 | skew_g0 | frac(g0<0) |
|---|---|---|---|---|
| base | 0.979 | 1.79 | +0.48 | 0.355 |
| sft-1000 | -0.032 | 1.56 | -0.39 | 0.020 |
| sft-15000 | 0.013 | 1.33 | -0.22 | 0.000 |
| instruct | 0.037 | 1.38 | -1.27 | 0.000 |
| rlz-math | 1.082 | 1.85 | +0.41 | 0.340 |
| rlz-code | 0.957 | 1.88 | +0.36 | 0.345 |

The high-excess models (base, both RL-Zero) all have a **wide, right-skewed gap
distribution that straddles zero** — ~35% of prompts already lean toward complying.
The excess-0 models are all deep and concentrated in refusal. Excess is essentially
a proxy for "is the unsteered refusal-gap distribution wide and near zero."

### What this kills
- **Section 5 mechanism ("SFT aligns probe and control axes"): unsupported.** The
  per-prompt test of proportional control, `corr(displacement, g0)`, does NOT track
  excess: sft-1000 has *worse* proportional control than base (-0.243 vs -0.149) yet
  collapsed excess. Cut this claim.
- **"The aligned model is the EASIEST to steer (lowest excess)": overclaim.** Low
  excess = concentrated gap distribution, not intrinsic ease of steering.
- **Heterogeneity was ruled out (good):** per-prompt displacement is uniform (CV ~0.15,
  frac_move 1.0) in every model, so excess is not a displacement-heterogeneity artifact.
  It is a *starting-distribution* artifact instead.

### "Refusal-specific" reduces to a distribution statement
base REFUSAL g0: std 1.79, skew +0.48, frac<0 0.35. base SENTIMENT g0: std **0.55**,
skew +0.15, frac<0 **0.00**. So "refusal shows base excess, sentiment does not" is
really "base is genuinely ambivalent about ~35% of harmful prompts pre-alignment,
while its sentiment classification is confident." True and mildly interesting, but a
claim about base refusal *behaviour*, not steering geometry.

## What SURVIVES (the defensible nucleus)
1. **Figure 1 (a,b) is solid:** fixed-magnitude ablation and fixed input-norm dosing
   rank the aligned model LAST.
2. **The cause is real:** the aligned model sits far from its boundary (unsteered gap
   7.87 vs 0.85).
3. **The aligned model is not intrinsically resistant:** a boundary-matched dose crosses
   it (behaviourally 1.00 -> 0.18).
4. **The practical claim holds and needs no excess statistic:** fixed-dose steering/
   ablation audits understate the controllability of aligned models, and the error grows
   with alignment. This is the paper.

## Required reframing
- Lead with the audit-misranking result (Figure 1). It is clean, useful, distribution-free.
- Demote excess-d50 to a *diagnostic*, and DISCLOSE its ~0.93 correlation with gap spread.
- Drop the Section 5 mechanism and the "easiest to steer" language.
- Recast Figure 2: excess collapses at SFT because SFT moves the whole gap distribution
  deep into confident refusal — which is the known alignment behavioural transition, not a
  new geometric event. Either frame it honestly as that, or cut it.

## The one experiment that makes the core claim bulletproof (ZERO GPU)
**Gap-matched steerability.** For each model, take only prompts in a fixed unsteered-gap
band (e.g. g0 in [0.5, 1.5]) and measure the coefficient / displacement needed to cross
*those* prompts. This controls distribution shape directly. If the aligned model crosses
its matched-distance prompts as easily as base does, the "not resistant, just distant"
claim becomes distribution-free and unattackable. All data is already on disk.

## Experiments worth ~a day (GPU), in priority order
1. **Supervised causal direction** (~30 min): fit the L20 direction by gradient on the
   behavioural gap, not diff-in-means. Tests whether base "inefficiency" is just
   diff-in-means being a lazy control axis. Already flagged in the draft's Section 6.
2. **Per-prompt crossing-coefficient distribution, gap-matched** (zero GPU, do first).
3. **A second safety concept (honesty direction)** through the same pipeline: is it
   "refusal-specific" or "safety-concepts-general"? ~30 min.
4. Finer early-SFT checkpoints (if any < step 1000 exist) to earn the word "abrupt".
5. Layer robustness of the misranking (extraction at 3 more layers).

---

# RESOLUTION (2026-08-29): the distribution-free core is EFFICACY-INVARIANCE

Two zero-GPU re-analyses run.

**1. Gap-matched steerability FAILS by construction — and that failure is informative.**
Post-SFT models have essentially NO prompts in a near-boundary band (they all sit at
gap 3.9–7.9), so base and Instruct have no overlapping starting distances to match. You
cannot compare them "at matched distance" because the aligned model has no near-boundary
prompts. Among the models that DO have band prompts (base, rlz-math, rlz-code), crossing
coefficient is identical (~0.14). So "matched-distance equality" is unprovable for the
base-vs-aligned comparison, and must NOT be claimed.

**2. Steering EFFICACY is invariant; only the MARGIN changes. THIS is the paper.**
Displacement produced per unit dose (slope of mean displacement vs |c|, near-zero region):

| quantity | range across all 10 checkpoints |
|---|---|
| unsteered margin | 0.81 .. 7.75  (**9.5x**) |
| steering efficacy | 3.63 .. 5.61  (**1.54x**, CV 0.127) |

The base refusal direction delivers nearly constant displacement per unit dose across the
ENTIRE flow (efficacy CV 0.13) while the margin it must overcome grows ~10x. Every model
clears the random-direction null (z 4–7) and every generation-tested model crosses
behaviourally. **The direction never goes stale as a control axis; the target moves.**
Fixed-magnitude ablation / fixed-c steering fail on aligned models for one reason: a fixed
push is now small relative to a 10x-larger margin.

## The rebuilt paper (distribution-free, no excess-d50 as headline)
- **Claim:** the base refusal direction's per-dose steering efficacy is invariant across
  pretraining -> SFT -> DPO -> RLVR (CV 0.13); what grows is the behavioural margin (9.5x).
  Fixed-magnitude steering/ablation audits therefore mis-read aligned models as
  "un-steerable" when the direction is in fact undiminished — the error is entirely the
  margin, and it grows with alignment.
- **Figure 1:** three instruments, three rankings (unchanged — still the hook).
- **Figure 2 (rebuilt):** efficacy (flat) vs margin (rising 10x) across the flow, twin axis.
  Replaces the excess-d50 trajectory.
- **excess-d50:** demoted to a methods appendix with its 0.93 gap-spread confound disclosed,
  OR cut entirely.
- **Sentiment / "refusal-specific":** drop as a headline; the honest version is a one-line
  observation that base is behaviourally ambivalent about ~1/3 of harmful prompts.
- **Reconciliation stands and is now correct:** "base vectors remain effective" (efficacy
  invariant — TRUE), "they go stale under fixed ablation" (margin grew — TRUE). Same object.

---

# BOX A RESULTS (2026-08-29) — E1/E2/E3

**E1 (causal direction): invariance is NOT a diff-in-means artifact.** Efficacy CV across
base/sft-1000/dpo/instruct: diff-in-means 0.184, logistic 0.191, **gradient-trained causal
0.196** — all <0.25. The gradient lever is ~6x more efficient (26.1 vs 4.5) and largely
orthogonal to diff-in-means (cos 0.21), yet equally invariant. Invariance is a property of
the direction as a control axis, not of the fitting method. Attack (i) answered.

**E3 (layers): invariance holds at mid-late layers; L20 representative, not special.**
Per-layer efficacy CV across checkpoints: L8 1.09, L12 1.27, L16 0.55, L20 0.19, L24 0.11,
L28 0.07. Early layers reorganise with alignment; L20/24/28 are invariant (L24/28 flatter
than L20). Attack (iii) answered.

**E2 (behavioural dose-response): THE IMPORTANT NUANCE — a dissociation the paper must adopt.**
5 checkpoints x 7 doses x 80 prompts, generation + HarmBench-13b-cls on 20/cell.
- **Refusal-ONSET behavioural efficacy IS invariant (CV 0.149)** while margin grows 0.76->7.88
  and c-to-halve grows 0.40->0.95. So the onset lever is non-circular — it holds on real
  generations, not just the logit gap. (Instruct at its c50: refusal 0.21, reproduces Gate A.)
- **BUT genuine-harm controllability DECAYS with alignment.** HarmBench-judged harmful peak:
  base 0.80 -> sft 0.60 -> dpo 0.45 -> rlvr 0.40 -> **instruct 0.05**. Flipping the aligned
  model's onset ("Sure, here...") no longer yields genuinely harmful content; heavy dose just
  degenerates it.
- **The prefix refusal classifier increasingly OVERSTATES compliance on aligned models:**
  corr(prefix-comply, HarmBench-harmful) falls 0.96 -> -0.53. This is itself a methods warning
  for the many papers using prefix-based refusal metrics.

**Consequence for the paper — two warnings, both useful, more honest than the clean version:**
1. (holds) Fixed-magnitude audits misrank aligned models as least controllable because the
   margin grows 9.5x while the per-dose ONSET lever is invariant (E1/E2-onset/E3).
2. (new) Onset-controllability and genuine-harm-controllability DISSOCIATE: the same lever that
   still flips an aligned model's refusal onset no longer elicits genuine harm, and onset/prefix
   metrics overstate compliance on aligned models. Auditors must not read onset-flips as harm.
Scope the invariance claim to the onset/representation lever; report the behavioural-harm decay
and the prefix-metric failure explicitly. Do NOT claim "aligned models are just as steerable
for harm."

---

# BOX B RESULTS (2026-08-30) — E4/E5/E6: the phenomenon generalizes, with honesty as the causal control

**E4 (honesty concept): efficacy-invariance replicates MORE strongly; margin barely grows —
this is the causal control for the whole paper.** Truth direction (Azaria-Mitchell, L20, dim
+ logistic, cos 0.16), carried unchanged across base/sft-1k/dpo/rlvr-last/instruct.
- efficacy 9.41/9.47/9.68/9.79/9.88, **CV 0.021** (refusal 0.13), z 9-13 vs null.
- **margin grows only 1.21x** (2.66->3.21) vs refusal's 9.5x. Models do not become
  dramatically more confidently honest.
- **=> the fixed-dose audit artifact is driven by MARGIN growth, which is concept-specific:**
  huge for refusal, near-absent for honesty. Honesty is the controlled contrast proving margin
  growth (not the lever) causes the misranking. Onset/genuine dissociation for honesty:
  inconclusive (independent frame flips as much as label frame; no clean label-only lever).

**E5 (Qwen3-8B, second family): margin-growth + audit-misranking + the onset/harm dissociation
all GENERALIZE; strict efficacy-invariance does NOT.** Refusal direction fit on Qwen3-8B-Base
(proportional layer 22 of 36), carried to Qwen3-8B.
- margin grows up to **6.2x** (2.21 base -> 8.15 instruct-neutral -> 13.67 instruct-chat).
- base direction still drives aligned refusal ONSET to zero (0.99->0.00 base, 1.00->0.00 chat).
- **onset-vs-genuine-harm dissociation REPLICATES:** HarmBench harm at c50 rises 0.03->0.75 on
  base but only 0.00->0.20 (chat) on the aligned model.
- **Scope limit:** efficacy is only APPROXIMATELY invariant off OLMo (3.23->8.33) and Qwen's
  random-direction null is wide (z 1.6-2.1). Strict efficacy-invariance is an OLMo property;
  margin-growth + misranking + dissociation are the general ones.

**E6 (OLMo-3 32B, scale): flat-efficacy/growing-margin REPLICATES.** Layer 40 of 64, R^5120
null, SFT in the 5e-5 lineage (P9 trap). efficacy 5.49/5.58/5.92/5.95 (**CV 0.041**, z 9.7-12.6);
margin grows 2.3x (more gradual; run ends at RLVR-last, no separate Instruct endpoint).

## Net picture for the paper (all axes)
- **Robust everywhere:** fixed-dose audits misrank aligned models because ALIGNMENT WIDENS THE
  MARGIN, not because the direction weakens. Margin grows: refusal 9.5x (7B), 6.2x (Qwen),
  2.3x (32B); the base direction still flips the aligned onset to ~0 in every case.
- **Causal control (honesty):** where margin barely grows (1.2x), the audit artifact is minimal
  -> margin growth is the driver, demonstrated not just asserted.
- **Second warning, also general (Qwen):** onset-control != genuine-harm-control; flipping the
  aligned onset does not yield harm; prefix metrics overstate compliance on aligned models.
- **Honest scope limits:** strict per-dose efficacy-invariance is clean on OLMo (7B CV 0.13,
  32B CV 0.04, honesty CV 0.02) but only approximate on Qwen (wide null); honesty's margin
  barely grows.

Title stays family-agnostic: "Distance to the Boundary Is Not Steering Resistance."

---

# FOUR-FAMILY GENERALIZATION (2026-08-30) — Llama + Gemma added (unsloth full-precision mirrors)

verify_fp confirmed full precision on every arm (Llama 8.03B, Gemma 9.24B, bf16, no
quantization_config); commits pinned; mirror caveat recorded per file.

| family | margin-growth | onset-flip | onset↔harm dissociation | strict efficacy-invariance |
|---|---|---|---|---|
| OLMo-3-7B | Y (9.5x) | Y | Y | **Y (CV 0.13)** |
| Qwen3-8B | Y (6.2x) | Y | Y (base 0.03→0.75, instr →0.20) | approx (~2x; wide null z≈2) |
| Llama-3.1-8B | Y (8.5–9.8x) | Y* degeneration | N (harm→0 everywhere) | N (grows ~4x; z≈1–2) |
| gemma-2-9b | Y (5.9–7.4x) | Y (chat 1.0→0.24) | Y (base 0.25→0.42, instr →0) | approx (5.2→6.2; z≈1.4–2.9) |

**THE RESCOPING THIS FORCES — and it makes the paper stronger, not weaker.**
Efficacy-invariance is NOT the universal claim (strict only on OLMo; Gemma approximate;
Llama's lever even GROWS ~4x, and its diff-in-means direction barely clears the null on base,
z≈1). The UNIVERSAL, 4-family claim is:

**Alignment widens the behavioural margin (5.9–9.8x in every family), which is SUFFICIENT on
its own to make fixed-dose audits misrank aligned models — regardless of what the direction's
own steering power does.** On OLMo the lever is provably invariant, so the misranking is PURELY
margin; on Llama the lever even strengthens; in no family does margin-growth fail to cause the
misranking. Efficacy-invariance becomes the cleanest DEMONSTRATION (OLMo, echoed approximately
in Gemma) that the direction need not change at all for the audit to fail — not the headline.

- Honesty (margin 1.2x, no artifact) is the causal control: margin growth is the driver.
- Dissociation (onset≠harm) holds on OLMo/Qwen/Gemma; Llama is the extreme form — the onset
  "flip" is pure degeneration (repeating tokens), no genuine harm anywhere. Strengthens the
  prefix-metric warning rather than weakening it.
- Off-OLMo the random-direction efficacy null is wide (z 1–3), so per-dose efficacy is a
  noisier axis there; report it honestly and lead with margin.

**Paper edits required:** abstract + Section 5 lead with margin-growth as the universal
mechanism across 4 families; efficacy-invariance demoted to "cleanest on OLMo, approximate on
Gemma, family-dependent elsewhere"; add the unsloth-mirror caveat and Llama's growing/noisy
lever to limitations.
