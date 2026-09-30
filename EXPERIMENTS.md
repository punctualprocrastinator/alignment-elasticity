# Coherence-budget thesis: experimental plan (24 GPU-hours)

**Thesis.** Steering a frozen base refusal direction across OLMo-3's training
flow involves three quantities, not two:

| | quantity | measured behaviour |
|---|---|---|
| lever | logit-gap displacement per unit dose | intact (CV 0.078; null z 4.4-7.0) |
| margin | unsteered distance to the boundary | grows 9.5x raw, 12.4x scale-free |
| budget | largest dose whose generations stay fluent | flat (c_half spread 1.42x) |

Steering succeeds iff `c_cross = margin / lever <= c_budget`. Alignment does not
blunt the lever and does not shrink the budget; it moves the target past the
budget.

---

## Research questions

**RQ1.** Is the coherence budget actually flat across the training flow, and is
it independent of how fluency is measured?

**RQ2.** Does the steerability ratio `c_budget / c_cross` predict whether a model
can be steered within budget?

**RQ3.** Where does the 12x margin growth live (representation or readout), and
is SFT the cause?

**RQ4.** Is the budget a property of the model, or of the steering direction?

**RQ5.** Does the law generalize across concepts, model families, and scale?

---

## Metrics

**RQ1:** coherent fraction vs absolute dose; `c_half` (dose where coherent
fraction crosses 0.50, interpolated); spread of c_half across checkpoints vs
spread of c_cross; agreement of c_half under three independent fluency measures
(repetition/distinct-3, perplexity under a held-out model, LLM fluency judge).

**RQ2:** ratio = c_half / c_cross per checkpoint; within-budget refusal-rate drop;
whether `ratio >= 1` separates steerable from non-steerable checkpoints; the same
under alternative readout token sets.

**RQ3:** the exact decomposition `margin = |h| * |u| * cos(h,u)`; residual-stream
projection onto the frozen base direction across layers; cosine between each
checkpoint's refit direction and the base direction; under LoRA-SFT induction,
trajectories of margin, lever, budget and cos-to-base vs fine-tuning step.

**RQ4:** c_half under random unit directions vs under the refusal direction at
matched absolute dose; the per-direction random floor (paired over the shared 20
directions).

**RQ5:** ratio and within-budget steerability for honesty (margin grows only
1.2x) vs refusal; the same for Qwen3-8B, Llama-3.1-8B, Gemma-2-9B, OLMo-3 32B.

---

## Experiments we already have

**RQ1:** thresholded budget on a c50-relative grid (5 checkpoints x 7 doses x 80
generations); threshold-free `c_half`; base and Instruct coherence curves shown
to overlay to within 0.031 at matched absolute dose.

**RQ2:** ratio computed for 5 checkpoints (base 2.71, SFT-1k 1.06, DPO 0.62,
RLVR-last 0.60, Instruct 0.62); within-budget refusal drop at thresholds 0.85
down to 0.30, base dominating at every one.

**RQ3:** margin growth localized in training, not mechanism - the two RL-Zero
checkpoints (RL on base, no SFT) show no margin growth (d' 0.44-0.53 vs base
0.45) while SFT-1k already reaches 2.43. `scripts/mechanism.py` written, unrun.

**RQ4:** random-direction floor per checkpoint from the existing sweeps
(significant at Instruct only, paired t = 7.8). No random-direction generations.

**RQ5:** margin growth for four families (3.7x-9.8x) and honesty (1.2x), all on
the logit readout only. No budget measured for any of them.

---

## Experiments we want to do

Roughly 20 GPU-hours of the 24, ordered so RQ1 gates everything else.

### RQ1 | 5.6 h

- Fine **absolute**-dose sweep: `c in {0.1, ..., 1.4}` (15 doses), 100 held-out
  harmful prompts, 512 new tokens, on base / SFT-1k / DPO / RLVR-last / Instruct.
  Every model measured on the same axis, no interpolation doing load-bearing
  work. **2.5 h**
- Extend the same grid to all 10 checkpoints, adding SFT-15k, SFT-43k, RLVR-first
  and both RL-Zero variants (the control arm: no SFT, so the thesis predicts
  their ratio stays above 1 like base's). **2.5 h**
- Score those generations with three independent fluency measures and report
  c_half under each. **0.6 h**

> *Thesis predicts:* curves overlay, c_half spread ~1.5x against c_cross 4.5x,
> and the three measures agree to within the grid step.
> *Negation predicts:* curves separate, c_half falls monotonically with
> alignment - robustness would then be increasing fragility too.
> *Falsifier:* c_half spread comparable to c_cross spread (>3x), or c_half
> depending on which fluency proxy is used.

### RQ2 | 1.3 h

- Re-run the logit sweep scoring alternative onset token sets: drop `" I"` (it
  opens both "I cannot help" and "I'd be happy to help"), a larger refusal
  vocabulary, and a behavioural-only readout. Save the **full logit vector** at
  the readout position so future token sets rescore with zero GPU. **1.3 h**
- Recompute ratio and within-budget steerability under each readout (free, on
  RQ1's generations).

> *Thesis predicts:* the 12x margin growth, the flat lever and the ratio's
> crossing of 1 all survive every token set.
> *Falsifier:* the growth is specific to the original three tokens.
> Answers reviewer R2 ("the distance is read from one choice of opening words").

### RQ3 | 4.5 h

- Run `scripts/mechanism.py`: B2 residual-stream projection across 6 layers, B3
  per-checkpoint direction refit vs base cosine, B4 the decomposition
  `margin = |h| * |u| * cos(h,u)`. **0.5 h**
- LoRA-SFT induction (`scripts/b7_induce.py`) on a pool disjoint from both
  splits, measuring margin / lever / budget / cos-to-base at steps
  0/25/50/100/200/400. **4.0 h**

> B4 is an identity, so it cannot return inconclusive: it must attribute the 12x
> to |h|, |u| or cos. h is post-final-norm, so |h| is near-fixed by construction
> and the growth must land in |u| (readout amplified) or cos (representation
> rotated toward the readout).
> *Induction predicts:* margin opens, lever flat, **budget flat**, ratio crosses 1.
> *Falsifiers (pre-stated in the script):* margin does not open, so SFT is not
> the cause; lever tracks the margin, so the two are not separable; cos < 0.7,
> so what was induced is not the released phenomenon.

### RQ4 | 2.0 h

- Generate under 5 random unit directions at matched absolute doses, on base /
  DPO / Instruct, and compute c_half for each. **2.0 h**

> If random directions break fluency at the **same** c, the budget is the model's
> tolerance for any perturbation - the cleanest version of the thesis, and it
> makes the budget a model constant. If the refusal direction is cheaper or
> dearer than random, the budget is direction-specific and must be reported
> per-direction. Both are publishable; they support different claims.

### RQ5 | 6.5 h

- Honesty as the within-concept control: fine absolute-dose sweep for the honesty
  direction on the same 5 checkpoints. **2.5 h**
- Cross-family: Qwen3-8B, Llama-3.1-8B, Gemma-2-9B, OLMo-3 32B, base->instruct
  pairs (8 loads). **4.0 h**

> Honesty is the sharpest prediction in the plan: same model, same readout style,
> same budget, and margin growth of only 1.2x, so the thesis predicts its ratio
> **never crosses 1** and it stays steerable where refusal is not. If honesty
> also becomes unsteerable, the budget rather than the margin is doing the work.
> Cross-family tests whether the ratio predicts steerability even where the lever
> behaves differently (flat on OLMo, growing on Qwen and Llama). Answers R3.

---

## The gate

**After RQ1's first 3.1 h (5 checkpoints + three measures), stop and check.**
Proceed only if the budget is flat and proxy-independent. If it shrinks with
alignment, the thesis is wrong as stated and the remaining ~17 hours should be
re-planned rather than spent.

This gate exists because the budget was measured on a **c50-relative** grid whose
floor scales with c50 (base 0.136, Instruct 0.612, 4.5x coarser). For DPO, RLVR
and Instruct the reported budget came out exactly equal to their first nonzero
dose - bounded above by the grid floor, never resolved. Interpolated c_half
partly rescues it, but Instruct's c_half is interpolated across a 0.61-wide gap
while base's is bracketed within 0.136. The aligned models are precisely the ones
whose budget is least well measured.

---

## Allocation

| RQ | hours | cumulative | deadline |
|---|---|---|---|
| RQ1 - budget flat and proxy-independent | 5.6 | 5.6 | abstract / preprint |
| RQ2 - readout robustness and the ratio | 1.3 | 6.9 | preprint |
| RQ3 - mechanism and causal induction | 4.5 | 11.4 | preprint / ICML |
| RQ4 - direction-specificity of the budget | 2.0 | 13.4 | preprint |
| RQ5 - honesty, families, scale | 6.5 | 19.9 | preprint / ICML |
| **slack for reruns** | **4.1** | **24.0** | - |

The slack is not padding. Three results in this project have died to adversarial
audits after looking finished; budget for re-running one block.

### Measured unit costs

| operation | cost | source |
|---|---|---|
| model load | ~40 s | `load_seconds` in sweep files |
| logit-readout forward pass | 0.09 s / prompt-dose | sweep `elapsed_seconds` |
| generation, 512 new tokens | 1.2 s / completion | `gateA_gen_*` `seconds` |
| HarmBench-13B load / judge | 55 s / 0.08 s per item | coherence run log |
| full 23-dose x 200-prompt sweep | 0.13 h | measured |

A 15-dose x 100-prompt generation sweep at 512 tokens = 0.50 h per checkpoint,
which dominates the cost of everything above.

---

## Not doing, and why

- **Re-running the HarmBench harm judgments** with the corrected upstream
  template. The omission under-counts hedged compliance, which *understates* the
  base-vs-aligned contrast rather than manufacturing it, so it cannot flip a
  conclusion. Cheap (~0.5 h) if a block finishes early.
- **More concepts beyond refusal and honesty.** Two concepts differing in margin
  growth is the controlled comparison; a third adds breadth, not inference.
- **Layer sweep of the budget.** The budget is defined on generations, which
  depend on all layers, so a per-layer budget is not well defined.
- **Anything at 32B beyond RQ5.** Scale is a robustness check here, not a claim.

## Zero-GPU work to do first (free)

1. Rescore the existing E2 generations with a perplexity-based coherence measure,
   as a partial preview of RQ1's third measure before spending any GPU.
2. Fit the ratio-to-steerability relation on the 5 checkpoints already measured,
   so RQ1 has a **pre-registered** prediction to confirm or break rather than a
   post-hoc fit.
3. Audit pass on `coherence_budget.py` and `lever.py` before RQ1 consumes GPU.

## Blocked on a decision

RQ3's induction needs an additive-steering helper wired into `pipeline.py` (it
currently lives in `e1_causal.py`) and a disjoint SFT pool chosen. The script
asserts disjointness on raw strings and refuses to run otherwise.
