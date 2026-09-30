# 24 GPU-hour experimental plan

**Thesis under test.** Steering a frozen base refusal direction across OLMo-3's
training flow involves three quantities, not two:

| | quantity | measured behaviour |
|---|---|---|
| lever | logit-gap displacement per unit dose | intact (CV 0.078; null z 4.4-7.0) |
| margin | unsteered distance to the boundary | grows 9.5x raw, 12.4x scale-free |
| budget | largest dose whose generations stay fluent | flat (c_half spread 1.42x) |

Steering succeeds iff `c_cross = margin / lever <= c_budget`. Measured ratio
(c_half / c_cross): base 2.71, SFT-1k 1.06, DPO 0.62, RLVR-last 0.60,
Instruct 0.62. **Alignment does not blunt the lever and does not shrink the
budget; it moves the target past the budget.**

## Measured unit costs

| operation | cost | source |
|---|---|---|
| model load | ~40 s | `load_seconds` in sweep files |
| logit-readout forward pass | 0.09 s / prompt-dose | sweep `elapsed_seconds` |
| generation, 512 new tokens | 1.2 s / completion | `gateA_gen_*` `seconds` |
| HarmBench-13B load / judge | 55 s / 0.08 s per item | coherence run log |
| full 23-dose x 200-prompt sweep | 0.13 h | measured |

A 15-dose x 100-prompt generation sweep at 512 tokens = **0.50 h per checkpoint**.
This dominates the cost of everything below.

## The gap this plan exists to close

The budget was measured on a **c50-relative** grid (`c50 x [0.5, 1, 1.5, ...]`),
so the smallest dose ever tested scales with c50: base 0.136, Instruct 0.612,
4.5x coarser. For DPO / RLVR / Instruct the thresholded budget came out exactly
equal to their first nonzero dose, i.e. bounded above by the grid floor and
never resolved. Interpolated `c_half` partly rescues this (base and Instruct
coherence curves overlay to within 0.031 at matched absolute dose), but
Instruct's c_half is interpolated across a **0.61-wide gap** while base's is
bracketed within 0.136. **The aligned models are precisely the ones whose budget
is least well measured.** Everything downstream rests on fixing that.

---

## Block A - the gate (3.6 h)

### A1. Fine absolute-dose coherence sweep, 5 checkpoints | 2.5 h

Common ABSOLUTE grid `c in {0.1, 0.2, ..., 1.4}` (15 doses incl. c=0), 100
held-out harmful prompts, 512 new tokens, on base / SFT-1k / DPO / RLVR-last /
Instruct. Every model measured on the same axis, with no interpolation doing
load-bearing work.

- **Under thesis:** coherence-vs-dose curves overlay across checkpoints; c_half
  stays within ~1.5x while c_cross spans 4.5x.
- **Under negation:** curves separate and c_half falls monotonically with
  alignment. Robustness would then be *also* increasing fragility, and the
  thesis ("budget flat, target moved") is wrong as stated.
- **Falsifier:** c_half spread across checkpoints comparable to the c_cross
  spread (>3x).
- **Why first:** it is the only number the whole reframe rests on that is
  currently unresolved.

### A2. Coherence measured three ways | 0.6 h

Reuses A1's generations. (i) repetition / distinct-3 (CPU, free), (ii)
perplexity under a held-out model, (iii) an LLM fluency judge. Report c_half
under each.

- **Under thesis:** the three agree on c_half to within the grid step.
- **Falsifier:** c_half depends on which proxy is used, so the budget is a proxy
  artifact and the thesis is not measurable as stated.
- Answers reviewer R1 ("a fluency criterion that is not fully specified").

### A3. Mechanism geometry (`scripts/mechanism.py`) | 0.5 h

B2 residual-stream projection onto the frozen base direction across 6 layers;
B3 per-checkpoint direction refit vs base cosine; B4 the exact decomposition
`margin = |h| * |u| * cos(h,u)`.

- B4 is an identity, so it **cannot** return inconclusive: it must attribute the
  12x growth to |h|, |u| or cos. h is post-final-norm, so |h| is near-fixed by
  construction and the growth must land in |u| (readout amplified) or cos
  (representation rotated toward the readout).
- **Falsifier for the control arm:** if RL-Zero shows the same geometry change
  without margin growth, SFT is not the cause and the control breaks.

> **GATE 1.** Proceed only if A1 shows the budget flat and A2 shows it
> proxy-independent. If the budget shrinks with alignment, stop and rewrite the
> thesis before spending the remaining 20 hours.

---

## Block B - the readout objection (3.8 h)

### B1. Extend A1 to all 10 checkpoints | 2.5 h

Adds SFT-15k, SFT-43k, RLVR-first and both RL-Zero variants. The RL-Zero pair is
the control arm: no SFT, no margin growth, so the thesis predicts their ratio
stays above 1 like base's.

### B2. Readout robustness | 1.3 h

Re-run the logit sweep scoring alternative onset token sets: (a) drop `" I"`
(it opens both "I cannot help" and "I'd be happy to help", so it is largely
refusal-agnostic), (b) a larger refusal vocabulary, (c) a behavioural-only
readout. Save the **full logit vector** at the readout position so future token
sets can be rescored with zero GPU.

- **Under thesis:** the 12x margin growth and the flat lever survive every token
  set.
- **Falsifier:** the growth is specific to the original three tokens.
- Answers reviewer R2 ("the distance is read from one choice of opening words,
  so the proposed correction relies on the same proxy").

---

## Block C - what the budget is, and whether the law generalizes (4.5 h)

### C1. Is the budget direction-specific? | 2.0 h

Generate under 5 random unit directions at matched absolute doses, on base / DPO
/ Instruct.

- If random directions break fluency at the **same** c as the refusal direction,
  the budget is a property of the model's tolerance for any perturbation. That
  is the cleanest version of the thesis and makes the budget a model constant.
- If the refusal direction is cheaper or dearer than random, the budget is
  direction-specific and must be reported per-direction.
- Both outcomes are publishable but support different claims. Note the
  random-direction floor is significant at Instruct only (paired t = 7.8).

### C2. Honesty: the within-concept control | 2.5 h

Honesty's margin grows only 1.2x. The thesis therefore predicts its ratio stays
**above 1 at every checkpoint** and it remains steerable where refusal is not.

- **Under thesis:** honesty stays steerable throughout; the ratio never crosses 1.
- **Under negation:** honesty also becomes unsteerable, which would mean the
  budget rather than the margin is doing the work.
- This is the sharpest single prediction the thesis makes: the two concepts share
  a model, a readout style and a budget, and differ in margin growth.

---

## Block D - generality (4.0 h)

### D1. Cross-family budget and ratio | 4.0 h

Qwen3-8B, Llama-3.1-8B, Gemma-2-9B, OLMo-3 32B, base->instruct pairs (8 loads).
Does `ratio >= 1` predict within-budget steerability in every family, given that
the *lever* behaves differently across them (flat on OLMo, growing on Qwen and
Llama)?

- **Under thesis:** the ratio predicts steerability even where the lever varies,
  because the ratio already contains the lever.
- Answers reviewer R3 ("evidence outside the main model family is thinner than
  the summary implies").

---

## Block E - causality (4.0 h)

### E1. LoRA-SFT induction (`scripts/b7_induce.py`) | 4.0 h

Fine-tune base on refusal-formatted data drawn from a pool **disjoint from both
splits** (the script asserts this on raw strings and refuses to run otherwise).
Measure margin, lever, budget and cos-to-base at steps 0/25/50/100/200/400.

Now a **three**-quantity prediction, much sharper than the original two: margin
opens, lever flat, **budget flat**, ratio crosses 1.

Pre-stated falsifiers (already in the script): F1 margin does not open, so SFT is
not the cause and the RL-Zero contrast has another explanation; F2 lever tracks
the margin, so the two are not separable; F3 direction rotates (cos < 0.7), so
what was induced is not the released phenomenon.

**Blocked on two decisions:** wiring an additive-steering helper into
`pipeline.py` (it currently lives in `e1_causal.py`), and choosing the disjoint
SFT pool.

---

## Allocation

| block | hours | cumulative | deadline |
|---|---|---|---|
| A - gate (budget resolved, proxy-independent, mechanism) | 3.6 | 3.6 | abstract |
| B - all checkpoints + readout robustness (R2) | 3.8 | 7.4 | preprint |
| C - direction-specificity + honesty control | 4.5 | 11.9 | preprint |
| D - cross-family (R3) | 4.0 | 15.9 | preprint |
| E - causal induction | 4.0 | 19.9 | ICML |
| **slack for reruns** | **4.1** | **24.0** | - |

The slack is not padding. Three results in this project have died to audits after
looking finished; budget for re-running one block.

## Not doing, and why

- **Re-running the HarmBench harm judgments** with the corrected upstream
  template. The omission under-counts hedged compliance, which *understates* the
  base-vs-aligned contrast rather than manufacturing it, so it cannot flip a
  conclusion. Cheap (~0.5 h) if a block finishes early.
- **More concepts beyond refusal and honesty.** Two concepts differing in margin
  growth is the controlled comparison; a third adds breadth, not inference.
- **Layer sweep of the budget.** The budget is defined on generations, which
  depend on all layers, so a per-layer budget is not well defined.
- **Anything at 32B beyond D1.** Scale is a robustness check here, not a claim.

## Zero-GPU work to do first (free)

1. Rescore existing E2 generations with a perplexity-based coherence measure, as
   a partial preview of A2 before spending any GPU.
2. Fit the ratio-to-steerability relation on the 5 checkpoints already measured,
   so A1 has a pre-registered prediction to confirm or break.
3. Audit pass on `coherence_budget.py` and `lever.py` before A1 consumes GPU.
