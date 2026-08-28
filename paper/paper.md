# The Refusal Direction Does Not Go Stale — The Margin Moves

### Steering efficacy is invariant across the OLMo-3 training flow; fixed-magnitude audits fail only because alignment widens the behavioural margin

*Workshop submission — InterpScience @ NeurIPS 2026. Draft, 2026-08-29.*

---

## Abstract

A refusal direction fit on a base language model is widely reported to "go stale" on the
model's aligned descendants: ablate the direction and the base model complies, but the
aligned model still refuses. We show the direction does not weaken at all. Measuring the
displacement a fixed steering dose produces along the model's own refusal readout, we find
this **steering efficacy is nearly invariant across the entire OLMo-3 flow** — pretraining,
SFT, DPO, and RLVR — varying by only 1.5× (coefficient of variation 0.13) while the model's
**behavioural margin grows 9.5×**, from 0.81 to 7.75 logits. A fixed-magnitude intervention
therefore fails on aligned models for one banal reason: the same push is now small against a
much wider margin, not because the direction lost its grip. The consequence is a measurement
artifact that inverts a natural ranking — under fixed-magnitude ablation and under fixed-dose
steering the aligned model ranks *least* controllable, yet dosed in proportion to its margin
it crosses cleanly, its refusal rate falling from 1.00 to 0.18, the largest behavioural swing
of any checkpoint. Three prior reports that base directions "remain effective," "steer
better," and "go stale" are reconciled: they are one invariant direction seen through
instruments that confuse **distance to the decision boundary** with **resistance to being
moved.** Safety audits that ablate a fixed direction systematically understate the
controllability of aligned models, and the error grows with how thoroughly a model was
aligned.

---

## 1. Introduction

Activation steering — finding a direction in the residual stream that correlates with a
behaviour and adding or subtracting it — is a standard tool for both interpreting and
controlling language models. A consequential question follows for anyone auditing a deployed
model: **does a direction found on an earlier checkpoint still control the behaviour after the
model is aligned?** If white-box safety interventions built on base-model analysis silently
expire during alignment, they expire exactly when they matter.

The published answers disagree. One line of work reports that base-derived steering vectors
**remain effective** on fully post-trained instruct models. Another reports that post-training
directions steer **better** than pretraining ones. A third — including our own earlier
ablation experiments — reports that the base refusal direction **loses its grip**: ablate it
and the aligned model keeps refusing. Three groups, three answers, one model family.

We argue the disagreement is instrumental, and we resolve it with a single measurement. Every
one of these studies fixes the *magnitude* of the intervention and reads off the *outcome*.
But a fixed intervention is not a fair probe of two models whose behaviour sits at very
different distances from the decision boundary. An aligned model refuses hard: its internal
margin in favour of refusal is large, so a fixed nudge toward compliance moves it a long way
and still leaves it refusing. The base model sits near indifference, so the same nudge tips it
over. Read naively, the direction looks "stronger" on the base model — a statement about where
the two models start, not about how much a unit of steering moves them.

The right quantity is **steering efficacy**: the displacement a unit of dose produces along
the model's own readout, independent of where the model starts. We measure it across the full
OLMo-3 flow and find it essentially constant while the behavioural margin grows an order of
magnitude. That one fact — invariant efficacy, growing margin — explains all three prior
observations, predicts exactly when fixed-magnitude audits mislead, and is the paper.

Section 2 sets up the measurement. Section 3 is the instrument artifact (Figure 1). Section 4
is the invariance result (Figure 2). Section 5 states what the measurement cannot decide.

---

## 2. Setup

**Models.** The OLMo-3 7B family (Ai2), whose entire training flow is public: the base model;
the Think lineage at SFT steps 1k/15k/43k, DPO, and the first and last RLVR steps; the
Instruct endpoint; and two RL-Zero variants (Math, Code) that apply reinforcement learning
*directly to the base model with no SFT*. Every checkpoint is pinned by commit hash.

**Direction.** A refusal direction at layer 20, fit **once on the base model** as the
difference in mean residual-stream activation between 200 harmful (AdvBench) and 200 benign
(Alpaca) prompts under a neutral scaffold applied identically to every checkpoint. The *same
base direction* is carried unchanged to every other model — this is the staleness question. We
fit it two ways, difference-in-means and logistic regression (`tol=1e-10`); they sit at cosine
0.74 and give the same conclusions, so we report difference-in-means.

**Steering and readout.** We use the input-normalised parameterisation `h_l ← h_l + c·μ_l·v̂`
(μ_l = mean residual norm at layer 20), so the coefficient c is comparable across checkpoints.
The behavioural readout is the **refusal logit gap**: log-probability of a refusal-onset token
minus a compliance-onset token at the first generated position (token ids fixed and reported).

**The two quantities.** For each model we record its **margin** — the mean unsteered refusal
gap, i.e. how far the model sits from indifference — and its **steering efficacy** — the
displacement of that gap produced per unit dose, estimated as the slope of mean displacement
against |c| in the near-zero regime where displacement is a function of dose. (Displacement
saturates and reverses at large |c|; we use only the linear region.)

**Rigour.** Every headline number carries a 95% bootstrap CI over prompts (1000 resamples;
paired for between-model comparisons). A 20-direction random-direction null band accompanies
every steering curve; the real direction clears it at z = 4–7 everywhere. Prompt splits are
fingerprinted and every run reproduces the same fingerprint (`99a7ac88`); the base model
reproduces its numbers bit-for-bit across sessions. Five pipeline smoke tests pass (hook
liveness, per-layer variation, truncation audit, degenerate-direction rejection,
random-direction control).

---

## 3. The instrument decides the answer (Figure 1)

Rank the same four models — base, Instruct, and the two RL-Zero variants — by "how steerable
is refusal," and the ranking depends entirely on the instrument.

Under **fixed-magnitude ablation** — remove the base direction outright, the intervention
behind the "it goes stale" claim — the Instruct model ranks **last**: its refusal survives.
Under **fixed input-norm dosing** at a single coefficient, Instruct again ranks **last**.
These are the two instruments the prior literature used, and both say the aligned model is the
hardest to move.

But dosed in proportion to its own margin, the Instruct model crosses from refusing to
complying as readily as any checkpoint: behaviourally its refusal rate falls from **1.00 to
0.18**, the **largest swing of any model** (base falls 0.80→0.62 at its own smaller
appropriate dose). The aligned model is not resistant to being moved; a fixed-magnitude
instrument simply applies too small a push to traverse its wider margin. Section 4 shows that
the push-per-dose was never the thing that changed.

> **Figure 1 — Three instruments, three rankings of the same four models.**
> Each panel ranks base, Instruct, and two RL-Zero variants by refusal steerability under a
> different instrument. (a) Fixed-magnitude ablation and (b) fixed input-norm dosing both place
> the aligned Instruct model *last* — the basis for "base directions go stale." (c) When each
> model is dosed to reach its own decision boundary, Instruct crosses cleanly (refusal
> 1.00→0.18, the largest behavioural swing here). The instruments disagree because the first two
> confuse distance-to-boundary with resistance-to-being-moved. Bars are bootstrap means;
> whiskers 95% CIs. `fig_threeInstruments.png`.

---

## 4. Steering efficacy is invariant; the margin moves (Figure 2)

If the aligned model is not harder to move per unit dose, then how much *does* a unit of dose
move each checkpoint, and how much does that change across training? We measure steering
efficacy and margin at ten points spanning the flow.

**Efficacy is nearly constant.** Across all ten checkpoints — base, six Think-lineage
post-training stages, Instruct, and both RL-Zero variants — the displacement produced per unit
dose ranges only **3.6 to 5.6 (a factor of 1.5, coefficient of variation 0.13)**. The base
refusal direction moves every descendant by essentially the same amount per unit of steering,
whether that descendant has undergone SFT, DPO, RLVR, or RL directly on the base. It clears
the random-direction null at every checkpoint (z = 4–7), and every generation-tested model
crosses behaviourally. **As a causal control axis, the direction does not decay.**

**The margin grows tenfold.** Over the same checkpoints the unsteered refusal margin climbs
from **0.81 at base to 7.75 at Instruct — a factor of 9.5** — rising steeply through SFT and
continuing through DPO and RLVR. The model refuses from ever deeper inside its own boundary.

These are the two moving parts, and only one of them moves. A fixed-magnitude intervention
delivers a fixed displacement (efficacy × dose); as the margin it must overcome grows an order
of magnitude while that displacement stays fixed, the intervention crosses fewer and fewer
prompts. The direction is not going stale — the target is receding. This is the whole
mechanism, and it requires nothing about rotation, redundancy, or representational
reorganisation: a constant lever against a growing load.

> **Figure 2 — The direction's steering power is invariant; only the margin grows.**
> Steering efficacy (displacement produced per unit dose; teal, right axis) is flat across
> pretraining → SFT → DPO → RLVR, range 3.6–5.6, CV 0.13. The behavioural margin the direction
> must overcome (clay, left axis) grows 9.5× over the same span, from 0.81 to 7.75 logits. A
> fixed-magnitude ablation delivers a fixed displacement, so it fails on aligned models purely
> because the same push is now small against a far wider margin. `fig_efficacy_margin.png`.

**The reconciliation.** "Base vectors remain effective" is true — efficacy is invariant. "They
go stale under fixed ablation" is also true — the margin grew and a fixed removal no longer
reaches the boundary. "Post-training directions steer better" is the same fact from the other
side — refit at a later checkpoint, a direction is calibrated to that checkpoint's larger
margin. The three reports are one invariant lever measured through instruments that read the
load as the lever.

---

## 5. What the measurement cannot decide

- **Efficacy is a per-dose displacement, not a full causal characterisation.** It establishes
  that a unit of dose moves the readout by a constant amount across training; it does not
  isolate whether the direction's *alignment* with the causal boundary changes in ways that
  cancel at the level of mean displacement. We claim invariance of the measured lever, not of
  every geometric property of the direction.
- **We cannot compare base and aligned models at matched starting distance.** Aligned models
  have essentially no prompts near their boundary (all sit at margin 3.9–7.8), so there is no
  overlapping regime in which to ask "at equal distance, is the aligned model equally
  steerable." The efficacy comparison sidesteps this by measuring the per-dose slope, which is
  defined regardless of where prompts sit; a distance-matched comparison is not available and
  we do not claim one.
- **One direction-fitting family, one layer, one lineage, one scale.** Difference-in-means and
  logistic agree despite cosine 0.74, but we did not test a supervised causal direction fit by
  gradient on the behavioural gap, which could in principle be a more efficient lever on base.
  All results are layer 20, 7B, OLMo-3; replication at other layers, scales, and lineages is
  future work.
- **A note on an appealing dead end.** We initially summarised the effect with an
  "excess displacement" statistic (dose needed past each model's own boundary). It correlates
  0.93 with the spread of the model's unsteered-gap distribution, so it largely restates the
  distribution's shape rather than a steering property; we report efficacy instead, which is
  distribution-free. The negative result is in the appendix as a caution.

---

## 6. Conclusion — so what

The practical consequence is immediate and cheap to act on. **A safety audit that probes a
model by ablating a fixed direction, or steering at a fixed magnitude, will rate an aligned
model as less controllable than it is — and the misrating grows with how thoroughly the model
was aligned.** The models we most want to stress-test for latent steerability are precisely the
ones a fixed-magnitude instrument most understates, because alignment widens the margin without
touching the lever. The fix costs nothing: dose relative to the model's own unsteered margin,
which the same sweep already measures.

The scientific consequence is that a small, well-measured invariant dissolves a three-way
disagreement. The base refusal direction does not go stale, does not need refitting to stay a
valid control axis, and does not steer "better" after alignment — it steers *the same*, per
unit dose, from the start of pretraining through RLVR. What changes is how far the model has
walked from its own decision boundary. Measured in the right units, "does the direction survive
alignment?" has a clean answer: the direction was never the thing that moved.

---

*Reproducibility. All code, pinned checkpoint commits, split fingerprints, seeds, per-experiment
result JSON, figure scripts, and the red-team appendix (including the excess-displacement
negative result) are released. Every headline number carries a bootstrap CI and a
random-direction null; the base model reproduces bit-for-bit across sessions.*
