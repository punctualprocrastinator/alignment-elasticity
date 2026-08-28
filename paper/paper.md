# Distance to the Boundary Is Not Steering Resistance

*Workshop submission — InterpScience @ NeurIPS 2026. Draft, 2026-08-29.*

---

## Abstract

A steering vector fit on a base language model is often reported to "go stale" on the
model's post-trained descendants: ablate the refusal direction and the base model complies,
but the aligned model still refuses. We show this conclusion is an artifact of the
instrument, not a property of the representation. The standard test applies a **fixed**
intervention to models that sit at very different distances from their own decision boundary
— the aligned OLMo-3 model's unsteered refusal margin is **+7.87 logits** against the base
model's **+0.85** — so a fixed push flips the shallow model and fails to flip the deep one
while doing nothing to the direction itself. When we instead dose steering in units of
**displacement achieved past each model's own boundary**, the ranking inverts: the aligned
model crosses at an excess displacement of **0.04 [−0.25, 0.24]** versus the base model's
**0.98 [0.66, 1.27]**, and behaviourally shows the *largest* refusal swing of any checkpoint
(1.00→0.18). Tracing this excess along the full OLMo-3 flow, it collapses from ~1.0 to ~0
**abruptly at the first supervised-fine-tuning step** and stays flat through DPO and RLVR; a
non-safety (sentiment) direction shows no comparable base excess, so the effect is specific
to the refusal direction. The reading: on the base model the difference-in-means refusal
direction is an excellent *probe* but an inefficient *steering* axis, and SFT — not RL —
aligns the two. Fixed-magnitude steering audits systematically understate the controllability
of aligned models, and the error grows with alignment.

---

## 1. Introduction

Activation steering has become a standard tool for both interpreting and controlling language
models: find a direction in the residual stream that correlates with a behaviour, add or
subtract it, and watch the behaviour change. A natural and consequential question follows for
anyone auditing a deployed model: **does a direction found on an earlier checkpoint still
control the behaviour after the model is fine-tuned and aligned?** If it does not, white-box
safety interventions built on base-model analysis would silently expire exactly when they
matter most.

The published answers disagree. One line of work reports that base-derived steering vectors
**remain effective** on fully post-trained instruct models. Another reports that post-training
directions steer **better** than pretraining ones. A third — including our own earlier
ablation experiments — reports that the base refusal direction **loses its grip** on the
aligned model: ablate it and the aligned model keeps refusing. Three groups, three answers,
one model family.

We argue the disagreement is not empirical but instrumental. Every one of these measurements
fixes the *size* of the intervention and reads off the *outcome*, and a fixed intervention is
not a fair comparison across models that sit at different distances from their own decision
boundary. An aligned model refuses hard: its internal margin in favour of refusal is large,
so a fixed nudge toward compliance moves it a long way and still leaves it refusing. The base
model sits close to indifference, so the same nudge tips it over. Read naively, the base
direction looks "stronger" on the base model — but that is a statement about where the two
models start, not about how controllable they are.

This paper makes the comparison boundary-fair. We dose steering in units of *displacement
actually achieved along the readout*, measure the displacement each model needs to move its
own median prompt **past its own boundary** (we call this the **excess displacement**), and
find that the story reverses cleanly, survives across the entire OLMo-3 training flow, and is
specific to the refusal direction. Section 2 sets up the measurement; Section 3 is the
inversion (Figure 1); Section 4 traces it along the flow (Figure 2); Section 5 is the
non-safety control (Figure 3); Section 6 states what the criterion could not decide.

---

## 2. Setup

**Models.** The OLMo-3 7B family (Ai2), whose full training flow is public: the base model,
the Instruct endpoint, and — as controls for the role of supervised fine-tuning — two RL-Zero
variants (Math, Code) that apply reinforcement learning *directly to the base model with no
SFT*. For the trajectory (Section 4) we use the Think lineage at SFT steps 1k/15k/43k, DPO,
and the first and last RLVR steps. Every checkpoint is pinned by commit hash.

**Direction.** A refusal direction at layer 20, fit on the base model as the
difference in mean residual-stream activation between 200 harmful (AdvBench) and 200 benign
(Alpaca) prompts, under a fixed neutral scaffold applied identically to every checkpoint. We
fit it two ways — difference-in-means and logistic regression (`tol=1e-10`) — which sit at
cosine 0.74 yet give the same conclusions throughout, so we report difference-in-means and
note where logistic agrees. The *same base direction* is carried unchanged to every other
model: this is the staleness question.

**Steering.** We follow the input-normalised parameterisation of prior work,
`h_l ← h_l + c · μ_l · v̂`, where μ_l is the mean residual-stream norm at layer 20, so the
coefficient c is comparable across checkpoints. The behavioural readout is the **refusal
logit gap**: the model's log-probability of a refusal-onset token minus that of a
compliance-onset token at the first generated position (token ids fixed and reported).

**The two axes.** For a swept coefficient we record, per prompt, the *achieved displacement*
of the refusal gap (how far steering actually moved it) and whether the prompt *crossed* from
refusing to complying. Two summary statistics follow. **d₅₀** is the achieved displacement at
which half the prompts have crossed. Because achieved displacement **saturates and reverses**
at large |c| (pushing harder eventually buys less displacement), d₅₀ is read on the rising
prefix of the curve only. **Excess displacement** is d₅₀ minus the median prompt's unsteered
distance to the boundary: the displacement a model demands *over and above* its own starting
distance. A perfectly efficient steering axis crosses the median prompt exactly at its
boundary, giving excess ≈ 0; a positive excess means the direction is an inefficient control
axis. Excess, not raw d₅₀, is the boundary-fair quantity — raw d₅₀ is mechanically tied to how
deep a model sits, so equal raw d₅₀ across models is close to impossible by construction and
is *not* the null of "no residual effect."

**Rigour.** Every number carries a 95% bootstrap CI over prompts (1000 resamples; paired for
between-model comparisons). A 20-direction random-direction null band accompanies every
steering curve (achieved-displacement z = 4.0–6.9 for the real direction). Prompt splits are
fingerprinted and every refusal run reproduces the same fingerprint (`99a7ac88`); the base
model reproduces its numbers bit-for-bit across sessions. Five pre-registered pipeline
smoke tests pass (hook liveness, per-layer variation, truncation audit, degenerate-direction
rejection, random-direction control).

---

## 3. The instrument decides the answer (Figure 1)

Rank the same four models by "how steerable is refusal" under three instruments and you get
three different orderings.

Under **fixed-magnitude ablation** — remove the base direction outright, the intervention
behind the "it goes stale" claim — the Instruct model ranks **last**: its refusal survives
almost intact. Under **fixed input-norm dosing** at a single coefficient, Instruct again ranks
**last**. Under **boundary-relative dosing** — excess displacement — Instruct ranks **first**:
it crosses its boundary at an excess of **0.037 [−0.251, 0.237]**, essentially zero, while the
base model demands **0.979 [0.662, 1.273]**. The paired difference is **0.942 [0.564, 1.375]**,
excluding zero.

The behavioural check confirms the reversal is real and not an artifact of the logit readout:
dosed at each model's own crossing point, the Instruct model's refusal rate falls from
**1.00 to 0.18** — the largest swing of any checkpoint — against the base model's 0.80→0.62.
The aligned model is the *easiest* to steer, once you stop mistaking its distance from the
boundary for resistance to being moved.

> **Figure 1 — Three instruments, three rankings of the same four models.**
> Each panel ranks base, Instruct, and two RL-Zero variants by refusal steerability under a
> different instrument. (a) Fixed-magnitude ablation and (b) fixed input-norm dosing both place
> the aligned Instruct model *last* — the basis for "base directions go stale." (c)
> Boundary-relative dosing (excess displacement, this work) places it *first*: it crosses its
> own decision boundary at excess 0.04 [−0.25, 0.24] versus base's 0.98 [0.66, 1.27]. Only (c)
> is fair across models that start at different distances from their boundary. Bars are bootstrap
> means; whiskers are 95% CIs. `fig_threeInstruments.png`.

---

## 4. When does the direction become steerable? Abruptly, at SFT (Figure 2)

If the aligned model is easy to steer per unit boundary-relative displacement but the base
model is not, *when* along training does the change happen? We fit the excess displacement of
the base direction at seven points spanning the flow.

The excess collapses from **0.979 [0.662, 1.273]** at base to **−0.032 [−0.275, 0.162]** by
**SFT step 1000** — the base-vs-SFT-1k difference is **+1.011 [0.642, 1.390]**, excluding zero
— and is then statistically flat through SFT step 43k, DPO, and both the first and last RLVR
steps (every post-SFT CI covers zero). Meanwhile the model's *distance* to its boundary climbs
monotonically (median unsteered gap 0.54 → 3.90 → 5.19 → …): the model refuses ever more
firmly even as the direction becomes an ever-cleaner control axis. These are two different
things, and only the fixed-dose instrument conflates them.

The RL-Zero controls close the loop. RL-Zero-Math and RL-Zero-Code apply reinforcement
learning to the base model *without SFT*, and both sit at base-like excess (paired differences
vs base cover zero: −0.10 [−0.32, 0.06] and +0.02 [−0.26, 0.18]), while every SFT-descended
model sits at ~0. The realignment of probe and control axes is a **supervised-fine-tuning
event**; reinforcement learning, applied either after SFT or directly to the base, does not
produce it.

> **Figure 2 — Excess displacement collapses abruptly at SFT and stays flat.**
> Excess displacement of the base refusal direction (displacement needed past each model's own
> boundary to flip its median prompt) across the OLMo-3 flow. It falls from 0.98 at base to ~0
> by the first SFT step (base-vs-SFT-1k diff +1.01 [0.64, 1.39]) and is statistically flat
> through DPO and both RLVR steps. RL-Zero variants (RL on base, no SFT; plotted off-axis) sit
> at base-like ~1.0. The realignment is an SFT event, not an RL event. Points are bootstrap
> means; bands are 95% CIs; both direction-fitting methods shown. `fig_excessD50_trajectory.png`.

---

## 5. Is it steering in general, or refusal specifically? (Figure 3)

The excess could be a generic fact about steering any direction on these models, or something
specific to the refusal representation. We repeat the whole procedure with a **sentiment**
direction (fit on SST-2 at layer 20, positive vs negative, with a matched positive/negative
continuation-token readout) on the base and Instruct models.

Sentiment shows **no elevated base excess**: base **−0.114 [−0.193, −0.041]**, Instruct
**−0.256 [−0.502, −0.145]**. The ~1-logit base excess that defines the refusal result is
simply absent — a sentiment direction is already an efficient control axis on the base model.
The base→Instruct change for sentiment is same-signed but roughly **six times smaller** than
for refusal and, in candour, its CI narrowly excludes zero, so the honest statement is "far
weaker for sentiment," not "literally zero." The boundary-excess phenomenon, and its abrupt
repair at SFT, is a property of the **refusal** direction.

This gives the mechanism its cleanest form. On the base model the difference-in-means refusal
direction is a fine *probe* — it separates harmful from benign prompts — but a poor *steering*
axis, because pushing along it does not efficiently move the decision. Supervised fine-tuning
aligns the probe axis and the causal axis for refusal specifically, which is exactly the
regime in which "decodable but not controllable" is known to arise, now given a developmental
locus.

> **Figure 3 — The excess is refusal-specific.**
> Excess displacement for a sentiment direction (base −0.11 [−0.19, −0.04], Instruct −0.26
> [−0.50, −0.14]) beside the refusal direction (base 0.98, Instruct 0.04). Refusal shows a
> large base excess that collapses at alignment; sentiment shows none. The phenomenon is a
> property of the refusal representation, not of activation steering in general. Bars bootstrap
> means, whiskers 95% CIs. `fig_sentiment_control.png`.

---

## 6. What our criterion could not decide

Rigour here means naming the limits, not just the results.

- **Excess is a within-direction, within-model quantity.** It cleanly measures inefficiency of
  a fixed direction as a control axis; it does *not* isolate whether SFT rotates the causal
  axis, sharpens it, or reduces per-prompt heterogeneity. We report the phenomenon and its
  locus, not its weight-level cause.
- **The sentiment control excludes zero.** The refusal effect is ~6× larger and the qualitative
  claim (a base excess that SFT removes) holds only for refusal, but we do not claim the
  sentiment excess is exactly zero.
- **One direction-fitting family per concept, two methods.** Difference-in-means and logistic
  agree despite sitting at cosine 0.74; we did not test a supervised causal direction (e.g.
  learned by gradient on the behavioural gap), which could be a cleaner control axis on base
  and would sharpen the probe-vs-control framing.
- **One lineage, one scale, one boundary metric.** 7B OLMo-3; the boundary is defined by a
  single logit-gap readout. Whether the SFT-abruptness replicates at 32B or on a second lineage
  is untested here.
- **Achieved-displacement saturation is real and bounds the design.** For deep-enough models,
  maximum achievable displacement can fall below a shallower model's d₅₀, so displacement-axis
  comparisons are only valid inside the overlap region (here 71–74% of the narrowest range,
  fully covered); we do not extrapolate past it.

---

## 7. Conclusion — so what

The practical consequence is direct. **A safety audit that probes a model by ablating a fixed
direction, or by steering at a fixed magnitude, will systematically rate an aligned model as
less controllable than it is — and the misrating grows with how thoroughly the model was
aligned.** The very models we most want to stress-test for hidden steerability are the ones a
fixed-dose instrument most understates. Boundary-relative dosing is a cheap fix: it requires
only the model's unsteered margin, which the sweep already measures.

The scientific consequence is a reconciliation. The base refusal direction does not go stale;
it was never an efficient control axis on the base model to begin with, and supervised
fine-tuning — abruptly, within its first thousand steps, and not reinforcement learning — turns
it into one. "Base vectors remain effective," "post-training vectors steer better," and "the
base direction loses its grip" are three shadows of one object seen through an instrument that
confuses distance-to-boundary with resistance-to-being-moved. Measured in the right units, the
object is simple: alignment makes the refusal direction *more* controllable, not less, and it
does so at the supervised-fine-tuning step.

---

*Reproducibility. All code, pinned checkpoint commits, split fingerprints, seeds, per-experiment
result JSON, and figure-generating scripts are released. Every headline number carries a
bootstrap CI and a random-direction null; the base model reproduces bit-for-bit across sessions.*
