# Distance to the Boundary Is Not Steering Resistance

### Alignment widens the behavioural margin, not the steering lever: so fixed-dose steering audits misrank aligned models

*Workshop submission — InterpScience @ NeurIPS 2026. Draft, 2026-08-30.*

---

## Abstract

A refusal direction fit on a base language model is widely reported to "go stale" on the
model's aligned descendants: ablate the direction and the base model complies, but the aligned
model still refuses. We show the direction does not weaken. The target moves. Across the full
OLMo-3 training flow the model's behavioural margin, its unsteered distance from the refusal
decision boundary, grows 9.5×, from 0.81 to 7.75 logits, while the direction's per-dose steering
efficacy stays essentially constant (coefficient of variation 0.13; a gradient-trained causal
direction, six times more efficient and near-orthogonal, is equally invariant). A fixed-magnitude
intervention therefore fails on aligned models for one banal reason: the same push is now small
against a far wider margin. This inverts a natural ranking. Under fixed ablation and fixed-dose
steering the aligned model looks least controllable, crossing 1% of prompts where the base model
crosses 92%, yet its per-dose lever is the strongest in the set. Honesty supplies a causal control:
its margin barely grows (1.2×), and its audit artifact nearly vanishes, so margin growth drives the
misranking rather than any property of the lever. The mechanism replicates across four model
families (OLMo-3, Qwen3-8B, Llama-3.1-8B, Gemma-2-9B), where the margin grows between 3.7× and 9.8×.
Margin growth alone is enough to cause the misranking, whatever the direction's own steering power
does: it is invariant on OLMo, roughly so on Gemma, and grows on Llama. The effect holds at larger
scale (OLMo-3 32B). A second, independent warning follows: onset-control is not harm-control. The
base direction still flips an aligned model's refusal onset to compliance, but HarmBench-judged
genuine harm decays with alignment (0.80 to 0.05 on OLMo, replicated on Qwen), and prefix-based
refusal metrics increasingly overstate compliance, their correlation with true harm flipping from
+0.96 to −0.53. Safety audits that ablate a fixed direction understate aligned-model
controllability; audits that read refusal-onset as harm overstate it.

---

## 1. Introduction

Activation steering finds a residual-stream direction that correlates with a behaviour and adds or
subtracts it, and it is a standard tool for interpreting and controlling language models. A
consequential question follows for anyone auditing a deployed model: does a direction found on an
earlier checkpoint still control the behaviour after the model is aligned? If white-box safety
interventions built on base-model analysis silently expire during alignment, they expire when they
matter.

The published answers disagree. One line of work reports that base-derived steering vectors remain
effective on aligned instruct models. Another reports that post-training directions steer better
than pretraining ones. A third, including our own earlier ablation experiments, reports that the
base refusal direction loses its grip: ablate it and the aligned model keeps refusing. Three
groups, three answers, one model family.

We resolve the disagreement with one measurement and one distinction. Every prior study fixes the
*magnitude* of the intervention and reads the *outcome*. But a fixed intervention is not a fair
probe of two models whose behaviour sits at different distances from the decision boundary. An
aligned model refuses hard; its margin is large; a fixed nudge moves it far and still leaves it
refusing. We separate the **lever** (displacement produced per unit dose) from the **load** (the
margin the lever must overcome), measure both across the OLMo-3 flow, and find the lever essentially
constant while the load grows an order of magnitude. Honesty, a concept whose load does not grow,
supplies the causal control. A validity check with an independent harm judge then reveals a second
effect the onset-level view hides: the direction keeps flipping the aligned model's onset long after
it stops eliciting genuine harm.

**Contributions.**
- We show that fixed-magnitude steering and ablation audits **systematically misrank aligned
  models as less controllable**, and identify the cause: alignment widens the behavioural margin
  (×3.7–×9.8 across four model families) while the direction's per-dose lever does not weaken (§3–4).
- We establish this causally with a **control concept**: honesty, whose margin barely grows (×1.2),
  shows essentially no audit artifact; margin growth, not the lever, drives the misranking (§5).
- We report a **second, opposite audit failure**: onset-control is not harm-control. The base
  direction still flips an aligned model's refusal onset, but genuine (HarmBench-judged) harm decays
  with alignment, and prefix-based refusal metrics increasingly overstate compliance (§6).
- We give the **honest scope**: strict per-dose lever-invariance is clean on OLMo, approximate off
  it; the universal, sufficient cause of the misranking is margin growth (§5, §7). All code, pinned
  commits, split fingerprints, and a discarded-statistic autopsy are released.

The paper is organized around these: §3 the instrument artifact (Figure 1), §4 the lever/load
decomposition (Figure 2), §5 the honesty control and four-family generalization (Figure 3), §6 the
onset-vs-harm dissociation (Figure 4), §7 related work, §8 the limits, §9 the takeaways.

---

## 2. Setup

**Models.** The OLMo-3 7B family (Ai2), whose entire training flow is public: base; Think lineage
at SFT 1k/15k/43k, DPO, first and last RLVR steps; Instruct; and two RL-Zero variants (RL applied
directly to the base, no SFT). For generalization: **OLMo-3 32B** (base/SFT/DPO/RLVR-last) and three further base→instruct
families: Qwen3-8B, Llama-3.1-8B, Gemma-2-9B (the latter two via unsloth full-precision
mirrors, verified bf16 with correct parameter counts, as the official repos are license-gated).
Every checkpoint pinned by commit hash.

**Directions.** A refusal direction at layer 20 (proportional layer for 32B/Qwen), fit **once on
the base model** as the difference in mean residual-stream activation between 200 harmful (AdvBench)
and 200 benign (Alpaca) prompts under a neutral scaffold applied identically to every checkpoint;
carried unchanged to every descendant (this is the staleness question). Fit three ways (difference-in-means, logistic (`tol=1e-10`), and a gradient-trained steering vector),
which disagree geometrically (cosine 0.16–0.74) yet agree on every conclusion. A honesty direction is
fit the same way on Azaria–Mitchell true/false statements.

**Lever and load.** Steering uses `h_l ← h_l + c·μ_l·v̂` (input-normalised, comparable across
checkpoints). The readout is the concept's logit gap (refusal- vs compliance-onset tokens; ids
fixed). **Load (margin)** = mean unsteered gap. **Lever (efficacy)** = slope of mean per-prompt
displacement vs |c| in the near-zero regime (displacement saturates at large |c|; we use the linear
region). We also measure behavioural efficacy (refusal-rate change vs dose on real generations)
and validate genuine harm with the HarmBench classifier.

**Rigour.** Every headline number carries a 95% bootstrap CI (1000 resamples; paired for
comparisons) and a 20-direction random-direction null. Splits are fingerprinted; every refusal run
reproduces `99a7ac88`, honesty `4382b598`; the base model reproduces bit-for-bit across sessions.
Five pipeline smoke tests pass. We report where a claim is clean (OLMo) and where only approximate
(Qwen).

---

## 3. The instrument decides the answer (Figure 1)

Rank the same four models (base, Instruct, two RL-Zero variants) by "how steerable is refusal,"
and the ranking depends entirely on the instrument. Reading each instrument's *native* metric off
the same sweep:

| model | fixed ablation (crossing) | fixed-dose (crossing) | **efficacy (lever)** | margin |
|---|---|---|---|---|
| base | 1.00 | 0.92 | 3.84 | 0.81 |
| **Instruct** | **0.33** | **0.01** | **5.61** | 7.75 |
| RL-Zero Math | 1.00 | 0.88 | 4.01 | 0.99 |
| RL-Zero Code | 1.00 | 0.93 | 4.12 | 0.83 |

Under fixed-magnitude ablation and fixed-dose steering, the two instruments the prior literature
used, the aligned Instruct model ranks last, crossing 1% of prompts where base crosses 92%.
Yet its per-dose lever is the strongest of the four. The instrument that reads "uncontrollable"
is measuring the model with the most intact lever.

> **Figure 1 — Three instruments, three rankings of the same four models.** (a) Fixed-magnitude
> ablation and (b) fixed-dose steering place the aligned model *last*, the basis for "base
> directions go stale." (c) Per-dose efficacy places it *first*. The disagreement is instrumental:
> the first two confuse distance-to-boundary (load) with resistance-to-being-moved (lever). Bars
> bootstrap means, whiskers 95% CIs. `fig1_three_instruments.png`.

---

## 4. The lever is invariant; the load grows (Figure 2)

Measuring lever and load at ten points across the flow: the displacement produced per unit dose
ranges only 3.6–5.6 (CV 0.13) across base, six Think post-training stages, Instruct, and both
RL-Zero variants, clearing the random-direction null everywhere (z 4–7). Over the same span the
margin climbs 9.5×, from 0.81 to 7.75 logits. A fixed intervention delivers a fixed
displacement (efficacy × dose); as the load grows tenfold while that displacement stays fixed, it
crosses fewer prompts. The direction is not going stale. The target is receding.

The invariance is not an artifact of how we fit the direction. A gradient-trained causal
direction, six times more efficient (mean 26.1 vs 4.5) and near-orthogonal to diff-in-means (cosine
0.21), is *equally* invariant (efficacy CV 0.196). The invariance holds across mid-to-late
layers (per-layer CV: L20 0.19, L24 0.11, L28 0.07; early layers reorganise with alignment), and
at 32B scale (CV 0.041, margin ×2.3). It is a property of the direction as a control axis, not
of the estimator, the layer, or the model size.

> **Figure 2 — The lever is invariant; only the load grows.** Steering efficacy (displacement per
> unit dose; right axis) is flat across pretraining → SFT → DPO → RLVR (3.6–5.6, CV 0.13) while the
> behavioural margin it must overcome (left axis) grows 9.5×. A fixed-magnitude ablation delivers a
> fixed displacement, so it fails on aligned models purely because the margin widened.
> `fig2_lever_vs_load.png`.

**Reconciliation.** "Base vectors remain effective": the lever is invariant. "They go stale under
fixed ablation": the load grew, so a fixed removal no longer reaches the boundary. "Post-training
directions steer better": refit at a later checkpoint, a direction is simply calibrated to that
checkpoint's larger load. Three reports, one invariant lever, seen through instruments that read
the load as the lever.

---

## 5. The universal cause is margin growth; honesty is the causal control (Figure 3)

If margin growth causes the misranking, a concept whose margin does *not* grow should show no
misranking. Honesty is that control. Its per-dose lever is even flatter than refusal's (efficacy
CV 0.021), but across the same checkpoints its margin grows only 1.2× (2.66→3.21): the model
does not become dramatically more confidently honest. Correspondingly, the fixed-dose audit artifact
that is dramatic for refusal nearly vanishes for honesty. This is the causal evidence, not just a
correlation, that margin growth, not any change in the lever, drives the misranking; the
artifact's magnitude tracks how much behavioural confidence a given concept accrues during alignment.

The margin-growth mechanism generalizes across four model families, and it, not lever-invariance,
is the universal cause. For each of OLMo-3, Qwen3-8B, Llama-3.1-8B, and Gemma-2-9B we fit the
refusal direction on the base model and carry it unchanged to the aligned model:

| family | margin growth | onset flip | onset↔harm dissociation | strict lever-invariance |
|---|---|---|---|---|
| OLMo-3-7B | 9.5× | yes | yes | **yes (CV 0.13)** |
| Qwen3-8B | 3.7× | yes | yes | approximate |
| Llama-3.1-8B | 9.8× | yes* | (degeneration) | no (lever *grows* ~4×) |
| Gemma-2-9B | 5.9× | yes (chat) | yes | approximate (5.2→6.2) |

The margin grows in every family (×3.7–×9.8, format-matched; native chat arms grow more), and in every family the fixed-dose audit misranks
the aligned model. What the direction's *own* steering power does varies (invariant on OLMo, roughly
so on Gemma, growing on Llama), yet the misranking appears regardless. **Margin growth is
therefore the universal, sufficient cause; lever-invariance is not required for the audit to fail.**
OLMo is the family where the lever is provably invariant, making it the cleanest demonstration that
the direction need not change at all for a fixed-dose audit to declare an aligned model
uncontrollable. The result also replicates at scale (OLMo-3 32B: flat lever, CV 0.041, margin
×2.3).

The clean scope statement: *the load grows and fixed-dose audits misrank* is universal across four
families and two scales; *the lever is strictly invariant* is specific to OLMo (approximate on Gemma,
family-dependent elsewhere, and measured against a wide null off OLMo).

> **Figure 3 — Margin growth is the universal driver.** Left: refusal margin grows 9.5× while
> honesty's grows only 1.2×, yet both levers are flat (efficacy CV 0.13 vs 0.02), and the audit
> artifact appears for refusal but not honesty; margin growth, not the lever, drives it. Right:
> margin growth and the misranking replicate across Qwen3-8B (×3.7), Llama-3.1-8B (×9.8),
> Gemma-2-9B (×5.9), and OLMo-3 32B (×2.3); format-matched neutral scaffold. `fig3_margin_universal_honesty.png`.

---

## 6. Onset-control is not harm-control (Figure 4)

The lever's invariance is measured on the refusal *onset*, the first-token disposition. Routing it
through full generations confirms the onset lever is non-circular: behavioural refusal-onset
efficacy is invariant across checkpoints (CV 0.149) while the margin grows, and the aligned model's
refusal rate falls from 1.00 to 0.18 when dosed to its own boundary. But a validity check with the
HarmBench classifier reveals a dissociation the onset view hides. As alignment proceeds, flipping
the onset stops producing genuine harm: HarmBench-judged harmful output at each model's crossing
dose decays 0.80 → 0.60 → 0.45 → 0.40 → 0.05 from base to Instruct, and heavy dosing merely
degenerates the aligned model. The same dissociation replicates on Qwen (genuine harm 0.03→0.75 on
base, 0.00→0.20 on the aligned model).

This carries a methods warning beyond our setting: the correlation between a prefix/onset refusal
classifier and true HarmBench harm flips from +0.96 to −0.53 across alignment. Prefix-based
refusal metrics, ubiquitous in steering and jailbreak evaluations, increasingly *overstate*
compliance on aligned models. Onset-control and harm-control must be measured separately.

> **Figure 4 — Onset-control survives; harm-control decays.** The base direction still flips the
> aligned model's refusal onset (behavioural onset efficacy invariant, CV 0.149), but HarmBench-
> judged genuine harm at the crossing dose decays 0.80→0.05 with alignment; a prefix refusal
> classifier's agreement with true harm flips +0.96→−0.53. `fig4_onset_vs_harm.png`.

---

## 7. Related work

**Steering directions across training.** Several recent studies carry a base-model direction across
the training flow. Persona-vector work (arXiv 2605.13329) tracks trait directions through OLMo-3
pretraining and post-training and reports that base-derived vectors remain effective on aligned
models; a checkpoint study of OLMo-3 refusal directions reports that post-training directions steer
better than pretraining ones; our own earlier ablations found the base refusal direction failing on
the aligned model. The three results appear to conflict. We reconcile them: each fixes the
intervention magnitude and reads an outcome, and that outcome depends on how far the model sits from
its decision boundary. The persona-vector protocol normalises dose by the local residual-stream
norm, an input-side quantity that does not equalise the boundary distance, which grows roughly
tenfold across the flow (Section 4). The boundary-relative view is what makes the three findings one.

**Probing versus steering.** That a direction can be decodable without being an effective steering
axis is established (e.g. arXiv 2608.12334). We do not re-establish the dissociation; we explain
part of it developmentally. The per-dose steering lever is invariant across the OLMo flow
(Section 4), so the apparent loss of control on an aligned model is not the direction weakening but
the margin widening.

**Monitor and probe staleness under updates.** Frozen activation probes are known to drift out of
alignment under fine-tuning while the underlying signal survives (arXiv 2606.15980). That work
concerns readout: does a frozen probe still classify? Ours concerns control: does a fixed
intervention still move behaviour? The two are complementary. Readout directions rotate, and we show
that the control lever's per-dose strength does not decay, so the failure of fixed-magnitude control
is attributable to the margin rather than the lever.

**Refusal metrics and a naming note.** Prefix-based refusal detection is standard in steering and
jailbreak evaluation. Section 6 shows it inverts on aligned models, overstating compliance where
genuine harm has not risen; classifiers that judge genuine harm (e.g. HarmBench, Mazeika et al.,
2024) are the right instrument. Finally, "elasticity" has been used for a different phenomenon, the
tendency of aligned models to revert toward pretraining behaviour under further fine-tuning (Ji et
al., ACL 2025); our subject is steerability under a fixed intervention, unrelated to that usage.

---

## 8. What the measurement cannot decide

- **Strict efficacy-invariance is OLMo-specific.** On Qwen the lever grows ~2.6× and its null is
  wide; we claim invariance for OLMo and approximate constancy elsewhere. The *load-grows /
  audits-misrank* claims are the general ones.
- **We cannot compare base and aligned models at matched starting distance.** Aligned models have
  essentially no near-boundary prompts, so there is no overlapping regime; efficacy sidesteps this
  by measuring the per-dose slope, and we do not claim distance-matched equality.
- **A negative result we discarded.** An "excess displacement" statistic (dose past each model's
  boundary) correlates 0.93 with the spread of the unsteered-gap distribution, so it restates that
  distribution rather than a steering property. We report efficacy instead; the excess analysis and
  its confound are in the appendix as a caution.
- **Onset-vs-harm mechanism is unresolved.** We show the dissociation and a prefix-metric failure;
  we do not yet localize *why* onset-control outlives harm-control (the verbalizable-workspace
  hypothesis is future work with the Jacobian lens).
- **Off-OLMo caveats.** Llama-3.1-8B and Gemma-2-9B use third-party unsloth mirrors (official
  repos gated); each was verified full-precision bf16 with the expected parameter count before use.
  Off OLMo the random-direction efficacy null is wide (z 1–3), so per-dose efficacy is a noisier
  axis; on Llama the base direction barely clears the null and its lever *grows* with alignment.
  These do not affect the universal margin-growth / misranking claim, which holds in all four
  families.
- **One layer family, one boundary metric per concept, two safety concepts, four model families.**

---

## 9. Conclusion: so what

Two practical warnings, opposite in direction, both cheap to act on. **A safety audit that ablates
a fixed direction, or steers at a fixed magnitude, understates an aligned model's controllability**,
and the understatement grows with how thoroughly the model was aligned, because alignment widens
the margin without touching the lever. The fix costs nothing: dose relative to the model's own
unsteered margin, which the same sweep already measures. Conversely, **an audit that reads
refusal-onset as harm overstates compliance on exactly those aligned models** — onset-control
outlives harm-control, and prefix metrics invert. The models we most want to stress-test are the
ones both instruments most mislead.

The scientific core is a small, well-controlled invariant. The base refusal direction does not go
stale, does not need refitting, and does not steer "better" after alignment. Per unit dose it
steers *the same*, from the start of pretraining through RLVR, at 7B and 32B, for refusal and (even
more flatly) for honesty. What changes is how far the model has walked from its own decision
boundary, demonstrated causally by a concept, honesty, whose boundary barely moves and whose audit
artifact barely appears. Measured in the right units, "does the direction survive alignment?" has a
clean answer: the direction was never the thing that moved.

---

*Reproducibility. All code, pinned commits, split fingerprints, seeds, per-experiment result JSON,
figure scripts, and the red-team appendix (including the discarded excess-displacement result) are
released. Every headline number carries a bootstrap CI and a random-direction null; the base model
reproduces bit-for-bit.*
