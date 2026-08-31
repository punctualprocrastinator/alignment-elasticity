# Distance to the Boundary Is Not Steering Resistance

**Alignment widens the behavioural margin, not the steering lever — so fixed-dose steering audits misrank aligned models.**

A refusal direction fit on a base language model is widely reported to "go stale" on its
aligned descendants: ablate the direction and the base model complies, but the aligned model
still refuses. This work shows the direction does not weaken — the target moves — and that the
"staleness" is a measurement artifact of applying a *fixed-magnitude* intervention to models
that sit at very different distances from their decision boundary.

Full paper draft: [`paper/paper.md`](paper/paper.md). *(This repo began as an observational
study, "Alignment Elasticity Along the Model Flow"; that framing was superseded — see
[History](#history).)*

---

## The result in one table

Rank the same four OLMo-3 models by "how steerable is refusal," and the answer depends entirely
on the instrument:

| model | fixed ablation (crossing) | fixed-dose (crossing) | **per-dose efficacy** | margin |
|---|---|---|---|---|
| base | 1.00 | 0.92 | 3.84 | 0.81 |
| **Instruct** | **0.33** | **0.01** | **5.61** | 7.75 |
| RL-Zero Math | 1.00 | 0.88 | 4.01 | 0.99 |
| RL-Zero Code | 1.00 | 0.93 | 4.12 | 0.83 |

Fixed-magnitude instruments rank the aligned model **last** (it crosses 1% of prompts where
base crosses 92%) — yet its per-dose lever is the **strongest** of the four. The instrument that
reads "uncontrollable" is measuring the model with the most intact lever.

## Findings

| Finding | Evidence |
|---|---|
| Alignment widens the behavioural margin | Refusal margin grows **9.5×** (0.81→7.75 logits) across the OLMo-3 flow |
| The steering lever does not weaken | Per-dose efficacy CV **0.13** across ten checkpoints; a gradient-trained causal direction (6× more efficient, near-orthogonal) is equally invariant |
| Margin growth is the *universal, sufficient* cause | Fixed-dose audits misrank the aligned model in **all four families** — OLMo-3, Qwen3-8B, Llama-3.1-8B, Gemma-2-9B (margin ×5.9–9.8) — regardless of what the lever does (invariant on OLMo, ~invariant on Gemma, *grows* ~4× on Llama) |
| Honesty is the causal control | A concept whose margin barely grows (**1.2×**) shows essentially no audit artifact — margin growth, not the lever, drives the misranking |
| Holds at scale | OLMo-3 32B: flat lever (CV 0.041), margin ×2.3 |
| **Onset-control is not harm-control** | The base direction still flips an aligned model's refusal *onset*, but HarmBench-judged *genuine harm* decays with alignment (0.80→0.05); prefix refusal metrics' correlation with true harm flips **+0.96→−0.53** |

Two practical warnings, opposite in direction: fixed-dose steering audits **understate** aligned
models' controllability (margin grew), while onset/prefix-based audits **overstate** compliance
(onset-flip ≠ genuine harm). Both errors grow with how thoroughly a model was aligned.

## How the claim was stress-tested

This result is what survived an adversarial self-review; the process is part of the record:

- An earlier "excess-displacement" statistic was **discarded** after it turned out to correlate
  0.93 with the shape of the unsteered-gap distribution rather than measuring a steering
  property. The negative result and the reasoning are kept in [`paper/redteam.md`](paper/redteam.md).
- Every headline number carries a **bootstrap 95% CI** and clears a **random-direction null**;
  splits are fingerprinted (`99a7ac88` refusal, `4382b598` honesty); the base model reproduces
  bit-for-bit across sessions.
- The four reviewer attacks were pre-registered and answered by experiment (E1 causal direction,
  E2 behavioural/non-circular, E3 layers, E4/E5/E6 concept/family/scale) — see
  [`paper/experiment-plan.md`](paper/experiment-plan.md).

## Repository layout

```
paper/paper.md            The paper (draft)
paper/redteam.md          Adversarial self-review + the discarded excess-displacement result
paper/experiment-plan.md  Pre-registered attacks and the two-box experiment program
protocol.md               Canonical measurement protocol (every number produced this way)
pipeline-notes.md         Full result log, including every correction and retraction
scripts/gate_a.py         Steering sweep: efficacy/margin, bootstrap, nulls (reviewed)
scripts/gate_a_analysis.py  Analysis + verdict logic (reviewed)
scripts/e{1,2,3}_*.py     Causal direction, behavioural dose-response, layer robustness
scripts/e5_*.py           Second-family / Llama / Gemma runners
figures/                  All figures (PNG, 200 dpi)
results/                  Per-experiment result JSON (gateA_*, E1..E6, per-model backing)
```

## Reproducibility

Config-driven, pinned HF revision commits, fixed seeds, split fingerprints, per-experiment
result JSON, and figure scripts are released. Off-OLMo families (Llama, Gemma) use unsloth
full-precision mirrors — verified bf16 with correct parameter counts — because the official
repos are license-gated; this is noted in the paper's limitations.

## History

The repo is named `alignment-elasticity` after its origin: an observational study of when
safety representations form during pretraining and whether capabilities RL erodes them. A
literature review ([`litreview.md`](litreview.md)) found the observational headline findings
were largely already established, and adversarial re-analysis then redirected the work to the
steering-audit result above. The original notes and the full correction history remain in
`pipeline-notes.md` — the trail is deliberately kept.

## Licence

Code MIT. Figures and text CC BY 4.0. Models: OLMo-3 (Ai2, Apache-2.0); Qwen3, Llama-3.1,
Gemma-2 via their respective licences.
