# Alignment Elasticity Along the Model Flow

**When do safety-relevant representations form during pretraining, and what does capabilities RL do to them?**

An interpretability study across the *entire* OLMo 3 training flow — pretraining, midtraining,
SFT, DPO, RLVR, and RL-applied-directly-to-base — using Ai2's released checkpoints as a
measurement substrate. Work in progress; results here are pre-preprint.

---

## Headline findings

| Finding | Evidence |
|---|---|
| A generalising harmfulness representation appears very early in pretraining | Departs its shuffled-label null around step 2,000 (~0.14% of the run); AUROC 0.547 → 0.804 by end of stage 1 |
| The rise is **front-loaded**, not log-linear | On the dense 1B suite, the interval step 1k→2k carries ~50% of the total rise across 1.9M steps, at 3.2× the SD of every later change |
| Post-training moves the geometry **once**, in early SFT | Cosine-to-base drops 0.185 [0.180, 0.190] between SFT steps 1k and 15k — ~40 bootstrap SDs from zero |
| Capabilities RL is representationally inert | Total drift across 1,350 RLVR steps: 0.0009 — 206× smaller than the cliff and *below the metric's own noise floor* (random directions in R^4096 sit at \|cos\| ≈ 0.0117). RL-on-base leaves cosine at 0.9993 |
| Information is preserved while the basis rotates | A probe frozen at base scores 0.812 AUROC vs 0.820 refit — gap 0.007, indistinguishable from zero at the best layer — while its direction rotates ~60° |
| **Reading survives; steering decays** | The base-era refusal direction still moves 0.670 of base prompts across the refusal/comply boundary but only 0.350 of the deployed model's; in generation it removes 93.5% of base refusals vs 15% of the deployed model's |
| Post-training **suppresses** rather than removes | Under prefill, every post-training checkpoint returns to 0.92–0.96 attack success — *above* the base model's own 0.865 |
| The refusal circuit is re-weighted once, then frozen | Top-20 attributed-component overlap with base falls to 0.43 through SFT, then is **exactly 1.00 for every pair** from SFT-43k through DPO and all of RLVR |
| **Formation timing does not predict elasticity** | Honesty becomes decodable ~2.5× later than harmfulness yet shows the same cliff, the same RL inertness, and a *smaller* frozen-probe gap. The pipeline stage decides, not the concept |

Fuller narrative with figures: run `python scripts/build_report.py` to generate `report.html`.

## What is not established

- **Scale.** Everything is 7B on one lineage. A 32B replication was attempted and lost when the
  sandbox was reclaimed mid-download — no 32B numbers exist.
- **Honesty at protocol strength.** The falsification above rests on day-2 numbers produced
  before the current protocol (see below); the clean re-run has not completed.
- **Whether any of this is alignment-specific.** Non-safety control concepts (sentiment, topic)
  have not been run. If they behave identically, the honest claim becomes "fine-tuning rotates
  all linear structure once, early" — a different paper. See H4b in `project.md`.
- The J-space / verbalizability result is **inconclusive**: it correlates with the read/steer
  split, but an unrelated control direction behaves the same way and the available lenses were
  fit on plain web text.

## Repository layout

```
protocol.md              Canonical measurement protocol (v1) — how every number is produced
project.md               Hypotheses with explicit failure points, smoke tests, experiment program
pipeline-notes.md        Full result log, including every correction and retraction
checkpoints.md/.json     Audit of released OLMo 3 / 3.1 revisions (1,487 pretraining branches)
figures/                 All figures (PNG, 200 dpi)
results/                 Per-experiment result JSON
scripts/pipeline*.py     Extraction, direction fitting, ablation hooks, bootstrap helpers
scripts/notebook_cells/  Mirrored notebook cell sources (sandboxes are ephemeral)
scripts/build_report.py  Builds report.html with figures embedded
```

## Method, briefly

Probes are trained on ordinary prompts and evaluated on **adversarial** ones — harmful requests
disguised as innocent, innocent requests dressed up to look alarming. Without that split,
harmfulness probing saturates at 0.99 by reading topic rather than harm, and every checkpoint
scores identically. A neutral scaffold (`User: … / Assistant:`) is applied identically to every
checkpoint, since base models have no chat template and mixing formats silently measures format
sensitivity. AUROC is the headline metric throughout; accuracy is a calibration diagnostic only
(a randomly initialised network scores 0.81 accuracy on the easy split). Causal claims use
directional ablation with random-direction nulls, and behavioural claims are judged by the
HarmBench classifier rather than by string matching.

Full rules, each traceable to a specific failure, are in [`protocol.md`](protocol.md).

## On the corrections

[`pipeline-notes.md`](pipeline-notes.md) records nine claims that controls or larger samples
overturned, including the project's own original hypothesis. A few worth knowing about, because
they are easy to repeat:

- **Raw effect size measured displacement, not outcome.** At n=6 the base-era direction looked
  like it retained 8% of its effect on the deployed model; at n=200 it inverts to 156%. The
  deployed model is pushed *further* but starts far enough inside refusal that it does not
  cross. Scored as crossing rate, the dissociation returns and separates cleanly.
- **A generation-length bug manufactured a clean result.** 44-token completions truncate inside
  the reasoning block, so the judge scored reasoning traces as harmless and attack success
  appeared to collapse to exactly zero. At 512 tokens it is 0.180.
- **sklearn's default solver tolerance leaves the fitted direction at cosine 0.978 to the
  optimum.** When the headline metric *is* a direction cosine, that is contamination. Fit at
  `tol=1e-10`.
- **A single shared shuffle permutation makes a null band worthless** — it inflated ours to
  0.658, wide enough to swallow the entire formation curve.
- **Prompt format changes probe calibration far more than probe ranking.** An earlier claim that
  the representation was "format-gated" was an accuracy-only artifact.

## Models and data

OLMo 3 7B / OLMo 2 1B checkpoints (Ai2, Apache-2.0). Contrast sets: WildGuardMix (gated),
AdvBench, Alpaca, Azaria–Mitchell true/false statements. Behavioural judging:
`cais/HarmBench-Llama-2-13b-cls`. Circuit analysis by attribution patching over all attention
heads and MLPs, validated by activation patching against random and mid-rank controls.

## Status

Pre-preprint. Numbers may move; the corrections list above is evidence that they do. Everything
needed to reproduce or contradict a claim — protocol, seeds, split fingerprints, pinned HF
revisions — is in the repo.

## Licence

Code MIT. Figures and text CC BY 4.0.
