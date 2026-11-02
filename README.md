# alignment-elasticity

Steering experiments on a refusal direction carried across the OLMo-3 7B post-training checkpoints,
with comparisons on Qwen3-8B, Llama-3.1-8B and Gemma-2-9B.

## Status

**The thesis this README used to state has been withdrawn.** It said that alignment widens the
refusal margin rather than weakening the steering lever, so fixed-dose steering audits misrank
aligned models. A later test showed the extra dose an aligned model needs is not free: it destroys
the fluency of the output. Under a realistic budget, the aligned models really are less
controllable (details below).

Current work from this project is under anonymous review. Its code and results will be added here
when the review period ends.

## What happened to the earlier claims

| Earlier claim | Status | Reason |
|---|---|---|
| Alignment widens the refusal margin across the OLMo-3 flow and in three other families | Stands | `results/gateA_*`, `results/E5_*.json` |
| The mean per-dose steering lever stays roughly constant across checkpoints | Stands, as an average | `results/gateA_*` |
| Fixed-dose audits misrank aligned models | **Withdrawn** | Keeping outputs as fluent as each model's own unsteered text, steering lowers base's refusal rate by 0.19 and every aligned checkpoint's by 0.00 (`scripts/coherence_budget.py`, `results/coherence_budget.json`) |
| Steering still controls an aligned model's refusal onset but no longer its harmful output | **Withdrawn** | At the dose needed to flip the onset, aligned models' completions degenerate, so the fall in judged harm reflects incoherent text. Prefix-based refusal metrics can still misread such text as compliance |
| Honesty as a control concept; the result holding at 32B | Not reassessed | The thesis they supported was withdrawn |

The full record of corrections and retractions is in [`pipeline-notes.md`](pipeline-notes.md).

## Repository layout

```
protocol.md               Measurement protocol for the steering numbers
pipeline-notes.md         Result log, including every correction and retraction
litreview.md              Literature review that redirected the original project
scripts/gate_a.py         Steering sweep: efficacy, margin, bootstrap, random-direction null
scripts/gate_a_analysis.py  Analysis of the sweep
scripts/e1_causal.py ... e5_family.py   Causal direction, behavioral dose-response, layers, honesty, other families
scripts/coherence_budget.py  The test that withdrew the misranking claim
figures/                  Figures (PNG)
results/                  Result JSON per experiment
```

## Reproducibility

Models are pinned to Hugging Face revision commits, seeds are fixed, and prompt splits are
fingerprinted. Llama and Gemma were run from unsloth full-precision mirrors, verified to be bf16
with the correct parameter counts, because the official repositories are license-gated.

## History

The repository is named after its origin: an observational study of when safety representations
form during training and whether later training erodes them. A literature review
([`litreview.md`](litreview.md)) found those findings were largely established already, and the work
moved to steering audits. The withdrawn thesis above came from that phase. The full trail is kept
on purpose.

## Licence

Code: MIT (see [`LICENSE`](LICENSE)). Figures and text: CC BY 4.0. Models are used under their own
licences: OLMo-3 (Ai2, Apache-2.0), Qwen3, Llama-3.1 and Gemma-2.
