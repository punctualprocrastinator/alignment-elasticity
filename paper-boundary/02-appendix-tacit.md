# 02 — Tacit-knowledge appendix (Neel: preserve what standard papers discard)

Draft material for an appendix + notes-to-self. Not all of this ships; the honest obstacles and
dead ends do, because they save the next person weeks.

## Dead ends we walked (so you don't)
- **Excess-displacement statistic.** Seductive and wrong: correlates 0.93 with the spread of the
  unsteered-gap distribution, so it restates distribution shape, not a steering property. It even
  produced a clean-looking "abrupt at SFT" story before the control killed it. Use per-dose
  efficacy instead. Full autopsy in `appendix-redteam.md`.
- **Gap-matched steerability.** Can't be done: aligned models have no near-boundary prompts, so
  there is no overlap to match base against. The absence is itself the finding.
- **"Onset flip = steerable."** The trap the whole paper is about. On Llama the onset flips to 0%
  refusal but the output is degenerate repetition — zero genuine harm. Always pair an onset metric
  with an outcome judge.

## Measurement gotchas that cost real time
- **sklearn LogisticRegression default `tol=1e-4`** stops LBFGS at ~14 iterations, leaving the
  fitted direction at cosine 0.978 to the optimum. When your metric IS a direction cosine, that is
  contamination. Use `tol=1e-10`.
- **Achieved displacement saturates and reverses** past |c|≈1.5. d50 must be read on the rising
  prefix only, ordered by dose magnitude from zero out (the stored grid is most-negative-first).
- **Prefix refusal classifiers overstate compliance on aligned models** (corr with HarmBench flips
  +0.96→−0.53). Judge genuine harm.
- **HarmBench-Llama-2-13b-cls** ships a sentencepiece `tokenizer.model` that transformers 5.x can't
  convert; load `hf-internal-testing/llama-tokenizer` instead (weights unaffected).
- **Fixed-dose null band off OLMo is wide** (z 1–3): the random-direction efficacy null on Qwen/
  Llama/Gemma is broad, so per-dose efficacy is a noisier axis there than on OLMo. Lead with margin.
- **Unsloth mirrors:** use full-precision repos only; a `-bnb-4bit` mirror would silently corrupt
  efficacy/margin. Assert bf16 + param count + no quantization_config before fitting.

## Reproducibility / infra reality (belongs in a repo README, summarized in the paper)
- Ephemeral GPU sandboxes were reclaimed mid-run repeatedly; the discipline that saved the work:
  pull every artifact the instant it exists, mirror notebook cells to disk, pin every commit,
  fingerprint the split, verify base reproduces bit-for-bit.
- Split fingerprints: refusal `99a7ac88`, honesty `4382b598`. Base reproduces exactly across
  sessions — the free determinism check.

## What we'd do next (future work, appendix)
- **Mechanism for onset≠harm** with the Jacobian lens: test whether the refusal direction keeps its
  grip on the verbalizable/onset workspace while losing the sustained trajectory. Needs a chat-fit
  lens (pre-fitted ones are wikitext) + a non-safety control direction.
- **Boundary-aware audit recipe:** a drop-in that doses relative to the measured margin; quantify
  how much latent steerability fixed-dose audits miss on a battery of released models.
- **Why the lever grows on Llama** but is invariant on OLMo — a family-difference worth its own study.
