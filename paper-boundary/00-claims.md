# 00 — Compressed narrative (Neel step 1–2, the thing everything checks against)

Rule: if a sentence in the draft does not serve one of the claims below, cut it.

## The one-line thesis
Fixed-magnitude steering/ablation audits misrank aligned models as uncontrollable, because
alignment widens the behavioural **margin**, not the steering **lever** — and reading a refusal
**onset** as harm misranks them the other way.

## The claims (Neel: 1–3, with confidence level)

**C1 — SYSTEMATIC (headline).** A fixed-magnitude steering/ablation intervention
systematically *understates* the controllability of aligned models, and the error grows with
alignment, because alignment widens the model's unsteered margin (5.9–9.8×) while the direction's
per-dose lever does not weaken.
- *Confidence:* systematic — 4 families (OLMo-3, Qwen3-8B, Llama-3.1-8B, Gemma-2-9B), 2 scales
  (7B, 32B), 2 concepts (refusal, honesty as control).
- *Crucial evidence:* Figure 1 (three instruments, three rankings; aligned model last under fixed
  instruments, first by per-dose lever). Figure 2 (efficacy CV 0.13 vs margin ×9.5, OLMo). Figure 3
  (four-family margin-growth + the honesty control: margin ×1.2 → artifact vanishes).
- *The causal clincher:* honesty. Where the margin barely grows, the misranking barely appears.
  This makes C1 causal (margin drives it), not correlational.

**C2 — SYSTEMATIC (second, independent).** Onset-control is not harm-control. The base direction
keeps flipping an aligned model's refusal *onset*, but HarmBench-judged *genuine harm* decays with
alignment (0.80→0.05); prefix/onset refusal metrics increasingly overstate compliance
(corr with true harm +0.96→−0.53). So onset-based audits misrank aligned models the *opposite*
way from fixed-dose audits.
- *Confidence:* systematic on OLMo; replicated on Qwen; Llama is the extreme case (onset "flip" is
  pure degeneration, no harm anywhere).
- *Crucial evidence:* Figure 4 (behavioural onset efficacy invariant while HarmBench harm decays;
  the prefix-metric correlation flip).

**C3 — SCOPED (the honest lever result).** On OLMo the per-dose lever is strictly invariant across
the entire flow (efficacy CV 0.13; robust to fit method — gradient causal direction equally
invariant; robust across mid–late layers; holds at 32B). Off OLMo it is only approximately
constant (Gemma ~invariant; Llama's lever *grows*). We state invariance as OLMo-specific and rest
C1 on margin growth, which is universal.
- *Confidence:* clean on OLMo; approximate elsewhere — stated as a limit, not hidden.

## What we deliberately do NOT claim (Neel: kill the overclaims yourself)
- NOT "aligned models are just as steerable for harm" (C2 refutes it).
- NOT "the steering lever is universally invariant" (Llama's grows; C3 scopes it).
- NOT "excess displacement measures steering resistance" (discarded; appendix — it's a
  distribution-shape proxy, corr 0.93 with gap spread).
- NOT any mechanism for *why* onset-control outlives harm-control (flagged as future work; the
  J-lens verbalizable-workspace test needs a chat-fit lens + non-safety control).

## The "so what" (both advisors demand this be explicit)
Two opposite, both-actionable audit failures, both cheap to fix:
1. Fixed-dose / ablation audits **understate** aligned-model controllability → dose relative to the
   model's own margin (the sweep already measures it).
2. Onset/prefix audits **overstate** compliance → measure genuine harm (an outcome judge), never
   read a refusal-onset flip as compliance.
The models we most want to stress-test are the ones both instruments most mislead.

## Figures (Neel: equal time as abstract; each must stand alone)
- **F1** three instruments / three rankings — `fig_threeInstruments.png` (exists)
- **F2** efficacy flat vs margin ×9.5 — `fig_efficacy_margin.png` (exists)
- **F3** four-family margin-growth + honesty control — needs composite build from
  `fig_E4/E5/E5_llama/E5_gemma/E6`
- **F4** onset-vs-harm dissociation + prefix-metric flip — `fig_E2_behavioural_doseresponse.png`
  (exists; check it carries the correlation-flip panel)

## Target / format
InterpScience @ NeurIPS 2026 (Interpretability as a Science). Non-archival, double-blind,
short ≤5pp or long ≤9pp. Deadline 1 Sep AoE.
