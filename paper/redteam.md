# Red-team of the Gate A paper (2026-08-29)

All checks below are **zero-GPU re-analyses of the existing per-prompt sweep JSONs**
(`gaps_massmean` is [n_c x 200] per checkpoint).

## The decisive problem: excess-d50 is largely a gap-distribution-shape statistic

`excess = d50 - median(unsteered gap)`. Across all 6 refusal checkpoints:

    corr(excess, std of unsteered gap)  = +0.927
    corr(excess, skew of unsteered gap) = +0.830

| model | excess | std_g0 | skew_g0 | frac(g0<0) |
|---|---|---|---|---|
| base | 0.979 | 1.79 | +0.48 | 0.355 |
| sft-1000 | -0.032 | 1.56 | -0.39 | 0.020 |
| sft-15000 | 0.013 | 1.33 | -0.22 | 0.000 |
| instruct | 0.037 | 1.38 | -1.27 | 0.000 |
| rlz-math | 1.082 | 1.85 | +0.41 | 0.340 |
| rlz-code | 0.957 | 1.88 | +0.36 | 0.345 |

The high-excess models (base, both RL-Zero) all have a **wide, right-skewed gap
distribution that straddles zero** — ~35% of prompts already lean toward complying.
The excess-0 models are all deep and concentrated in refusal. Excess is essentially
a proxy for "is the unsteered refusal-gap distribution wide and near zero."

### What this kills
- **Section 5 mechanism ("SFT aligns probe and control axes"): unsupported.** The
  per-prompt test of proportional control, `corr(displacement, g0)`, does NOT track
  excess: sft-1000 has *worse* proportional control than base (-0.243 vs -0.149) yet
  collapsed excess. Cut this claim.
- **"The aligned model is the EASIEST to steer (lowest excess)": overclaim.** Low
  excess = concentrated gap distribution, not intrinsic ease of steering.
- **Heterogeneity was ruled out (good):** per-prompt displacement is uniform (CV ~0.15,
  frac_move 1.0) in every model, so excess is not a displacement-heterogeneity artifact.
  It is a *starting-distribution* artifact instead.

### "Refusal-specific" reduces to a distribution statement
base REFUSAL g0: std 1.79, skew +0.48, frac<0 0.35. base SENTIMENT g0: std **0.55**,
skew +0.15, frac<0 **0.00**. So "refusal shows base excess, sentiment does not" is
really "base is genuinely ambivalent about ~35% of harmful prompts pre-alignment,
while its sentiment classification is confident." True and mildly interesting, but a
claim about base refusal *behaviour*, not steering geometry.

## What SURVIVES (the defensible nucleus)
1. **Figure 1 (a,b) is solid:** fixed-magnitude ablation and fixed input-norm dosing
   rank the aligned model LAST.
2. **The cause is real:** the aligned model sits far from its boundary (unsteered gap
   7.87 vs 0.85).
3. **The aligned model is not intrinsically resistant:** a boundary-matched dose crosses
   it (behaviourally 1.00 -> 0.18).
4. **The practical claim holds and needs no excess statistic:** fixed-dose steering/
   ablation audits understate the controllability of aligned models, and the error grows
   with alignment. This is the paper.

## Required reframing
- Lead with the audit-misranking result (Figure 1). It is clean, useful, distribution-free.
- Demote excess-d50 to a *diagnostic*, and DISCLOSE its ~0.93 correlation with gap spread.
- Drop the Section 5 mechanism and the "easiest to steer" language.
- Recast Figure 2: excess collapses at SFT because SFT moves the whole gap distribution
  deep into confident refusal — which is the known alignment behavioural transition, not a
  new geometric event. Either frame it honestly as that, or cut it.

## The one experiment that makes the core claim bulletproof (ZERO GPU)
**Gap-matched steerability.** For each model, take only prompts in a fixed unsteered-gap
band (e.g. g0 in [0.5, 1.5]) and measure the coefficient / displacement needed to cross
*those* prompts. This controls distribution shape directly. If the aligned model crosses
its matched-distance prompts as easily as base does, the "not resistant, just distant"
claim becomes distribution-free and unattackable. All data is already on disk.

## Experiments worth ~a day (GPU), in priority order
1. **Supervised causal direction** (~30 min): fit the L20 direction by gradient on the
   behavioural gap, not diff-in-means. Tests whether base "inefficiency" is just
   diff-in-means being a lazy control axis. Already flagged in the draft's Section 6.
2. **Per-prompt crossing-coefficient distribution, gap-matched** (zero GPU, do first).
3. **A second safety concept (honesty direction)** through the same pipeline: is it
   "refusal-specific" or "safety-concepts-general"? ~30 min.
4. Finer early-SFT checkpoints (if any < step 1000 exist) to earn the word "abrupt".
5. Layer robustness of the misranking (extraction at 3 more layers).
