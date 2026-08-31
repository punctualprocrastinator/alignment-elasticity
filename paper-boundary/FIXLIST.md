# Author revision notes — claimed fixes (round 1)

Reviewers R1-R3 converged; below, each fix with its location in the revised PDF.

| # | Reviewer concern (R1/R2/R3) | Fix | Where in revised PDF |
|---|---|---|---|
| 1 | Figure 2 caption says 9.5x but plot reaches ~7x (Instruct off-axis) [R1,R2,R3] | Regenerated Fig 2 to end at Instruct; line now reaches 9.5x, lever line shown rising to 1.46x (honest, not fake-flat); caption rewritten | Fig 2 + caption (Sec 4) |
| 2 | "invariant" oversold; 1.56x spread whose max is the aligned model [R1,R2,R3] | Downgraded to "non-weakening and mildly increasing" for refusal; "invariant" reserved for honesty (CV 0.021); abstract "equally invariant"->"similarly near-constant, CV 0.196" | Abstract; Sec 4 lead + gradient para |
| 3 | Appendix A (excess-displacement autopsy) cited but absent [R1,R2,R3] | Added Appendix A: the 0.93 confound table AND the key check that efficacy does NOT inherit it (corr 0.23 vs excess's 0.93) | New Appendix A (p.10) |
| 4 | CV 0.13 vs L20 0.19 inconsistency; span unspecified [R1,R3] | Disambiguated: 0.13 is over the ten-point flow; 0.19 is the 3-checkpoint layer-robustness subset; both stated | Sec 4 lead |
| 5 | mu (dose norm) never reported; efficacy-in-c could be a mu artifact [R2] | Reported mu CV 0.035 across the flow, so efficacy-in-c = efficacy-in-activation-norm | Sec 2 (Lever and load) |
| 6 | "causal control" overstates an n=2 concept contrast [R1,R2,R3] | Retitled to "convergent control"; abstract and Sec 5 reworded; section heading changed | Abstract; Sec 5 heading + body; intro |
| 7 | Sec 3-5 "most controllable" vs Sec 6 "no harm" tension unreconciled [R3] | Added signpost: "controllable" is onset-level; Sec 6 shows onset != harm | Sec 3 |
| 8 | Onset/harm dissociation confounded with dose-driven degeneration [R2, mentor] | Added explicit limitation acknowledging the confound and that a coherence-matched harm measurement is needed | Sec 8 (limitations) |
| 9 | Fig 3 caption claims "levers flat" but figure shows only margins [R2] | Caption fixed to describe the plotted margins; lever CVs marked "not plotted here" | Fig 3 caption |
| 10 | LaTeX \ldots rendering bug in Appendix B [R3] | Fixed the generator escaping; \ldots now renders correctly | Appendix B table |
| 11 | Conclusion "the direction was never the thing that moved" contradicts Table 2 (Llama lever grows) [R3] | Softened to "the direction changed far less than the margin, and for refusal did not weaken at all" | Sec 9 |

NOT fixed (out of scope for a text revision; acknowledged as limitations):
- Running an actual published audit pipeline unchanged (R3 Q5): would require re-implementing [6]'s pipeline; the reconciliation remains argued, and this is flagged.
- New coherence-matched harm experiment (R2 Q3): needs compute; added as a stated limitation (#8) rather than run.
- Printing full per-checkpoint CIs on every table/figure (R1/R2/R3): CIs are in released JSON and now referenced in Fig 2 caption; a full CI-band pass is deferred to camera-ready.
