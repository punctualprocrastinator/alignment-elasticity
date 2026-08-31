# 01 — Outline (Neel step 3–4): every section maps to a claim, or it is cut

| § | Content | Serves | Status in paper.md |
|---|---|---|---|
| Title + subtitle | "Distance to the Boundary…"; margin-not-lever | C1 | done |
| Abstract | numbers-first; C1, C2, C3-scope, so-what | all | done — needs a rigor pass |
| 1 Intro | 3-groups-3-answers hook → lever/load distinction → honesty control → onset≠harm; **end with bullet contribution list (Neel)** | C1,C2 | draft lacks the bullet contribution list — ADD |
| 2 Setup | models (4 families, 32B), directions (3 fits), lever/load defs, HarmBench, rigour (CI, null, fingerprint) | all | done |
| 3 Fig 1: instrument decides | native-metric table; aligned last vs first | C1 | done |
| 4 Fig 2: lever invariant, load grows | efficacy CV 0.13 vs margin ×9.5; E1 causal, E3 layers, E6 32B | C1,C3 | done |
| 5 Fig 3: margin universal + honesty control | 4-family table; honesty ×1.2 → no artifact | C1 (causal),C3 | done |
| 6 Fig 4: onset ≠ harm | behavioural onset invariant; HarmBench decay; prefix flip | C2 | done |
| 7 Limitations | OLMo-specific invariance; no matched-distance; excess-d50 dead end; mechanism unresolved; mirrors | honesty | done |
| 8 Conclusion / so-what | two audit warnings, both cheap to fix | C1,C2 | done |
| App A | excess-d50 negative result (from appendix-redteam.md) | rigor | in appendix-redteam.md |
| App B | tacit knowledge: obstacles, dead ends, what we'd do next | Neel appendix | 02-appendix-tacit.md |

## Neel/Carlini checklist to enforce on the next edit pass
- [ ] **Contribution bullets** at end of intro (Neel: explicit).
- [ ] Abstract, intro, figures each get a dedicated edit pass (Neel time-allocation).
- [ ] Every figure caption states the takeaway and stands alone (both).
- [ ] Rigor: report CIs everywhere; note where p-equivalent is marginal; **no cherry-picking** —
      say which prompts/examples were random vs chosen. Add a "randomly selected generations"
      appendix table for the onset-vs-harm claim (Neel).
- [ ] Related work: short, contrastive — persona-vectors (2605.13329), the LessWrong OLMo posts,
      the monitor-staleness paper (2606.15980), Ji et al. (name collision). From litreview.md.
- [ ] Cut anything not serving C1/C2/C3. One idea.
- [ ] Colorblind-safe figures; no red/green load-bearing (Neel).
- [ ] Reproducibility statement + repo link.

## Open build tasks
1. Composite **Figure 3** (4-family margin bars + honesty control) — not yet built.
2. Verify **Figure 4** contains the prefix-metric correlation-flip panel; if not, rebuild.
3. Related-work section — draft from litreview.md.
4. Randomly-sampled generation examples table (onset-flip-but-safe) for the appendix.
