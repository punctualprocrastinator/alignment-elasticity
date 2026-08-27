# Alignment Elasticity Along the Model Flow — audit

**Repo:** github.com/punctualprocrastinator/alignment-elasticity
**Claim:** Interpretability across the entire OLMo-3 training flow. Seven claims:
(1) safety representations form very early, front-loaded; (2) post-training rotates the
basis once, in early SFT, then freezes; (3) capabilities RL is representationally inert;
(4) information preserved while basis rotates; (5) reading survives, steering decays;
(6) post-training suppresses rather than removes; (7) formation timing does not predict
elasticity.

| | |
|---|---|
| **Current level** | **L2 — workshop tier as-is** |
| **Expected level** | **L3-L4** (ICLR/COLM/ICML) after controls + steering fix |
| Code trust | 6/10 (present code ~8/10; E1/E1b/E5/E7/E8 have no code or data) |
| Results reliability | 5/10 |
| Methodology | 6.5/10 |
| Hypothesis | 5/10 |
| Novelty | Claim 4 genuinely new; most others established |

**Engineering hygiene is top-decile in this cohort** — 9 logged retractions, computed
noise floors, self-killed headlines. The weakness is the inferential chain from
measurement to claim, not the measurement.

---

## What passes

**The frozen-probe check passes.** `e2_base_cache` fits `StandardScaler` +
`LogisticRegression` once on anchor activations; every checkpoint scores through
`e2_scores(_B["sc"], _B["clf"], ...)` — base scaler, base classifier, unchanged. No
silent refit. This was the single most likely way the headline could have been an
artifact, and it isn't.

**Verified numbers:** 0.547→0.804 · −0.1852 [−0.1903, −0.1800] · 0.0009 · 0.0117 ·
0.9993 · 59.86° · 0.670/0.350 · 93.5%/15%.

**The noise floor is sound.** Analytic √(2/(π·4096)) = 0.01247 vs reported 0.0117 —
1.4 SE low given ~320 effective draws (20 directions reused across 13 tags), correct to
2 s.f.

## Blocking findings

### 1. The control that would decide the framing was already run — and buried
E8 ran a non-safety control (**sentiment**) and it behaved **identically, only harder**,
across the flow. That result is quarantined in an "inconclusive" section and does not
appear in the README.

The repo's stated gap is "non-safety controls have not been run." The truth is worse:
the first one was run and points the wrong way. If controls behave identically, the paper
becomes *"fine-tuning rotates all linear structure once, early"* — true, weaker, and not
about alignment.

**Required suite:** sentiment + topic + formality + a *semantically arbitrary but
decodable* concept (missing entirely), all at **matched base-probe AUROC**, reported as
between-concept **difference intervals**. Equivalence requires TOST/ROPE, not two similar
point estimates.

### 2. Claim 5 is unsupported and probably inverted
From `a1_summary.json`, the base-era direction removes **6.81** logits from Instruct vs
**4.37** from base — non-overlapping CIs. **Steering got stronger, not weaker.**

The crossing rate falls only because Instruct starts at +7.85 vs base's +0.85 — an
unmatched dose measured against a shifted threshold. Fix with `c_50` / equal-displacement
steering. Roughly one afternoon of work, and it may flip the claim.

### 3. Claim 7 — the declared headline — collapses on a retracted number
"Formation timing does not predict elasticity" rests on honesty showing a *smaller*
frozen-probe gap than harmfulness: 0.012 vs **0.04**. But 0.04 is retracted. Against the
corrected 0.007, **the sign flips.**

The paper's central falsification does not survive its own correction.

### 4. The 0.812 vs 0.820 headline is a layer-average hiding its own structure
It averages 8 layers including **L4, which is at chance** (frozen 0.513, gap +0.066) and
carries the entire +0.007. Real frozen AUROC is 0.88-0.90, and **at L24/L28/L31 the
frozen probe beats refit** (over-fit refit; n=800 in R^4096).

Also: the layer-mean gap CI [0.00018, 0.01498] **excludes zero**, while "indistinguishable
from zero" belongs to the best-layer quantity — two different numbers presented as one.
And `best_layer = auroc_refit.idxmax()` on the same eval set is selection-on-test.

*Both auditors found this independently.*

### 5. "Exactly 1.00 for every pair" — unverifiable, do not publish as stated
No E5 code, no E5 result file; prose in `pipeline-notes.md` plus an orphan PNG. Rendering
the heatmap shows legitimate Jaccard values (`k/(40−k)`, k=12-20) with two exact-identity
blocks: {sft-43k, dpo, rlvr-25/400/1375} and {base, rlz-100, rlz-1900}.

Evidence *against* a caching bug: the notes' Spearman drifts (0.410/0.410/0.412/0.414)
while the top-20 set does not — a cached tensor could not produce that. But tie-break,
rank-20/21 margin, and the source tensor are all unauditable.

### 6. Figure integrity is systemic
**23 of 29 figures are orphans. All 11 figures in the report are orphans.** Only 6 have
generators; none reads `results/*.json`; `build_report.py` hardcodes every number.

Two figures actively contradict the text:
- `fig_e2_frozen_vs_refit.png` prints "elasticity gap 0.056 AUROC" beside text saying 0.007
- `fig_e4_behaviour_vs_geometry.png` shows the retracted ASR 0.000 with mixed prompt formats

### 7. Other
- **Claim 3 is a null of insufficient treatment** — RLVR ΔASR is flat, so behaviour barely
  moved; representational inertness is then expected, not informative.
- **Claim 6 is undercut by its own data**: `rlz-math` prefill ASR **0.805 < base 0.865**,
  pointing to coherence rather than suppression. Prefill "0.92-0.96" also excludes sft-1k at 0.88.
- Onset: file says `onset_step: 1000`, README says 2,000 (README is right only under a
  selection-matched null, recomputed at 0.6070).
- "1k→2k carries ~50%" is **40%**; 50% is the cumulative share *through* 2,000.
- Missing **split-half direction-reliability ceiling** — the random-direction floor is the
  wrong reference. This gates claims 2 and 4.
- No multiple-comparison correction anywhere. `asr_prefill` has no CI (violates the
  protocol's own P4 on a headline number). n=500 vs protocol's 1,000. Seed 20260805 vs
  protocol's 42.

**Tally: 15 MATCH · 5 MISMATCH/partial · 6 UNVERIFIABLE** (26 line items).

## Pre-registration: no

The repo has **only 2 commits** at full depth, so git provides no chronology. More
decisively, `protocol.md` is explicitly retrospective by its own text — "rationale for
each rule is a specific failure already observed" — and P1b was added the same day as the
results it changes. **Present it honestly as a post-hoc SOP.**

---

## Novelty, per claim

| # | Claim | Verdict | Closest prior work |
|---|---|---|---|
| 1 | Early front-loaded formation | Established | Moskvoretskii (2605.13329) — persona vectors at 0.22% of *OLMo-3-7B*; Qian (ACL-F 2024) |
| 2 | Single early-SFT rotation | Partly | Du (2509.06795) |
| 3 | RL representationally inert | Established | Zhu (2511.08567) — RLVR rotation < SFT |
| 4 | **Info preserved while basis rotates** | **GENUINELY NEW** | — |
| 5 | Reading survives, steering decays | Established | "Perfect Detection, Failed Control" (2606.24952); She (EMNLP-F 2025) |
| 6 | Suppresses not removes | Fully established | Qi et al., ICLR 2025 (shallow safety alignment) |
| 7 | Timing doesn't predict elasticity | New but **unsupported** | — (currently the declared headline) |

*Six of these citations could not be verified — arxiv.org is egress-blocked for
subagents. Treat unverified IDs as leads, not references.*

**Recommended headline: claim 4.** It is the one genuinely new result and it survives
even if the control suite (H4b) comes back unfavourable.

### Title collision — rename required
[Ji et al., *Language Models Resist Alignment: Evidence From Data Compression*, **ACL 2025
Best Paper**](https://aclanthology.org/2025.acl-long.1141/) (PKU-Alignment) coins
"elasticity" for post-alignment models **reverting toward the pretraining distribution
under fine-tuning** — close to the inverse of this repo's usage. *(Verified.)*

Same term, Best Paper status, opposite direction. As titled, reviewers will read this as a
claim about Ji et al. **Rename, and cite in §1.**

**Bibliography: zero** (one arXiv citation in ~1,200 lines). 27 must-cites identified.

---

## Improvements, prioritized

### Priority 1 — required before submission
1. **Run the control suite** (sentiment + topic + formality + arbitrary-decodable), matched
   on base-probe AUROC, reported as difference intervals with TOST. **Un-bury the E8
   sentiment result** and report it in the README whichever way it falls.
2. **Fix claim 5 with equal-displacement steering.** One afternoon; may invert the claim.
3. **Withdraw or re-derive claim 7** — it currently rests on a retracted number and flips sign.
4. **Regenerate every figure from `results/*.json`.** Delete `build_report.py`'s hardcoded
   numbers. 11/11 report figures are currently orphans and two contradict the text.
5. **Rename the paper.**

### Priority 2
6. Report per-layer frozen-vs-refit, not the layer-average; drop L4 or justify it. Select
   `best_layer` on a held-out split.
7. Add the split-half direction-reliability ceiling (gates claims 2 and 4).
8. Publish E5 code + data or remove the "exactly 1.00" claim.
9. Add CIs to `asr_prefill`; apply multiple-comparison correction; reconcile n and seed
   with the protocol.
10. Write a bibliography.

### Priority 3
11. Either scale beyond one lineage or scope the title to OLMo-3 explicitly.
12. Validate the refusal classifier; report whether `rlz-math`'s sub-base ASR is coherence.

---

## Venue ladder

| Tier | Venue | Deadline | Notes |
|---|---|---|---|
| **Primary** | **NeurIPS 2026 workshops** | **~29 Aug 2026** (suggested; confirm per-workshop) | Qualifies **as-is** at workshop tier. 8 days out. Lead with claim 4. |
| **Secondary** | **ICLR 2027** | abstract ~18 Sep, full ~25 Sep 2026 | Only after the control suite and the steering fix. |
| **Tertiary** | **COLM 2027** (~late Mar) or **ICML 2027** (~22 Jan 2027) | | The complete paper, with controls and a second lineage. |

*Deadlines from subagent research; confirm each against the official CFP before relying
on them.*

---

## Product / deployment assessment

**Verdict: no direct product; one real operational implication.**

This is measurement science about training dynamics. There is nothing to ship.

The actionable implication — and it is genuinely valuable to a frontier safety team — is
about **monitor staleness**: if reading survives basis rotation while steering does not,
then probe-based monitors trained on base-era activations remain valid across
post-training, while steering-based *interventions* calibrated on base-era directions
silently decay. That is a concrete operational rule for anyone maintaining activation
monitors across a training pipeline.

**But claim 5 is precisely the claim the audit found may be inverted.** So the
operational advice is currently unsafe to act on. Fix the steering dose-matching first;
if the claim survives, this becomes a short, useful practitioner note — plausibly more
cited than the paper.
