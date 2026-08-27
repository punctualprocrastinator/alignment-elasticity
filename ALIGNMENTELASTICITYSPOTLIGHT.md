# alignment-elasticity → spotlight

> ## POST-GATE-A REVISION (27 Aug) — supersedes the thesis below
>
> Gate A resolved: **verdict B with inverted sign.** Instruct's excess-d50 is 0.037
> [-0.251, 0.237] vs ~1.0 for base/rlz-math/rlz-code; behavioural swing 1.00 -> 0.18,
> the largest of any checkpoint. The deployed model is the EASIEST to steer per unit
> boundary-relative displacement. So the paper is no longer the three-curves /
> verbalizability thesis — it is the INSTRUMENT claim, per Carlini's "one singular
> idea":
>
> **Fixed-dose steering interventions read distance-to-boundary as steering
> resistance. Dose in boundary-relative units and published "steering decay"
> reverses.**
>
> Everything in the paper connects to that or gets cut. Formation curves: cut
> (ceded anyway). Monitors, J-space: cut (J-space is a follow-up). The three-curve
> material survives only as motivation for why instruments disagree.
>
> ### Maximal-version gaps to close before writing (each ~cheap, all on one box)
> 1. **Trajectory, not endpoints** — run the Gate A sweep on ~6 more checkpoints
>    (SFT step1000/15000/43000, DPO, RLVR first/last). Does excess-d50 ~ 0 arrive
>    abruptly with SFT like everything else? ~1 h GPU.
> 2. **One non-safety control** — a sentiment direction through the identical
>    sweep. Refusal-specific or general? ~20 min GPU.
> 3. **The three-instruments figure** — fixed ablation (A1), input-norm dosing
>    (persona-vectors parameterisation, which our sweep already uses — read off
>    at their fixed c), boundary-relative dosing (Gate A). Same models, three
>    instruments, three different rankings, one resolution. This is Figure 1.
> 4. Self-contained captions; abstract with the 0.037-vs-0.98 numbers in it.
>
> ### Title/name
> Rename per Ji et al. collision. Working options, instrument-first:
> "Steering in the Wrong Units", "Boundary-Relative Dosing for Activation
> Steering", "Distance to the Boundary Is Not Steering Resistance".
>
> ### Carlini checks
> - De-risk: DONE (Gate A was exactly this).
> - "Why is everyone doing this obviously wrong?": three published groups, three
>   answers, one confound — yes.
> - Conclusion so-what: steering-based safety evals mis-rank models; any audit
>   that ablates a fixed direction understates a deployed model's steerability.
> - Luck/timing: J-lens follow-up crowd is 7 weeks old; instrument critique is
>   uncontested ground. Do not sit on it.


**A concrete paper plan.** Supersedes the L4/L5 doc's framing in light of the OLMo landscape
review (26 Aug) and the J-lens material already in the repo.

**Target:** InterpScience @ NeurIPS 2026 — *"Interpretability as a Science: toward rigorous
foundations for understanding LLMs."*

---

## 1. The thesis

> **Verbalizability, not decodability, predicts controllability.**
>
> Post-training does not erase a safety direction. It moves it out of the model's *verbalizable
> workspace* — and that, not any loss of information, is what breaks steering while leaving probes
> intact.

Three measurements on one direction across one training flow:

| Curve | Question | What the repo already shows |
|---|---|---|
| **1. Decodable?** | probe AUROC | **Survives** — frozen 0.812 vs refit 0.820 |
| **2. Verbalizable?** | J-space semantic coherence | **Collapses at chat-tuning** — 1.000 → 0.227; RL-Zero retains **0.909** |
| **3. Controllable?** | steering efficacy | **Collapses at chat-tuning** — causal 0.08; RL-Zero **0.865** |

**Curves 2 and 3 track each other almost exactly (0.227↔0.08, 0.909↔0.865). Curve 1 diverges from
both.**

That is a *mechanism* for a known-but-unexplained phenomenon. "Probe directions aren't causal
directions" is established. **Why** is not. J-space residency is a candidate answer, and it is
measurable.

---

## 2. Why this is spotlight-shaped at *this* venue

The field's stated open gap — from the training-dynamics review — is:

> *"When do causal directions emerge vs decodable ones? Nobody traces both curves across training on
> one lineage."*

**Two curves is the open gap. You have three, and the third explains the relationship between the
other two.**

And the workshop's own thesis is about what makes interpretability rigorous. The paper's spine is
therefore not a finding but a **measurement argument**: three operationalisations of "the direction
is still there," which disagree, and a principled account of which one predicts behaviour.

Most submissions will be "we found a thing." This is "here is what our constructs actually measure,
and here is the one that earns its keep."

---

## 3. Stop treating "it's general" as the failure case

**Two independent experiments in this repo already say the effect is not safety-specific**, and both
were filed as inconclusive:

| Experiment | Result | Where it sits |
|---|---|---|
| **E8 sentiment control** | Behaved **identically, only harder** | "inconclusive" section, absent from README |
| **E8 J-space control** | An unrelated direction de-verbalized the same way — *"a general de-verbalization of base-era directions in SFT-descended checkpoints"* | Demoted from mechanism to correlation by protocol P3 |

Both point the same way. And the landscape review independently rates **safety-vs-general-concept
comparison as "OPEN, fully unclaimed — the most defensible framing available."**

**So the general version is not the fallback. It is the likely result, it is unclaimed, and it is the
bigger claim:**

> Post-training de-verbalizes base-era directions *as a class*, and de-verbalization — not
> information loss — is what breaks steering.

That applies to everyone who ships a steering vector against a fine-tuned model. Safety becomes the
motivating case, not the scope.

---

## 4. What you already have

More than the audit implied. Inventory:

- **J-lenses for the entire OLMo-3 flow** — a third-party ladder of **11 lenses** (base, Instruct
  SFT/DPO/final, Think SFT/DPO/final, RL-Zero math/code/if/general), plus the official
  `neuronpedia/jacobian-lens` for base + 32B.
- **A provenance check on them** — 64 random vectors through two independent fits agreeing at cosine
  0.978→0.9998 (L12–L30). *Better lens hygiene than the affective-coupling project managed.*
- **`jlens.from_hf` auto-detects Olmo3 with zero patching.**
- **E8 already run** — 11 checkpoints × 5 layers × 3 directions × 200 nulls, in **6.5 minutes**, no
  fitting required.
- **An independent localisation result**: the lens puts refusal at **L16–L24, not L12** — corroborating
  E5's attribution (L15–L23 MLPs) over E4's ablation (L12). An independent method adjudicating
  between two of your own conflicting results.
- Attribution patching at **~3 s/checkpoint**.
- Ai2's four RLVR domain series + RL-Zero, **unprobed by anyone**.

**The marginal cost of the spotlight paper is much lower than it looks.** The expensive parts are
already done or already cheap.

---

## 5. Two gates — run before anything else

Both are cheap, both are decisive, and **each has a publishable outcome either way.**

### Gate A — dose-matched steering *(one afternoon)*
Curve 3's collapse may be an artifact. The base direction removes **6.81** logits from Instruct vs
**4.37** from base (non-overlapping CIs) — steering got *stronger*; the crossing rate falls only
because Instruct starts at +7.85 vs base's +0.85.

Re-measure with equal displacement (`c_50`) or norm-normalized α. **Also refit the direction with
mass-mean** rather than logistic regression — Marks & Tegmark report mass-mean gives more causally
implicated directions, and a reviewer will ask.

- **Collapse survives** → curve 3 stands, proceed.
- **Collapse vanishes** → the paper becomes the methods note in §8, and you have independently
  replicated Moskvoretskii with a null calibration they lack.

### Gate B — one chat-fit lens *(~10 GPU-h)*
Curve 2's collapse may be the lens linearising chat-tuned models worse — the lenses were fit on
plain web text, and every checkpoint that drops is chat-tuned while RL-Zero (which doesn't drop)
isn't. That correlation is perfect and damning.

Fit a lens on chat-formatted data for **one** Instruct checkpoint. Do not fit the ladder.

- **Coherence stays at 0.227** → real de-verbalization, proceed.
- **Coherence recovers** → J-lens linearisation degrades on chat-tuned models. That is a
  **lens-validity finding of direct use to the J-lens community**, and squarely venue-appropriate.

> **Both gates fail → you still have two publishable methods findings and no wasted compute.**
> That property is why they go first.

---

## 6. Experiments

### E1 — The control suite *(the anchor)*
Safety (harmfulness, refusal) vs **arbitrary-but-decodable** (hash partition — the pure null),
**syntactic** (POS/agreement), **factual** (entity type), **stylistic** (sentiment — *already run;
report it*).

All matched on **base-probe AUROC**. Report **between-concept difference intervals with TOST**.

Run all three curves on every concept. This is what converts a correlation into a claim about
whether safety is special.

### E2 — Three curves across the full flow
Decodability · J-space coherence · dose-matched steering, per concept, per checkpoint. Include
midtraining and long-context stages, which nobody has touched.

### E3 — Four RLVR domains *(cheap, unclaimed)*
Math, code, instruction-following, general chat. Converts "RL is representationally inert" from a
single-lineage null into **"inert across four independent reward domains."** Pure re-run.

RL-Zero is also your **natural control for chat-tuning** — RL on base without chat SFT. If
de-verbalization tracks chat-tuning rather than post-training volume, RL-Zero is the arm that proves it.

### E4 — Probe transfer matrix
Fit at checkpoint *i*, evaluate at *j*, full matrix, per concept. **Nobody has one.** Cheap,
visually striking, and the natural figure for monitor staleness.

### E5 — Methodological floor *(a contribution in itself here)*
TOST on every null · direction-reliability ceiling via split-half at the same checkpoint · subspace
agreement not single-direction cosine · random-init (`stage1-step0`) floor · continuous **and**
thresholded metrics for any transition claim · layer-marginalized reporting.

### Cut
Claim 4 (owned by prior work) · claim 6 (established) · the "exactly 1.00" circuit-overlap claim
(no code, no data, and Tigges et al. kills the standalone version).

---

## 7. Paper skeleton

| § | Content |
|---|---|
| **1. Intro** | Steering vectors go stale after fine-tuning; probes don't. Everyone has hit this; nobody has explained it. **Rename** — "elasticity" collides with Ji et al., ACL 2025 Best Paper, in the inverse sense. Disambiguate in ¶1. |
| **2. Setup** | The OLMo-3 flow as a measurement substrate. Ai2 has published no representation work on it and explicitly invites this. |
| **3. Three operationalisations** | Decodable / verbalizable / controllable. **The core methodological contribution.** Define each, state what each would mean if they disagreed. |
| **4. They disagree** | The main result. Curves 2 and 3 track; curve 1 diverges. Per-concept. |
| **5. Is it safety-specific?** | E1. Report the answer honestly whichever way it lands. |
| **6. Null calibration** | Shuffled-label, random-direction, random-init floors. TOST on every null. |
| **7. What our criterion could not decide** | The dose confound and how it manufactured an apparent decay; the chat-lens confound and Gate B's verdict; the control that behaved the same. **This is the section that wins the venue.** |
| **8. Related work** | Moskvoretskii (same model, opposite conclusion — engage head-on), 2509.23024, 2606.15980, Tigges, Du, Qi, Marks & Tegmark, Ji. |

---

## 8. Figures — four carry the paper

1. **The three curves** — decodability, J-coherence, steering, on one x-axis over the flow. One panel
   per concept. *This is the paper in one image.*
2. **Curve 2 vs curve 3 scatter** — the 0.227↔0.08 / 0.909↔0.865 correspondence, with the control
   concepts overlaid. Shows the mechanism.
3. **Probe transfer matrix** — heatmap, fit-checkpoint × eval-checkpoint.
4. **The dose confound** — crossing rate vs α (misleading) beside crossing rate vs achieved
   displacement (correct). The methods figure.

**Every figure regenerates from `results/*.json`.** Currently 23 of 29 are orphans and two contradict
the text — that must be fixed regardless.

---

## 9. Threat register

| Threat | In-text handling |
|---|---|
| **Moskvoretskii 2605.13329** — same model, finds base vectors *still steer* | Engage directly. If Gate A agrees with them, say so and contribute the null calibration they lack |
| **2509.23024** — OLMo geometry, pretraining→post-training | Yours is concept-level and goes past DPO to RLVR ×4 and RL-Zero |
| **2606.11375** — probe accuracy saturates by step 4K | Don't lead with a formation curve; report fragility-style metrics |
| **2606.15980** — owns "info preserved, basis rotates" | Cut claim 4 as a headline |
| **Tigges NeurIPS 2024** — circuits stable, heads churn | Drop the standalone circuit claim |
| **Ji et al. ACL 2025** — "elasticity" collision | Rename |
| **LessWrong OLMo harmfulness post** | **Read before writing** — may contain claim 1 and the timing gap |
| Third-party lens provenance | You already have the cosine check — put it in an appendix |

---

## 10. Timeline

**29 Aug (3 days) does not fit this paper.** Gate B alone is ~10 GPU-h and E1 is ~a week.

- **Now → 29 Aug:** submit the **methods note** — the dose confound, null calibration, and "what our
  criterion could not decide." No new compute. Venue-appropriate, and it becomes §6–7 of the full paper.
- **Sept:** Gates A and B, then E1.
- **Oct–Dec:** E2–E5, write.
- **Target:** ICLR 2027 workshops, or **ICML 2027 (~22 Jan)** for the full paper.

---

## 11. What would make this fail

Be honest about these up front — they are the reviewer's questions.

1. **Gate A kills curve 3** → no dissociation to explain. Falls back to a methods note. *(Given the
   6.81 vs 4.37 evidence, treat this as the most likely single outcome.)*
2. **Gate B shows a lens artifact** → curve 2 is measurement noise. Falls back to a lens-validity finding.
3. **All concepts behave identically AND the effect is weak** → nothing to report beyond "fine-tuning
   changes things." Unlikely given E8's magnitudes, but state the criterion in advance.
4. **The LessWrong post already contains it.** Check first.

**Two of four failure modes still produce a workshop paper.** That is the argument for running the
gates now rather than building the full apparatus first.
