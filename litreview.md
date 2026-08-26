# Literature review: OLMo-checkpoint interpretability (27 Aug 2026)

Verification sweep of prior art bearing on the alignment-elasticity project. Supersedes
earlier informal reviews. **Headline: substantially more of our observational programme is
already claimed than the plan assumed — including two of our three headline findings.**

---

## 1. Direct prior art on OLMo checkpoints

| Work | Coverage | What it already establishes |
|---|---|---|
| **"Harmfulness Directions in OLMo"** (LessWrong) | **39 OLMo-3 7B checkpoints**, full trajectory: pretraining s1/s2/s3, base, SFT, DPO, Instruct | Linear probes at L15, safe-to-harmful centroid directions (= diff-in-means), **in-distribution AND OOD AUROC**, behavioural steering across coefficients. Finds geometry "forms in the first few thousand steps rather than gradually"; **"largest single shift" at the SFT transition**; post-training directions steer *better* than pretraining ones |
| **Tracing Persona Vectors Through LLM Pretraining** (2605.13329, May 26) | OLMo-3-7B pretraining + Apertus-8B; steers base, SFT, DPO, **RLVR** | Persona vectors form **within 0.22% of pretraining** and **remain effective for steering fully post-trained instruct models** |
| **When Probing Accuracy Saturates, Fragility Resolves** (2606.11375, Jun 26) | OLMo-2 1B early-training (37 ckpts) + OLMo-3 7B (20 ckpts) | Probe accuracy on moral/neutral pairs saturates ~95% **by step 4K**; introduces *fragility* (noise level at which the probe collapses) as the fix |
| **Tracing Representation Geometry: Pretraining to Post-training** (2509.23024) | OLMo + Pythia, through SFT/DPO/**RLVR** | Three geometric phases; **SFT/DPO entropy-seeking vs RLVR compression-seeking** |
| **Tracing Eval-Awareness Emergence Through OLMo 3** (Goodfire/UK AISI, LessWrong) | 5 stages incl. **10 RLVR checkpoints**, OLMo-3 vs 3.1 | SFT introduces eval-awareness, DPO suppresses, RLVR re-amplifies ~2x. **Purely behavioural — no probing or steering.** Calls RLVR mechanisms "the exciting next step" |
| **Trustworthiness Dynamics** (2402.19465) | OLMo pretraining | Middle-layer linear separability of trustworthiness emerges early |
| **Crosscoding Through Time** (2509.05291) | Pretraining | Crosscoders tracking emergence/consolidation of representations |

## 2. Adjacent claims that fence us in

| Work | Owns |
|---|---|
| **Do Activation Monitors Survive Model Updates?** (2606.15980) | **"Information preserved, basis rotates."** Verbatim: staleness is "drift rather than erasure — the signal survives but the frozen readout direction silently falls out of alignment." Also: refusal-compliance probes comparatively stable; quantization benign, fine-tuning harmful |
| **Obfuscation Atlas** (2602.15515) | Representation drift **during RL** with deception probes. Tension with our "RL is inert" — but theirs is RL *against a detector penalty*; ours is unpressured RLVR. That distinction is ours to make |
| **A Single Neuron Is Sufficient** (2605.08513) | Refusal signal discriminative **before** alignment; encoded in pretraining |
| **Ji et al., ACL 2025 Best Paper** — *Language Models Resist Alignment* | The word **"elasticity"**, in the *inverse* sense (models revert to pretrained behaviour). **Name collision — rename** |
| **Anthropic, Verbalizable Representations / J-space** (2607.15495, Jul 26) | J-lens, J-space. **Critically: they already report directions that steer strongly while sitting OUTSIDE J-space** — so verbalizability does not simply predict controllability |
| **Tigges et al.** (NeurIPS 2024) | Circuits stable while heads churn — kills a standalone circuit-stability claim |
| Decodability != controllability generally (2608.12334 and others) | The dissociation itself is **established**; what is missing is a *mechanism* |

## 3. What is genuinely open

Searched and found no claimant:

- **(a) Safety vs non-safety concept comparison across a training flow.** Nobody has run matched
  control concepts (sentiment, syntax, arbitrary-but-decodable) alongside safety concepts through
  the same pipeline. This decides whether any of this is an alignment claim at all.
- **(b) RLVR x 4 domains and RL-Zero at the representation level.** The eval-awareness study is
  behavioural and explicitly defers RLVR mechanisms; the harmfulness post stops at Instruct.
  Ai2 built RL-Zero decontaminated *specifically* to enable this study. Unclaimed.
- **(c) Verbalizability traced across a training flow.** J-lens is seven weeks old. Nobody has
  traced J-space residency developmentally. Our within-direction/across-training claim is
  complementary to Anthropic's cross-sectional one — but must be framed as such, not as
  contradicting it.
- **(d) Dose-matched steering with null calibration.** Prior steering studies vary the
  coefficient; none equalise *achieved displacement*. Our A1 result — that raw displacement
  inverts the ordering (deployed model 156% of base by displacement, 52% by crossing rate) —
  is unstated in the literature and explains a live disagreement (see section 4).
- **(e) Probe transfer matrix** (fit at checkpoint i, evaluate at j) across a full flow.
- **(f) Three operationalisations measured on one direction** — decodable / verbalizable /
  controllable — with difference intervals and equivalence testing.

## 4. The live disagreement we can adjudicate

Three groups, three answers about whether base-era directions still steer post-trained models:

| Source | Answer |
|---|---|
| Persona vectors (2605.13329) | Vectors **remain effective** on post-trained instruct models |
| Harmfulness Directions (LessWrong) | Post-training directions steer **better** than pretraining ones |
| **This project (A1, n=200)** | Base direction moves the deployed model **further** (156% of base displacement) yet flips **fewer** decisions (crossing rate 0.350 vs 0.670) |

**CAUTION (verified 27 Aug, methods section of 2605.13329):** the persona-vectors paper is more
careful than the summary above implies. It **normalises the steering coefficient by the local
residual-stream norm** so that dose is "comparable across checkpoints"; it **reports per-model
unsteered baselines separately**; it reports **pass rates** (fraction of generations exceeding a
trait-score threshold) alongside the raw delta; and it uses **paired permutation tests**. So the
simple critique "they measured displacement, we measured outcome" does NOT land.

What survives is narrower and must be stated precisely: their normalisation is **input-side**
(activation norm), not **boundary-relative**. Our A1 result shows the unsteered distance to the
behavioural decision boundary varies ~9x across the flow (+0.85 logits at base vs +7.85 at
Instruct). An input-side norm match does not equalise that, and an absolute pass-rate threshold
does not control for it either. That is a refinement of their method, not a refutation — and any
write-up must present it that way.

These are not contradictory — they are the same phenomenon under different metrics. A direction
that produces a larger logit displacement on a model sitting deeper inside refusal can still
change fewer decisions. **Nobody has said this.** It is cheap to demonstrate, it reconciles
published results, and it is exactly the kind of contribution an "interpretability as a science"
venue exists for.

## 5. Verdict on our original novelty claims

| Original claim | Status |
|---|---|
| Safety representations form early in pretraining | **CEDE** — LessWrong post, persona vectors, trustworthiness dynamics, single-neuron paper |
| The rise is front-loaded, not gradual | **CEDE** — "forms in the first few thousand steps rather than gradually" (LessWrong) |
| Post-training moves geometry once, at SFT | **CEDE** — "largest single shift at the SFT transition" (LessWrong) |
| Information preserved while basis rotates | **CEDE** — 2606.15980 states it verbatim |
| Probe metrics saturate and need a harder split | **CEDE the problem**, keep our solution as a variant — 2606.11375 owns the problem and offers *fragility*; we offer *adversarial OOD*. Compare, do not claim |
| RL is representationally inert | **PARTIAL** — 2509.23024 covers RLVR geometrically; Obfuscation Atlas finds drift under *pressured* RL. Our unpressured RLVR x 4 domains + RL-Zero is still open |
| Formation timing does not predict elasticity | **OPEN but underpowered** — needs the clean honesty re-run and control concepts |
| Reading survives, steering decays | **CONTRADICTED as stated**; survives as a *metric* claim (section 4) |
| "Alignment elasticity" as a name | **COLLIDES** with Ji et al. — rename |

## 6. Recommended positioning

Do not lead with formation or with the SFT cliff. Lead with the **measurement argument**:

> Published results disagree about whether safety directions survive post-training. We show the
> disagreement is an artifact of measuring displacement rather than outcome, and that a third
> operationalisation — verbalizability — tracks controllability where decodability does not.
> Matched non-safety controls decide whether any of this is specific to safety.

Contributions, in order of defensibility: (1) the dose/displacement confound and the
reconciliation of section 4; (2) matched safety-vs-general controls with equivalence testing;
(3) RLVR-domain and RL-Zero representational coverage; (4) verbalizability as a
developmental curve.
