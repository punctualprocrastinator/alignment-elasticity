# Alignment Elasticity Along the Model Flow

**When do safety-relevant representations form, what sets them, and can training-time interventions change how durably they stick?**

*Status: observational phase complete (6 Aug 2026). Interventional phase not started.*
*Target: ICLR 2027 main track (abstracts ~mid-Sept 2026); ARR as fallback.*
*Relevant to: alignment-pretraining / robustness-to-RL work, and AI-control monitoring research.*
*Hardware: 1-4x RTX PRO 6000 Blackwell (102 GB) via molab sandboxes.*

> Detailed results, numbers, retractions and engineering gotchas live in `pipeline-notes.md`.
> Checkpoint inventory lives in `checkpoints.md`. Figures in `figures/`. Report: `report.html`.

---

## 1. Where the project actually stands

One day of parallel sweeps (8 experiments, ~120 checkpoint evaluations) settled the observational
questions and **falsified the original headline hypothesis**. Summary of what is now measured:

| Claim | Status | Key number |
|---|---|---|
| Harmfulness representation forms early in pretraining | **Confirmed** | Departs chance ~step 2,000 (7B); AUROC 0.547 -> 0.804 by end of stage 1 |
| The early rise is front-loaded, not log-linear | Confirmed (1B) | Step 1k->2k carries ~50% of total rise, 3.2x SD of later changes |
| Post-training moves geometry once, in early SFT | **Confirmed** | Mean-cos delta 0.230 (SFT 1k->15k); RLVR total drift 0.0006; RL-Zero cos 0.998 |
| Information survives while the basis rotates | Confirmed | Frozen probe 0.888 vs refit 0.929 AUROC; direction rotates to cos ~0.50 |
| Behaviour and geometry co-move as one event | Confirmed | HarmBench ASR 0.350 -> 0.000 within first 1k SFT steps, flat after |
| Refusal circuit re-weighted once, then frozen | Confirmed | Top-20 Jaccard 1.00 for every pair from SFT-43k through RLVR; base top-5 retains 84-87% causal effect |
| Reading survives, steering decays | **Confirmed** | Base direction retains 86-87% causal effect on RL-Zero, 8% on Instruct (matched interface) |
| Black-box monitors go stale with training distance | **Refuted** | Gap-vs-AUROC correlation -0.26; capability, not freshness, is the binding constraint (base monitor catches 1.7% @1% FPR vs SFT's 43%) |
| Formation timing predicts elasticity | **FALSIFIED** | Honesty forms ~2.5x later (step ~5,000) yet drifts identically; frozen-refit gap 0.012 vs harm's 0.04 |

**Revised thesis.** Elasticity is a property of *the pipeline stage*, not of the concept: everything is
set in early SFT and capabilities RL is representationally inert. The sharpest open claim is the
**read/steer dissociation** — a direction stays readable while losing causal control.

**What the paper is now about:** (i) the single-event structure of alignment formation across a full open
flow, (ii) read/steer dissociation and what it means for monitors vs interventions, (iii) whether
training-time interventions can move any of it (Phase B).

---

## 2. Hypotheses, with explicit failure points

Each hypothesis states what result would kill it. A hypothesis without a failure point is not one.

### H1 — The SFT cliff is caused by safety data, not by format acquisition
Fine-tuning on instruction data *without* safety/refusal examples will not rotate the refusal direction.
- **Predicts:** no-safety SFT gives cos-to-base >= 0.95 at 15k steps; full Dolci SFT reproduces ~0.68.
- **Failure point:** if the cliff appears without safety data, the rotation is instruction-format
  acquisition, not alignment learning — and every "alignment is set by SFT" statement in the paper must
  be reworded to "alignment-relevant geometry moves when the model learns the assistant format."
- **Why it matters most:** this single result reinterprets the entire observational phase. Run it first.

### H2 — Alignment-relevant priors injected during (mid)training increase robustness to later RL
The alignment-pretraining thesis ("priors baked in early survive later training"),
tested as an A/B rather than inferred from an observation.
- **Predicts:** intervened branch shows smaller post-RL drift and/or higher retained causal control than
  the control branch after *identical* SFT + RL.
- **Failure point:** no difference between branches. Given the observational null (RL is already inert),
  the realistic risk is a **ceiling effect** — nothing to improve. Mitigate by pairing with H3: measure
  robustness under RL pressure strong enough to actually break the control branch.

### H3 — There exists an RL dose/objective that breaks alignment geometry
Our "robust to this much RLVR" null is weak until we find where it stops holding.
- **Predicts:** a threshold exists in (learning rate x steps x objective hackability) beyond which cos-to-base
  falls below ~0.9 and ASR rises materially.
- **Failure point:** alignment survives even deliberately adversarial RL at 1B/7B scale. That is a *strong,
  publishable* result — "representations are far more RL-robust than the field assumes" — but it must be
  reported as a bounded search, with the dose grid explicitly stated.

### H4 — Curriculum order changes durability, holding data quantity fixed
Safety data front-loaded vs back-loaded in SFT.
- **Predicts:** front-loaded gives a direction that retains more causal control after subsequent RL.
- **Failure point:** no difference -> durability is set by *whether* alignment data is present, not *when* —
  consistent with H-formation's falsification and worth stating as such.

### H4b — Elasticity is alignment-specific, not a general property of linear directions
**The most dangerous open question in the project.** Every result so far says the
flow sets everything in early SFT and RL touches nothing. If a NON-SAFETY concept
(sentiment, topic, formality) shows the identical cliff-then-flat pattern, then
none of this is about alignment — it is about what fine-tuning does to linear
structure generally.
- **Predicts (for the paper's current framing to hold):** safety-relevant concepts
  show larger SFT rotation and/or different causal-retention behaviour than
  non-safety concepts at matched probe quality.
- **Failure point:** non-safety concepts behave identically. Then the honest paper
  is "fine-tuning rotates all linear structure once, early, and RL rotates none of
  it" — still novel and still useful to control, but not an alignment claim.
- **Early warning already exists:** in E8, a sentiment direction de-verbalized
  across the flow exactly like refusal, only harder.

### H4c — Formation and elasticity are properties of the concept, not the dataset
Concept and dataset are currently one-to-one (harmfulness IS WildGuardMix,
honesty IS Azaria-Mitchell), so every formation time is confounded with dataset
decodability.
- **Predicts:** two independent datasets for the same concept give onsets within
  noise of each other and the same elasticity.
- **Failure point:** onsets differ materially between datasets for one concept —
  in which case "when concept X forms" is not a well-defined quantity and every
  formation claim must be restated per dataset.

### H5 — Read/steer dissociation generalises beyond refusal
Frozen probes transfer while base-era steering decays, for honesty and at least one non-safety property.
- **Predicts:** same pattern (readout retention high, causal retention low) on honesty.
- **Failure point:** if steering retention is high for honesty, the dissociation is refusal-specific — still
  interesting, but a narrower claim, and E8's sentiment control already hints the phenomenon is general.

### H6 — Monitor capability, not monitor freshness, is the binding constraint (control claim)
- **Predicts:** at fixed FPR, catch rate is predicted by the monitor's own capability, with training
  distance adding no additional explanatory power in a regression.
- **Failure point:** distance term becomes significant once the experiment is properly powered — entirely
  possible, since the current version has too few harmful positives to resolve it.

---

## 3. Smoke tests (mandatory pre-flight)

Yesterday produced **five retracted claims and four silent bugs**, every one caught by a control rather than
by inspection. These checks are now mandatory before any sweep is trusted. Each is cheap; skipping one has
already cost a wrong figure.

### 3.1 Metric validity
- [ ] **Random-init baseline.** Run the metric on `stage1-step0`. If it scores well above chance, the metric is
      reading surface form. *(A randomly initialised net scored 0.81 accuracy on the easy harm split.)*
- [ ] **Shuffled-label control.** Must sit at 0.49-0.51. Anything else means leakage.
- [ ] **Report AUROC and accuracy together.** Accuracy conflates ranking with threshold and has been
      misleading in every experiment so far. AUROC is the headline; accuracy is a calibration diagnostic.
- [ ] **Dynamic range check.** If the metric saturates (>0.98) or is pinned at chance across all checkpoints,
      stop — it cannot support a curve. This is what the adversarial contrast split exists to fix.

### 3.2 Format and tokenisation
- [ ] **Three-format probe on fixed weights.** Same checkpoint, raw / neutral scaffold / chat template.
      Expect a large *calibration* gap and a small *ranking* gap. A large AUROC gap means format is
      confounding the experiment.
- [ ] **One format per figure.** Never mix raw and templated numbers in a single curve. Base checkpoints have
      no template; a common anchor must be chosen explicitly.
- [ ] **Template drift check.** When comparing templated checkpoints, diff the actual template strings — Think
      SFT/DPO/RLVR ship different ones, so part of any measured drift is template *text*.
- [ ] **Truncation audit.** Log the fraction of prompts hitting `max_length`. Use **left** truncation whenever
      the last token is the readout position. *(128 + right-truncation silently cut 76% of adversarial prompts
      mid-jailbreak and pinned AUROC at 0.50.)*
- [ ] **Padding/readout check.** Confirm the "last token" index lands on a real token, not padding.

### 3.3 Hooks, directions, and causal machinery
- [ ] **Hook liveness.** Assert the intervention changes logits. Olmo3 decoder layers return a bare Tensor, not
      a tuple — a hook written for the tuple silently no-ops.
- [ ] **Ablation completeness.** Include `embed_tokens`, not just decoder layers; the embedding re-injects the
      direction (2/6 -> 6/6 refusal flips when added).
- [ ] **Per-layer variation check.** Assert values differ across layers. *(An `attn_sum +=` broadcast produced
      32 identical rows and a convincing fake "head 2 dominates everywhere" result.)*
- [ ] **Degenerate-direction check.** Reject any layer whose mass-mean direction is ~zero. *(Layer 0 is the
      embedding of an identical final token; it deflated every layer-averaged cosine by 32/33.)*
- [ ] **Random-direction and mid-rank controls** on every causal claim. Effect must exceed both.
- [ ] **Generation-length check.** Confirm completions are long enough to contain the judged behaviour.
      *(44-token generations truncated inside `<think>`, making refusal classification meaningless.)*

### 3.4 Specificity and power
- [ ] **Non-target control direction.** Any claim of the form "X is special about the safety direction" must
      show a non-safety direction (sentiment/topic) behaving differently. *(This is what demoted the J-space
      result from mechanism to correlation.)*
- [ ] **Power statement.** Report n and the minimum resolvable effect. Causal claims need >=100 prompts;
      n=6-8 gives effects quantised to 0.125 and cannot support a headline.
- [ ] **Multiple operationalisations.** Where a construct has more than one reasonable measure, report all and
      state disagreements rather than selecting the flattering one.

### 3.5 Data and checkpoint hygiene
- [ ] **Dedupe on prompt text.** WildGuardMix has one row per *response*; 38,907 duplicates otherwise.
- [ ] **Duplicate-branch check.** Hash the safetensors. *(RL-Zero-Code step_100 and step_1000 are
      byte-identical upstream.)*
- [ ] **Pin revisions.** Every result file records the exact HF revision string.
- [ ] **Purge weights between checkpoints** and track HF cache size (`shutil.disk_usage` is unreliable on molab).

### 3.6 Infrastructure
- [ ] **Long jobs run in a kernel-side thread.** Client disconnects and any `run_cell` interrupt a running cell.
- [ ] **Idempotent, disk-backed sweeps.** Skip completed revisions; a restart must never lose work.
- [ ] **Render-and-inspect every figure** before it is used; overlap bugs are invisible in code.

---

## 4. Experiment program

### Phase A — harden the observational paper (weeks 1-2)

| # | Experiment | Purpose | Cost |
|---|---|---|---|
| A1 | Re-run causal staleness and layer profiles at n=200 with bootstrapped CIs | Current n=6-8 will not survive review | ~4 GPU-h |
| A2 | Non-linear probe (MLP head) + mean-pooled readout on the formation sweep | Rules out "linearly undecodable != absent" | ~6 GPU-h |
| A3 | Fixed-template deconfound across post-training checkpoints | Separates template text from representation drift | ~3 GPU-h |
| A4 | 32B replication, 5 checkpoints (base/SFT/DPO/RLVR/RL-Zero) | Kills the "one scale" objection | ~1 day |
| A5 | Power E6 with adversarially-elicited harmful completions | Current monitoring curve is flat within noise | ~6 GPU-h |
| A6 | Recover RL-Zero-Mix (4 ckpts) + 3.1-Code step_1900 | Completes dose-response to 4 domains | ~30 min |
| A7 | Length-matched adversarial control | Adversarial prompts are ~7x longer than vanilla | ~2 GPU-h |
| A8 | **Concept matrix** (see below) | Two properties support "the flow decides" weakly; and concept is confounded with dataset | ~1-2 days |

### A8 concept matrix (expanded 2026-08-07)

Two axes: more safety concepts, AND non-safety controls. Two datasets per concept
so concept is not confounded with dataset (H4c).

| Concept | Datasets | Why it is in the set |
|---|---|---|
| Harmfulness | WildGuardMix; HarmBench-derived | anchor (measured) |
| Honesty | Azaria-Mitchell; TruthfulQA-derived | anchor (E7-clean running) |
| Sycophancy | Anthropic model-written evals | the ONE concept the literature says RLHF should move — discriminating test of our RL null |
| Power-seeking / self-preservation | Anthropic advanced-AI-risk evals | classic alignment concept, public, familiar to reviewers |
| Eval-awareness | constructed eval-like vs deployment-like | highest novelty, weakest prior art — exploratory |
| **Sentiment** | SST-2; IMDB | **non-safety control (H4b)** |
| **Topic / formality** | constructed | **second non-safety control (H4b)** |

Run each through the same protocol-v1 formation + elasticity pipeline. The
headline output is a **concept x (onset, SFT rotation, RL drift, frozen-refit
gap, causal retention) table with CIs**, and the between-concept difference CIs —
equivalence claims need difference intervals, not similar-looking point estimates.

### Concept validation criteria (adapted from Anthropic's emotion-concepts work,
arXiv 2604.07729, Apr 2026)

Probe AUROC alone cannot distinguish "the model represents X" from "the probe
reads X-related vocabulary". Every concept entering the matrix must satisfy:
1. **Contextual activation** — fires in appropriate contexts, quiet otherwise
   (this is what AUROC already measures; necessary, not sufficient).
2. **Causal influence under graded steering** — see below.
3. **Internal-external decoupling** — cases exist where the internal
   representation and the external behaviour disagree (direction active while the
   model complies; quiet while it refuses). A concept that never decouples may be
   an echo of surface form rather than an internal state.
   *Bonus for H4b:* if safety concepts decouple and non-safety controls do not,
   that is evidence the safety concepts are special after all.

### Graded steering replaces binary ablation (protocol addition)

Current causal measure is all-or-nothing (project out the direction, count
flips). Replace with a **dose-response curve**: sweep the steering coefficient
and record the behavioural response, reporting the coefficient at which the
model crosses from refusing to complying (`c_50`) with a bootstrap CI.
- *Why:* far more sensitive, and it is the right instrument for the read/steer
  dissociation. A1 showed Instruct sits at +7.85 logits of refusal vs base's
  +0.85, so "how much push is needed to cross zero" is the natural quantity —
  a proper effect size instead of a rate.
- Report `c_50` per checkpoint per concept; the elasticity claim becomes "how
  does the steering dose required to change behaviour evolve along the flow".
- Anthropic's result that a 0.05 amplification moved blackmail 22% -> 72% with
  no trace in the output text is also directly relevant to the control arm:
  behavioural monitoring can miss internal-state manipulation entirely.

### SAE work — NOW IN SCOPE (pre-trained SAEs exist, 2026-08-07)

`decoderesearch/olmo-3-saes` on HF (listed at neuronpedia.org/olmo3-7b-sae) has
SAEs for **allenai/Olmo-3-1025-7B base**: layers **4, 16, 28**, matryoshka
architecture, 65k width, k=100. Load via `SAE.from_pretrained(
"decoderesearch/olmo-3-saes", "<sae_id>")` (`pip install sae-lens`).
**Base model only** — nothing for SFT/DPO/RLVR, as expected.

Layer coverage happens to suit us: L4 is the lexical-shortcut layer that
collapses under adversarial reframing, L16 is where the generalising harmfulness
representation lives (and the best layer under mean pooling), L28 is deep. Our
refusal direction sits at L20, not covered — use L16 as the nearest proxy and
say so.

**Never train one SAE per checkpoint.** The blocking problem is not cost but
comparability: features from independent training runs have no correspondence
(different init -> different basis and ordering), so per-checkpoint SAEs cannot
measure drift at all. Exactly two designs work, each needing one training run —
or in our case, zero:

- **(a) Frozen dictionary across the flow** *(free — the base SAE already
  exists)*. Apply the base SAE unchanged to every later checkpoint; measure
  reconstruction quality, feature death, and activation-frequency drift. This is
  the feature-level analogue of the frozen-probe result (which loses only ~0.007
  AUROC across the flow).
- **(b) Crosscoder** on base + post-SFT (+ post-RLVR) with per-checkpoint
  decoders — the purpose-built tool for "which features did SFT create or
  destroy". Still requires training; remains the stretch goal.

Three experiments unlocked by (a):
1. Frozen-dictionary survival across the flow (above).
2. **Concept discovery that dissolves the dataset confound (H4c):** find features
   firing on harmful prompts at base, then track those feature directions across
   the flow with existing cheap machinery. Concepts then come from the model, not
   from a dataset we chose.
3. **Decompose the refusal direction into SAE features** — what is it made of,
   and does its composition change across the flow? A far more mechanistic
   account of the ~60 degree rotation than a cosine.

**Two mandatory caveats.** (i) Third-party SAEs with unstated training details
(tokens, dataset, sparsity) — apply the same provenance check used for the J-lens
ladder: validate reconstruction quality on base before trusting anything
downstream. (ii) A base-trained dictionary WILL reconstruct later checkpoints
worse partly because it is stale, so an H4b-style control is required: does
reconstruction degrade equally on a non-safety subspace? Without it, "the SAE
fits worse" gets misread as "the representation changed."


### Phase B — model organisms and training-time interventions (weeks 3-5)

Ai2 released Dolma 3, **Dolci** (post-training data), **Open Instruct** (post-training pipeline), OLMo-core and
OlmoRL, all Apache 2.0 — so counterfactual branches from any checkpoint are reproducible.

| # | Experiment | Hypothesis | Notes |
|---|---|---|---|
| B0 | **Reproduction spike:** ~1k steps of Dolci SFT from base; confirm the cliff appears where released checkpoints say | — | Gate for everything below. If our control branch does not match the released SFT checkpoint, stop and fix the pipeline. |
| B1 | SFT with safety/refusal data ablated vs full Dolci | H1 | Cheapest high-information experiment; run first after B0 |
| B2 | Midtraining injection of alignment-relevant corpus, then identical SFT+RL, intervened vs control | H2 | 1B for the full counterfactual flow; 7B if budget allows |
| B3 | Adversarial RL dose grid (LR x steps x hackable objective) until geometry breaks | H3 | Converts a weak null into a measured threshold |
| B4 | Curriculum order: safety data front- vs back-loaded, quantity fixed | H4 | Interventional counterpart to the falsified formation hypothesis |
| B5 | Deliberately backdoored organism: insert a known trigger at a known step; test whether probes/monitors catch it | H6 | Connects the control arm to a ground truth we constructed |

**Scale policy.** 1B is the primary organism (full flow reproducible in-house); 7B is used for SFT-only
interventions where our measurements already live. Pretraining from scratch is out of scope permanently.

**Engineering reality:** standing up Open Instruct and matching Ai2's SFT closely enough that the control
branch reproduces the released checkpoint is a 2-3 day lift *before* any science. Budget it explicitly.

---

## 5. Compute, cost, and throughput (measured, not estimated)

- Checkpoint download ~30 s (~15 GB); activation extraction ~10-50 s per 1-2k prompts; probe fitting
  ~36 s per checkpoint (32 logistic regressions). **~100-125 s per checkpoint end to end.**
- A 20-checkpoint sweep is ~40 min. Downloads are *not* the bottleneck; HF cache disk is (~15 GB/model,
  purge between revisions).
- Attribution patching is cheap (~3 s/checkpoint). Jacobian-lens *fitting* is not (~10 GPU-h/checkpoint) —
  use pre-fitted lenses only.
- Training costs (Phase B, to be measured in B0): 1B SFT 1k steps and 7B LoRA SFT both expected in hours.

## 6. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| H1 shows the cliff is format acquisition | Medium | Reframe rather than retract; the finding is stronger, not weaker, and B1 is cheap enough to learn early |
| Ceiling effect kills H2 (RL already inert) | **High** | Pair with H3 — measure robustness under pressure that demonstrably breaks the control |
| Open Instruct reproduction does not match released checkpoints | Medium | B0 is an explicit gate; fall back to LoRA-based interventions with a matched control branch |
| Results superseded by concurrent work | Medium | Post the workshop-length preprint early; the report page already exists |
| Statistical power objections | **High** | A1/A5 exist specifically for this; no causal claim ships at n<100 |
| Single lineage / single scale | High | A4 (32B) + a second lineage if time allows |

## 7. Timeline to ICLR 2027 (abstracts ~mid-Sept)

| Week | Milestone |
|---|---|
| 1 | A1-A3, A6-A7 (power, non-linear probes, deconfounds); preprint v0 posted |
| 2 | A4 (32B), A5 (monitor power), A8 (properties 3-4) |
| 3 | B0 reproduction gate + B1 (the H1 fork) |
| 4 | B2-B3 (intervention A/B, RL breaking point) |
| 5 | B4-B5 if B0-B3 landed; otherwise consolidate |
| 6 | Writing, ablations reviewers will ask for, submission |

## 8. Reproducibility

Config-driven runs, pinned HF revision strings, fixed seeds, per-experiment result JSON + saved probe
directions. Release code, probe weights, and the figure-generating notebooks. This audience will check.
