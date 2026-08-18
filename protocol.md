# Canonical measurement protocol (v1, 2026-08-07)

Every number that appears in the paper is produced this way. Deviations must be
stated in the result file and in the caption. Rationale for each rule is a
specific failure already observed — see `pipeline-notes.md`.

## P1. Splits and seeds
- `SEED = 42` for all probe/formation/elasticity work. Record the seed **and** a
  SHA-1 fingerprint of the split index in every result JSON.
- Aggregation refuses any result file whose fingerprint disagrees.
- Probe fits repeat over **>=3 seeds**; report mean and spread, not a single fit.
- *Why:* `pipeline.py` carried a different seed and silently shifted OOD AUROC by
  0.01-0.03 everywhere until caught by an exact-reproduction check.

## P1b. Solver convergence (added 2026-08-07)
- Fit logistic probes at `tol=1e-10`. sklearn's default `tol=1e-4` stops LBFGS
  after ~14 iterations, well short of the L2 optimum (objective 1.9685 vs 1.8224
  at 74 iterations), leaving the fitted **direction at cosine 0.978 to the true
  optimum**.
- *Why:* when the headline metric IS a direction cosine, ~0.02 of arbitrary
  solver-path rotation contaminates it. This plausibly explains part of the
  frozen-vs-refit gap correction (0.041 -> 0.007).
- **Any probe-*direction* number produced before this rule must be re-checked.**
  Probe AUROCs are largely unaffected.

## P2. Sample sizes (minimums)
| Quantity | Minimum n |
|---|---|
| Probe pools (per cell of the 2x2 contrast) | 1,000 |
| Causal ablation readout | 200 held-out prompts |
| Behavioural generation check | 100 prompts, >=200 new tokens |
| Monitoring / detection | >=150 harmful positives |
| Random-direction null | 20 directions |
| Bootstrap resamples | 1,000 |
- *Why:* n=6-8 gave effects quantised to 0.125 and produced an ordering that
  inverted at n=200.

## P3. Nulls and controls (mandatory on every claim)
- **Shuffled-label null** with an **independent permutation per (layer, pooling,
  seed)**. A single shared permutation inflated the null band to 0.658 — wide
  enough to swallow the entire formation curve.
- **Random-direction null** (matched norm) for every causal claim; report the
  effect as a z-score against it.
- **Mid-rank control** for component/circuit claims.
- **Non-target control direction** (sentiment or topic) for any claim of the form
  "X is special about the safety direction". This demoted the J-space result from
  mechanism to correlation.
- **Random-init checkpoint** (`stage1-step0`) as the floor for any formation claim.

## P4. Metrics
- **AUROC is the headline everywhere.** Accuracy is a calibration diagnostic only
  and is reported beside AUROC, never instead of it. (A random-init net scores
  0.81 accuracy on the easy split; OOD accuracy stayed 0.51-0.57 across a run in
  which AUROC went 0.547 -> 0.804.)
- **Causal effects are reported as OUTCOMES, not displacements.** Raw logit delta
  is a displacement and inverts the true ordering: report **crossing rate**
  (fraction moved from refusal- to compliance-leaning) plus a **behavioural
  generation** number. Raw delta may appear as a secondary column.
- Every headline number carries a **bootstrapped 95% CI**; comparisons between
  models use **paired** bootstrap.
- Where a construct admits several operationalisations, report all and state
  disagreements rather than selecting.

## P5. Pooling and depth
- Report **both last-token and mean-pooled** readouts everywhere.
- **No depth claim may be made from a single pooling.** "Layer 4 never leaves
  chance" held for last-token and collapsed under mean pooling (0.530 -> 0.719,
  best layer L31 -> L16).

## P6. Prompt format
- One format per figure. Default: neutral scaffold `User: {prompt}\nAssistant:`
  applied identically to every checkpoint including base.
- The model's own chat template is a **separate axis**, never mixed into a
  scaffold curve. When comparing templated checkpoints, diff the template strings
  first — Think SFT/DPO/RLVR ship different ones.
- Any cross-format claim states which of ranking vs calibration it concerns.
  Ranking is largely format-invariant; calibration is not.

## P7. Tokenisation and readout position
- `MAX_LEN = 384`, **left** truncation whenever the readout is the last token.
- Log the fraction of prompts hitting `MAX_LEN` in every result file.
- Index the last **non-pad** token via the attention mask.
- Reject any layer whose mass-mean direction norm is ~0 (layer 0 is the embedding
  of an identical final token and deflated every layer-averaged cosine by 32/33).

## P8. Intervention machinery
- Ablation hooks cover `embed_tokens` **and** all decoder layers.
- Assert hook liveness (intervention must change logits) and per-layer variation
  (values must differ across layers) before trusting any sweep.
- Handle Olmo3 decoder layers returning a bare Tensor as well as a tuple.

## P9. Data hygiene
- Dedupe WildGuardMix on `prompt` (one row per response; 38,907 duplicates).
- Hash safetensors to detect duplicate upstream branches (RL-Zero-Code step_100
  and step_1000 are byte-identical).
- Pin exact HF revision strings; `main` may differ from the last step branch.

## P10. Execution
- Long jobs run in a **detached kernel-side thread** taking config as **thread
  arguments**, never reading notebook globals (re-running an owning cell kills
  the thread with `NameError`).
- **Launch cells must guard on live threads and pending work.** Re-running a
  worker cell re-runs its descendant launch cell and spawns a *duplicate* thread;
  two sweeps then race on the GPU. Config-as-arguments prevents the NameError
  death but not the duplication.
- Report a **noise floor** for any geometric quantity: two random unit vectors in
  R^4096 have |cos| ~ 0.0117, so drift below that is not meaningful however
  statistically resolvable it is.
- Idempotent and disk-backed: skip completed revisions, survive restarts.
- Purge each checkpoint's HF cache after use.
- Render and inspect every figure before use.
