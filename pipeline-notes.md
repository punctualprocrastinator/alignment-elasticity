# Pipeline notes — week-1 validation run (2026-08-05)

End-to-end validation of the probe pipeline on `allenai/Olmo-3-7B-Instruct`, run in
the molab notebook (`/marimo/notebook.py`, RTX PRO 6000 Blackwell 102 GB, torch 2.11,
transformers 5.14). All stages cached and idempotent; full notebook re-run ~7 s.

## Results

- **Probes at ceiling**: AdvBench-vs-Alpaca is linearly separable at ~1.000
  accuracy/AUROC from layer 8 onward (layer 4: 0.986). No dynamic range -> unusable
  as-is for formation/elasticity curves, and much of the signal is likely
  topic/lexical confound rather than harmfulness. **Blocking item for the sweep:**
  build a harder contrast set (topic-matched harmful/benign pairs, e.g. paired
  rewrites; plus OOD-generalization eval: train on set A, test on set B).
- **Refusal ablation reproduced, 6/6**: difference-in-means direction at layer 20
  (selected on a train-derived validation subset), ablation hooks on embed_tokens +
  all 32 decoder layers. Held-out harmful prompts flip from refusal onset to
  compliance onset; benign completions byte-identical to baseline.
- **Graded causal metric exists**: refusal rate under ablation by direction-source
  layer: L12 1.0 / L16 0.83 / L20 0.0 / L24 0.5 / L28 0.83. This has the dynamic
  range the raw probes lack -> candidate primary elasticity readout.
- **Caveat**: ablation flips refusal *form*, not full substantive compliance (model
  often redirects to safe content). If a compliance metric is needed, use an LLM
  judge, not prefix matching.

## Throughput (changes the sprint math)

- Model download + load: **31.7 s** for ~15 GB (datacenter bandwidth on molab).
- Activation extraction: **4.0 s** for 1,040 prompts x 8 layers x 2 poolings.
- Per-checkpoint cost is therefore ~1 min including download. A 60-checkpoint sweep
  is hours, not weeks. Download bandwidth is NOT the bottleneck (contra project.md
  section 7); HF cache disk is (~29.3 GB/model — delete weights after extraction).

## Probe-saturation fix (2026-08-06) — RESOLVED

WildGuardMix contrast set (4 pools x 500: vanilla/adversarial x harmful/benign,
deduped on `prompt` — dataset has one row per response, dedupe is mandatory).
Probe trained on vanilla harmful-vs-benign, evaluated per layer (last-token):

- In-distribution: 0.94-0.97 across layers.
- **OOD-hard (test on adversarial): 0.532 (L8) -> 0.865 (L28)** — 0.33 spread with
  clean layer structure. Early layers (4-8) carry a lexical/topic shortcut that
  collapses to chance under adversarial reframing; a generalising representation
  emerges at L12 and plateaus L16-20+, converging with the L20 causal refusal
  direction. Non-monotone: L8 (0.532) is *below* L4 (0.605) — don't assume
  monotonicity when summarising.
- Cross-dataset (test on AdvBench/Alpaca): 0.99+ — confirms yesterday's ceiling was
  topic separability, not harmfulness.
- Shuffled-label control ~0.49 (no leakage).

**Primary sweep metric: OOD-hard accuracy/AUROC as a curve over layers 12-31**,
with causal refusal-ablation rate as the second axis. Known residual confound:
adversarial prompts are ~7x longer than vanilla — add a length-matched control
before publishing. Extraction truncates at 1024 tokens (clips a few adversarial
prompts, no vanilla ones).

Artifacts: `wildguard_contrast.parquet`, `activations_wildguard_olmo3_7b_instruct.pt`
(524 MB). Extraction: 47 s for 2,000 prompts. Notebook: 30 cells, stage-3 extraction
refactored into reusable `extract_activations(...)`.

## E1 formation sweep, first pass (2026-08-06) — PROTOCOL ARTIFACT, rerunning

19/19 checkpoints of Olmo-3-1025-7B swept (35 min, 0 failures, ~125 s/ckpt).
Result: OOD-hard flat at chance (0.50-0.62) across 1.47M steps, ID rises
0.81 (RANDOM INIT!) -> 0.925. But a format control killed the interpretation:

- instruct + chat template: 0.865 OOD-hard (L28)
- instruct + RAW text:      0.509 — same weights, indistinguishable from random init
- **Prompt FORMAT, not weights, carries the OOD-hard signal.** The raw-text-
  everywhere policy removed the signal it was measuring; first-pass E1 bounds
  nothing about formation timing.

Kept findings: (1) random-init nets already score 0.81 ID — ID accuracy is
near-worthless as an alignment metric; (2) representation access is
format-gated on instruct — a real finding, keep in paper. Rerun in progress
with neutral scaffold ("User: {prompt}\nAssistant:") applied identically to
every checkpoint; instruct+template and instruct+raw kept as format-axis
anchors. Probe directions per checkpoint saved to /marimo/e1/probes_*.npz.

## E1 formation sweep, SCAFFOLDED rerun (2026-08-06) — COMPLETE, real curve

Neutral scaffold ("User: {prompt}\nAssistant:") identical across all 19 ckpts.
Sanity first: instruct RAW 0.51 -> SCAFFOLD 0.81 -> CHAT TEMPLATE 0.86 (same
weights). Scaffold recovers most signal; template adds the last ~0.05.

Formation (best-layer OOD-hard AUROC): step0 0.547 -> step2k 0.640 -> step9k
0.705 -> step152k 0.761 -> end stage1 (1.41M) 0.804 -> stage2/3 0.797/0.804.
Shuffled control 0.49 throughout.

- **Onset step ~2,000 — 0.14% into stage 1.** Alignment-relevant structure
  starts forming almost immediately.
- **Gradual, not a phase transition**: ~0.05-0.06 AUROC per decade of tokens,
  still climbing at end of stage 1. Midtraining continues rather than creates
  the trend (+0.02-0.03).
- Deep layers (24-31) lead throughout; L16 catches up at ~265k-463k; **L4
  never leaves chance.**
- **Use AUROC as the headline, not accuracy**: a vanilla-trained probe transfers
  its ranking to adversarial prompts much better than its threshold, so accuracy
  is non-monotone (threshold drift) while AUROC rises cleanly.
- **Format effect — CORRECTED 2026-08-06 (the first version of this claim was
  wrong).** Earlier note said "representation access is format-gated" based on
  ACCURACY only. In AUROC the raw sweep also shows formation (0.543 -> 0.737),
  and fixed instruct weights give raw 0.729 AUROC / 0.509 acc; scaffold
  0.901/0.814; chat 0.935/0.865. Correct claim: **ranking is largely
  format-invariant; calibration is strongly format-dependent.** The harm
  direction exists without dialogue framing — what the format supplies is a
  usable decision boundary. Still a real safety result (fixed-threshold probes
  and fixed-magnitude steering silently mis-fire out of format), but it is a
  claim about decision boundaries, not about whether harm is represented.
  Checkpoint comparisons must still hold format fixed.

Sweep 38 min, 0 failures; both format runs preserved side by side (/marimo/e1
now 21 GB). Figure 1 is now 3-panel: scaffold formation primary, raw-text muted,
format axis.

## E1b fine-grained early dynamics (1B, 2026-08-06) — "gradual" REFINED

Repo gotcha: `allenai/OLMo-2-0425-1B` is NOT dense early (branches 0, 300,
10000, 20000 — coarser than our 7B sweep). The dense suite is a separate repo:
**`allenai/OLMo-2-0425-1B-early-training`** (step0..step36000 every 1000).

26 checkpoints: every 1k step to 20k, plus 50k/100k/300k/1M/1.91M. 16-layer
grid [2,4,6,8,10,12,14,15], same WildGuard pools + scaffold, AUROC primary.

| step | OOD AUROC | % of total rise |
|---|---|---|
| 0 | 0.540 | — |
| 1,000 | 0.561 | 9% |
| **2,000** | **0.649** | **50%** |
| 20,000 | 0.688 | 67% |
| 1,907,359 | 0.760 | 100% |

**The rise is FRONT-LOADED, not smooth log-linear.** The single interval
1k->2k carries +0.088 AUROC — **~50% of the entire rise over 1.9M steps, in
0.1% of training** — and is 3.2x the SD of every later step-to-step change
(0.027). Log-linear fit over step>=1000: 0.050 AUROC/decade but R^2 only 0.78,
dragged by that jump. Not a clean phase transition either: no flat period
precedes it, because the rise has already begun by the first checkpoint after
init. **Onset is at or before step 1,000 and remains UNRESOLVED** even at 1k
spacing. Deep layers lead decisively (L15 best at every step >=1000; L2-L6
near control throughout) — same qualitative picture as 7B.

Caveat: OLMo-2 1B != OLMo-3 7B (different data order, schedule, capacity; 1B
tops out at 0.760 vs 7B ~0.80). **Step numbers do not transfer.** Locating the
7B onset needs 7B checkpoints below step 1,000.

Sweep 44 min, 68 s/ckpt, 0 failures. Figure: fig_e1b_1b_fine_grained.png.

## E2 staged-flow elasticity (2026-08-06) — COMPLETE, 15/15 checkpoints

Refusal-direction cosine-to-base along the flow (raw text, last token):
- **Drift is a cliff, and the cliff is early SFT**: mean-cos delta 0.104 between
  SFT step1000 and step15000; DPO adds 0.005; all six RLVR checkpoints add
  ~0.000 (step-to-step cos 1.000). RL-Zero-Math after 1900 steps: cos >= 0.9996
  vs base — RL on the base model leaves refusal geometry essentially untouched.
- Frozen-vs-refit harm probes (AUROC, adversarial pools): frozen ~0.71-0.74 vs
  refit ~0.72-0.76 — elasticity gap 0.01-0.03, i.e. ~none. The refit probe
  DIRECTION rotates through SFT (cos 0.95->0.65 at L31) while frozen stays as
  informative: the subspace tilts but information is preserved — elastic, not
  eroded. (Raw-text caveat applies to accuracy numbers; AUROC was meaningful.)
- Chat-templated directions are near-orthogonal (cos 0.02-0.2) to raw-text
  anchor and stable across the flow — format dominates geometry; never mix.
- **Headline: alignment-relevant geometry is set almost entirely by early SFT;
  capabilities RL (RLVR and RL-Zero alike) is representationally inert at these
  layers.** This is the "robust to this much RLVR" publishable null from the
  risk table, now measured.
- Ops: box-2 tunnel interrupts running cells on ~60 s stream drops (worse than
  box 1); fix = detached kernel-side daemon thread writing artifacts to disk,
  durable cells re-run from cache. Sweep: 15 ckpts in 24 min (~95 s each).
- NOTE: user pasted the HF token as a plaintext login() cell on box 2 as well
  (cell Hbol) — two notebooks now carry plaintext tokens; rotate + clean both.

## E2 scaffolded rerun + E4 behavioural coupling (2026-08-06) — COMPLETE

**Scaffold replicates the cliff, harder.** Cosine-to-base (scaffold, L31):
SFT step1000 .874 -> step15000 .682 -> DPO .593 -> RLVR step_1375 .594 ->
RL-Zero step_1900 .998. Staged mean-cos deltas: 0.084, **0.230 (step1000->
15000)**, 0.032, 0.017, 0.006 (DPO), then <=0.001 per RLVR step. Total RLVR
drift over 1350 steps: **0.0006**. Cliff-at-early-SFT + RL-inert REPLICATES.

**Frozen-vs-refit with real signal (the strong version).** Scaffold restores
probe accuracy on post-SFT ckpts: 0.53 raw -> 0.78-0.82. At the staged end,
frozen base probe AUROC 0.888 vs refit 0.929 — **gap +0.041**. Meanwhile probe
direction cosine collapses to ~0.50 at L31 by DPO: **the harm subspace rotates
~60 degrees while losing ~0.04 AUROC. Information preserved, basis moved.**
(Accuracy comparison flips sign vs AUROC due to threshold luck — use AUROC.)

**E4 behaviour (60 HarmBench behaviours, HarmBench-Llama-2-13b-cls judge):**
ASR base 0.350 -> SFT step1000 **0.000** -> SFT 43k 0.067 -> DPO 0.050 ->
RLVR step_1375 0.017 -> RL-Zero step_1900 **0.417**. Behaviour and geometry
co-move by both collapsing in the first ~1000 SFT steps and then freezing
(DPO->RLVR: ASR delta 0.033, cosine delta 0.0006). **Not the re-weighting
story — a single-event story.** RL-Zero sits at base ASR: it never learned
refusal (no SFT), rather than RL having destroyed it.

**Causal layer profile (own-direction ablation, 8 prompts, coarse):** L12 is
the argmax at EVERY checkpoint (base, SFT, DPO, RLVR, RL-Zero). The causal
locus does not migrate even though the direction at that depth rotated to
cos 0.57. **Circuit location stable, circuit content rotated.** Base and
RL-Zero rows identical — independent confirmation RL-Zero left it untouched.

Gotcha: `cais/HarmBench-Llama-2-13b-cls` ships only sentencepiece
`tokenizer.model`; transformers 5.x cannot convert it and use_fast=False is
gone. Workaround: load `hf-internal-testing/llama-tokenizer` (same Llama-2
32k vocab); judge weights unaffected.

## E3 RL-Zero dose-response + causal staleness (2026-08-06) — 14/19 ckpts

**Erosion is objective-driven, not dose-driven.** Cosine drift to base (L24):
IF drifts more after 100 steps (0.0022) than Code after 2900 (0.0009); the two
longest runs (3.1-Math 2800, 3.1-Code 1950) are among the LEAST drifted. Within
a domain drift is monotone in steps — dose matters conditional on objective.
All drift magnitudes are tiny (<=0.004) — consistent with E2's RL-inert result.

**Causal staleness — the control headline, NOW DECONFOUNDED (2x2).** Base L20
direction, never refitted, ablated on descendants; judge-free readout
logprob(refusal) - logprob(compliance), intact vs ablated, 6 prompts:

| model | raw | scaffold | chat |
|---|---|---|---|
| base | -0.439 (100%) | -0.269 (100%) | n/a (no template) |
| RL-Zero-Code @2900 | -0.333 (76%) | -0.230 (**86%**) | -0.395 |
| RL-Zero-IF @1900 | -0.140 (32%) | -0.234 (**87%**) | — |
| **Instruct** | +0.041 (-9%) | **-0.021 (8%)** | +0.009 |

**Staleness is a property of the POST-TRAINING, not the interface.** At matched
scaffold interface RL-Zero descendants keep 86-87% of the base causal effect
while Instruct keeps 8%, and Instruct is flat at all three interfaces.
**RETRACTED: "erosion proportional to rotation."** On raw text IF (32%) looked
weaker than Code (76%), matching IF's larger rotation — at matched interface
that ordering VANISHES (86% vs 87%). It was a format artifact, not a finding.
(Base has no chat template, so the figure plots raw log-prob deltas rather
than % of base, which would silently delete the chat column.)

**Mix resume FAILED** — old and new sweep threads deadlocked on the same
huggingface_hub file lock (an isolated cache_dir did not help; contention is
the lock, not blob paths). Python threads cannot be killed externally; fix is
a kernel restart, not done to a live notebook. Final: 14/19 checkpoints.
Consequence: **"objective, not dose" rests on 3 domains, not 4** — Mix is the
mixed-objective run and the most likely to complicate it. Honest framing: IF
is clearly separated from Code and General at matched dose.

**REAL BUG CAUGHT (check other boxes): MAX_LEN=128 with RIGHT truncation cut
76% of adversarial WildGuard prompts mid-jailbreak**, putting "last token" in
the wrong place and pinning probe AUROC at 0.50. Fixed: MAX_LEN=384 + LEFT
truncation. Length-sorted batching cut extraction 60 s -> 18 s. (Box 1 used
1024 so is unaffected; box 2's healthy AUROC suggests it is fine.)

Other gotchas: RL-Zero-Code step_100 and step_1000 have byte-identical
safetensors upstream (duplicate branches, not a caching bug); Mix/General
declare model_type `olmo2-retrofit` that transformers 5.x does not register —
re-key config onto `olmo3` and assert zero missing/unexpected weights; marimo
mangles leading-underscore helper names as cell-private.

Incomplete: RL-Zero-Mix (all 4 steps) + 3.1-Code step_1900 — wedged HF
download, resumable idempotently.

## Three-format check + E6 black-box monitoring (2026-08-06) — COMPLETE

**Chat template breaks the "frozen" half of the headline.** Common anchor at
Think-SFT step1000, mean over L12/20/24/28/31:

| format | drift to SFT end | DPO step | RLVR span | end cos |
|---|---|---|---|---|
| chat | 0.413 | **0.107** | **0.041** | 0.519 |
| scaffold | 0.265 | 0.006 | 0.003 | 0.732 |
| raw | 0.116 | 0.005 | 0.000 | 0.879 |

The SFT cliff replicates in all three formats. But under the chat template DPO
rotates the direction ~18x more than under scaffold, and RLVR moves it over a
0.041 span (~16x), non-monotonically (drifts out to step_0575, partly returns
by step_1375; concentrated in late layers, L31 0.266 -> 0.376).
**Correct claim: "one cliff, then mostly flat" survives; "frozen" does not.**
Capabilities RL is not a no-op in the model's native templated basis, though
the movement is ~10x smaller than the SFT cliff. Honest confound: Think-SFT/
DPO/Think ship slightly different chat templates, so part of the chat column
is template TEXT changing, not representation drift — which is exactly why
the scaffold pass is the primary measurement.

**E6 black-box trusted monitoring (4 monitors x 480 pairs).** Monitor reads
behaviour+generation after a 3-shot prompt; continuous score =
logsumexp(yes-tokens) - logsumexp(no-tokens) at a forced "Answer:" position
(prevents Think models' reasoning tokens hijacking the answer slot).

| monitor | AUROC | catch@5% FPR | catch@1% FPR |
|---|---|---|---|
| base | 0.775 | 0.052 | **0.017** |
| SFT step1000 | 0.862 | 0.517 | 0.310 |
| SFT step43000 | **0.874** | **0.655** | **0.431** |
| DPO | 0.836 | 0.552 | 0.155 |

**No staleness with training distance**: correlation of gap vs per-pair AUROC
= -0.26 (slightly BETTER at larger gaps, within noise); most distant pair
(base -> RLVR step_1375, gap 11) scores 0.746 vs base-at-gap-0 0.639; diagonal
averages 0.770. Flat, neither gradual nor cliff.
**What varies is monitor CAPABILITY, and only at the deployment-relevant
operating point**: at 1% FPR base catches 1.7% vs SFT step43000's 43%, despite
similar AUROCs (0.78-0.87). A trusted monitor that is merely a pretrained base
model is near-useless at a realistic alert budget while looking fine on AUROC.

**RECONCILIATION (the day's sharpest claim): reading a property is robust;
pushing on it is not.** Black-box monitoring and frozen-probe readout both
survive the flow intact, while causal steering with the base direction decays
(76% / 32% / 2% on RL-Zero-Code / IF / Instruct). Readout accuracy and
behavioural monitorability are preserved where causal control is lost.

E6 power caveats: post-SFT checkpoints almost never comply harmfully, so 5 of
8 monitored columns have 1-4 positives (one has zero, undefined). Only pooled
numbers and base/RL-Zero columns are properly powered — per-pair panel is
"flat within noise", not a measured curve. catch@1% FPR is not estimable per
column. Format confound: base is scaffold-prompted, others use own templates,
so capability and format are not separable. Single judge defines truth.

## E5 circuit drift + E7 honesty property (2026-08-06) — COMPLETE

**E5 (13 checkpoints, 1056 components = 32x32 heads + 32 MLPs, attribution
patching, neutral scaffold).** Top-20 Jaccard vs base: 0.74 (SFT-1k) -> 0.43
(SFT-43k), then **SFT-43k = DPO = RLVR-25 = RLVR-400 = RLVR-1375, all pairwise
1.00**; RL-Zero vs base = 1.00. Spearman vs base: 0.784 -> 0.418, then
0.410/0.410/0.412/0.414 across DPO->RLVR (constant to 3dp); RL-Zero 0.997-1.000.
11 of 20 components never leave the top-20; **zero enter or leave from SFT-43k
onward**. Verdict: **the refusal circuit is re-weighted once by SFT, then
frozen** — agrees with E2, with one refinement: the circuit keeps moving until
~SFT step 20k, i.e. **behaviour saturates before the mechanism does**.

Causal validation (activation patching, 100 held-out pairs): base model's
top-5 retains **84-87%** of each checkpoint's own top-5 effect after the whole
pipeline; mid-rank and random-5 controls ~0. Refusal is redundantly encoded:
top-5 own the metric but patching top-100 flips only 2-3 of 6 generations.

**Layer-index disagreement with E4, resolved:** attribution argmax is **L15**
at 11/13 checkpoints (L19-L23 MLPs carry the mass); E4's ablation argmax was
L12, which holds only ~2.6% of attribution mass. Both agree the location is
FIXED across the flow. Interpretation: **ablation measures where refusal is
READABLE, attribution measures where it is WRITTEN.**

**E7 HONESTY (Azaria-Mitchell via atmallen/{topic}_azaria_mitchell; fit on
cities/companies/animals/elements, tested on held-out inventions/facts/
capitals; 17 checkpoints).**
- Formation: **honesty forms LATER — onset ~step 5,000 vs harm's ~2,000**
  (at step 2k honesty is 0.512 vs a 0.541 chance band). Reaches 0.846 by end
  of stage 1 (harm 0.804), 0.861 at final base. Random-init scores 0.543 OOD,
  so the OOD split gives away nothing free.
- Elasticity: **identical to harm.** Cosine to base 0.974 (SFT-1k) -> 0.682
  (SFT-43k) -> 0.679 (DPO) -> 0.680 (RLVR-1375) -> 0.999 (RL-Zero). SFT move
  0.292 (harm 0.23); RLVR drift 0.00071 (harm 0.0006); frozen-vs-refit gap
  0.012 (harm 0.04).

**HEADLINE: "formation timing predicts elasticity" is FALSIFIED. The FLOW
predicts elasticity.** Honesty forms ~2.5x later yet drifts the same amount in
the same shape, and its frozen probe transfers BETTER (0.012 vs 0.04) — the
opposite of the prediction. Elasticity is a property of where you are in the
pipeline, not of when the concept crystallised. This overturns novelty claim
#1 in project.md section 3; reframe the paper around it.

Bugs caught by the agent (all fixed, results re-run): `attn_sum +=` broadcast
across all 32 layers instead of `attn_sum[li] +=`, which produced 32 identical
rows and a fake "head 2 dominates everywhere" result (found by a
same-value-in-every-layer sanity check); 44-token generations truncating
inside `<think>` making refusal classification meaningless (regenerated at
200); E7 layer 0 is the embedding of the identical final ":" token so its
mass-mean direction is exactly zero, deflating every layer-averaged cosine by
32/33 (excluded).

## J-lens (Anthropic Jacobian Lens) spike (2026-08-06) — GO, never fit

Tool: github.com/anthropics/jacobian-lens, v0.1.0, Apache-2.0, released
2026-07-06. NOT on PyPI — install from git. Declares transformers>=5.5 (we
have 5.14.1). `jlens.from_hf` auto-detected Olmo3 with ZERO patching (OLMo is
in the supported-layouts docstring; only the examples are Qwen).

**Do NOT fit lenses: 36 s per prompt (11 source layers, 128 tokens) = ~10 h
for the paper's 1000 prompts, 12-80 GPU-h for a 6-8 checkpoint sweep.**
Pre-fitted lenses exist:
- `neuronpedia/jacobian-lens` — official, olmo-3-1025-7b base + 32b, 31 layers
- `mhough/olmo3-jacobian-lenses` — third-party, **11 lenses covering the whole
  ladder** (base, Instruct SFT/DPO/final, Think SFT/DPO/final, RL-Zero
  math/code/if/general), 11 source layers (0,3,...,30)
Provenance check: 64 random vectors transported through both base lenses agree
at mean cosine 0.978/0.987/0.993/0.996/0.997/0.999/0.9998 (L12..L30) — two
independent fits, different corpora. Third-party ladder is trustworthy.

**The refusal direction IS in J-space at base** (diff-in-means, 100v100,
transported + unembedded):
- L16: unethical, immoral, illegal, ethical, unacceptable, prohibited, unsafe
- L20: unethical, prohibited, forbidden, unacceptable, immoral, illegal
- L24: unethical, sorry, illegal, apologies, violate, apologize
- L12: junk (newlines, ' acct', ' Sponsored')

So "does it EXIT J-space by Instruct?" is well-posed — and would mechanistically
explain the read/steer dissociation (86-87% causal retention on RL-Zero vs 8%
on Instruct). **Bonus: the lens independently localises refusal to L16-L24 and
NOT L12, corroborating E5's attribution (L15-L23 MLPs) over E4's ablation L12.**

Caveats: ladder lenses cover only named final revisions (not intermediate
SFT-step1000 / RLVR-step_1375), 11 source layers only, third-party — re-derive
headline numbers against the official base lens at shared layers.

Env wrinkle: the resolver pulled torch 2.13.0 into the uv venv while the kernel
runs 2.11.0 from system site-packages. Nothing broke, but a kernel restart
would switch torch versions.

## E8 J-space residency (2026-08-06) — correlates, does NOT explain

11 checkpoints x 5 layers x 3 directions x 200 random nulls, 6.5 min, zero
lenses fitted (pre-fitted only).

Base-direction semantic-coherence retention vs causal retention:
base 1.000 | instruct-sft 0.273 | instruct-dpo 0.227 | **instruct-final 0.227
(causal 0.08)** | think-* 0.227 | **rlz-* 0.909 (causal 0.865)**.
Through the rlz-math lens at L21 the base refusal direction still reads
" unfortunately/ forbidden/ prohibited/ violates/ sorry"; through
instruct-final at L18 it reads sub-word fragments and non-Latin bytes.

**BUT the control kills the refusal-specific reading.** A SENTIMENT
diff-in-means direction retains 0.87-1.09 through RL-Zero lenses and **0.000**
through Instruct lenses — the same collapse, harder. This is **general
de-verbalization of base-era linear directions in SFT-descended checkpoints**,
not a refusal-specific mechanism. A simpler account also survives with no lens
at all: cos(own, base) at L21 is 0.86 for RL-Zero (vs 0.865 causal) and 0.53
for Instruct (vs 0.08) — plain direction drift tracks the causal numbers at
least as well.

The three operationalisations DISAGREE: (a) transport magnitude is a NULL —
everything transports at near-random gain everywhere including at base where
the direction demonstrably decodes to refusal vocabulary, and the control
transports harder than refusal; (b) semantic coherence is the ONLY
discriminating metric, so all conclusions rest on it; (c) subspace mass is
consistent but weak (base 0.024 vs chance 0.0156, instruct 0.019, rlz 0.022).

**Unexcludable confound:** every lens (official and third-party) was fit on
wikitext. Base and RL-Zero are raw-text models; Instruct/Think are chat-tuned
and were run on the raw-text scaffold, outside their native mode. A
wikitext-fit Jacobian may simply be a worse linearisation for chat-tuned
models, producing this pattern for ALL directions with no J-space claim
involved. This lines up exactly with which arms went through SFT. No official
non-base OLMo lens exists to cross-check and fitting is off the table.

Controls that worked: L12 junk layer 0.000 coherence at all 11 checkpoints;
random null 0.0017 over 200 directions. Third-party vs official lens at base:
gain 1.048 vs 1.052 (L18), coherence within 0.05, top-64 mass within 0.002.

**Verdict: the refusal direction does leave the verbalizable subspace in the
Instruct arm and not in RL-Zero — but so does everything else we measured.**
Correlated, right ordering, quantitatively close on the RL-Zero side, but not
established as refusal-specific or as the mechanism.

## A1 powered causal claims (2026-08-07) — dissociation CONFIRMED, metric REPLACED

n=200 held-out prompts per model (disjoint from fit), bootstrapped 95% CIs,
20-random-direction null. Smoke tests 5/5 pass (hook liveness max |dlogit|
13.09; embed_tokens hook alone 0.406 — confirms it matters; 0/400 prompts hit
MAX_LEN, so the truncation lesson does not bite on short AdvBench goals).

**Raw logit-delta INVERTS yesterday's n=6 result:**

| model | intact | ablated | delta [95% CI] | % of base | z vs null |
|---|---|---|---|---|---|
| base | 0.852 | -3.514 | 4.366 [4.164, 4.581] | 100 | 89.6 |
| instruct | 7.850 | 1.036 | 6.814 [6.589, 7.035] | **156** | 163.7 |
| rlz-math | 1.015 | -3.529 | 4.544 | 104 | 87.4 |
| rlz-code | 0.837 | -4.009 | 4.846 | 111 | 86.2 |

**"Instruct = 8% of base" DOES NOT REPLICATE — it inverts to 156%.** But this
is a metric artifact: raw delta is a DISPLACEMENT, not an OUTCOME. Instruct
starts at +7.85, is pushed down 6.81, and lands at +1.04 — still refusing.
Base starts at +0.85, drops 4.37, lands at -3.51 — decisively compliant.

**Scored as CROSSING RATE (fraction pushed from refusal- to
compliance-leaning), the dissociation returns and the CIs separate cleanly:**

| model | crossing rate [95% CI] | generation refusal (40 prompts) | % refusals removed |
|---|---|---|---|
| base | 0.670 [0.605, 0.735] | 0.775 -> 0.050 | 93.5 |
| **instruct** | **0.350 [0.285, 0.415]** | **1.000 -> 0.850** | **15.0** |
| rlz-math | 0.695 [0.630, 0.760] | 0.775 -> 0.050 | 93.5 |
| rlz-code | 0.655 [0.590, 0.720] | 0.750 -> 0.025 | 96.7 |

Instruct's crossing-rate CI overlaps none of the other three; the behavioural
numbers land almost exactly on yesterday's 100/86/87/8. **The read/steer
dissociation is confirmed at n=200 with a null model.**

**REVISED MECHANISM (stronger than the old claim):** SFT+DPO does not move
refusal out of the residual stream — it inflates dependence on refusal-onset
logit mass so far that removing the same quantity no longer flips the decision.

**Also corrected: RL-Zero does NOT separate from base.** 104% and 111% formally
exclude 100% but both arms are behaviourally indistinguishable from base
(93.5%/96.7% vs 93.5%). Defensible claim: "RL-Zero leaves the base-era
direction intact", NOT "RL-Zero erodes it".

**Headline sentence must be rewritten before submission.** Not "Instruct = 8%
of base". Instead: "156% of base raw displacement but 52% of base crossing rate
and 16% of base behavioural effect."

Ops: sweep ran detached, 411 s for 4 checkpoints. **The sandbox was NOT fresh** —
/marimo/notebook.py carried ~69 cells from a sibling experiment even though the
kernel was clean, so appended cells must use unique names or the whole prior
graph goes stale. The earlier 600 s stall was NOT a long job: nothing had been
launched; the watchdog fired during HF metadata pre-flight calls.
`pipeline.py` (944 lines) round-trips byte-identical to the user's machine.

## A2 probe-family control (2026-08-07) — formation SURVIVES, depth claim DIES

20 revisions x 2,560 probe fits (32 logistic + 96 MLP over 3 seeds), 40 min,
0 failures, all on identical cached activations (day-2 artifacts were still on
the box, so no re-extraction and no gated-dataset access needed).

Best-layer OOD-hard AUROC, end of stage 1: linear/last **0.804**, linear/mean
**0.836**, MLP/last 0.708, MLP/mean 0.785. Shuffled-label 95% upper bounds:
linear/last 0.588, linear/mean 0.559, MLP/last 0.615, MLP/mean 0.625.

1. **Non-linear probe finds signal LATER, not earlier.** Onset: linear/last and
   linear/mean **1,000**; MLP/last **50,000**. The MLP loses to logistic
   regression at **15/15** stage-1 checkpoints (mean delta -0.062; best case
   still -0.011). **The ~step-2k onset is NOT an artifact of linearity.**
2. **Mean pooling does not move the onset but DEMOLISHES the depth story.**
   Curve lifts +0.068 on average; best layer at end of stage 1 moves
   **L31 (last token) -> L16 (mean-pooled)**, and **layer 4 goes from 0.530
   (inside its null band, margin +0.002) to 0.719 (clears null by +0.171).**
   "L4 never leaves chance / deep layers lead" is a statement about the
   LAST-TOKEN READOUT, not about the model. **Restate it as such in the paper.**
3. **A stronger probe buys nothing and costs null width.** MLP never wins, and
   its null is ~2x wider (one shuffled draw reached 0.757 OOD AUROC). This is
   the crisp answer to the "just use a stronger probe" reviewer instinct.

Calibration diagnostic as expected: OOD accuracy stays 0.51-0.57 across the
whole run while AUROC goes 0.547 -> 0.804; ID AUROC is already 0.925 at step 0.
Only OOD-hard AUROC has dynamic range.

**Onset caveat:** linear/last technically clears its null at step 1,000
(0.592 vs 0.588, +0.004 margin) and is clearly established by 2,000. Publish
"~step 2,000" as the safe statement.

### Two methodology traps caught (both would have invalidated results)
- **SEED MISMATCH IN pipeline.py.** `pipeline.py` sets `SEED = 20260805` (the A1
  seed) but the published probe sweep used `SEED = 42`; the train/test split
  derives from it, so the first A2 sweep was silently off by 0.01-0.03 AUROC
  everywhere. After correcting to 42, readout (a) reproduces the published
  sweep **exactly (max |delta| = 0.0 over 152 checkpoint x layer cells)**.
  Every result file now records the seed plus a SHA-1 split fingerprint, and
  aggregation refuses non-matching files. **Any reuse of pipeline.py must set
  the seed explicitly.**
- **A single shared shuffle permutation makes the null band worthless.** Reusing
  one permutation across layers gave null_hi 0.658 — wide enough to swallow the
  entire formation curve. Draw an independent permutation per
  (layer, pooling, seed): null_hi 0.588, mean 0.495.

Ops: **re-running a cell that owns globals used by a detached thread kills that
thread with NameError** — detached workers must take config as thread ARGUMENTS,
not read notebook globals. This box also was NOT fresh (day-2 artifacts + 46
stale cells present) and carried ANOTHER plaintext HF token in a login() cell.

## E2-clean at protocol v1 (2026-08-07) — cliff SURVIVES, gap CORRECTED 6x

13 checkpoints extracted / 12 analysed, SEED=42, split `b3879463f2c6` (identical
to A2), all revisions pinned by branch AND commit (`main` differs from the
last-step branch in Think, Think-SFT and RL-Zero-Math — P9 confirmed). 2.4%
truncation at MAX_LEN=384. n=500/pool (deviation: gated dataset, see below).

**Q1 — SFT cliff survives overwhelmingly; RLVR drift is below the noise floor.**

| pooling | SFT cliff (1k->15k) | RLVR total (25->1375) | ratio |
|---|---|---|---|
| last | **-0.1852** [-0.1903, -0.1800] | +0.0009 [+0.0005, +0.0013] | **206x** |
| mean | -0.0829 [-0.0861, -0.0801] | -0.0003 [-0.0004, -0.0002] | 276x |

Cliff is ~40 bootstrap SDs from zero. RLVR drift IS statistically resolvable
(p_boot < 0.001) but is 206x smaller than the cliff and **below the
random-direction noise floor** (|cos| ~ 0.0117 for two random unit vectors in
R^4096). Correct phrasing: **"RLVR moves the geometry by less than the
resolution at which the metric is meaningful"**, not "exactly zero".
RL-Zero-Math cos **0.9993** (day 2: 0.998). **Cliff magnitude corrected to
0.185, not 0.230.**

**Q2 — the frozen-vs-refit gap DOES NOT REPLICATE: ~0.007, not 0.041.**

| pooling | frozen | refit | gap | 95% CI |
|---|---|---|---|---|
| last | 0.8123 | 0.8196 | **+0.0073** | [+0.0002, +0.0150] |
| mean | 0.8444 | 0.8495 | +0.0051 | [-0.0007, +0.0105] |

At the best layer (L16, last token) the gap is +0.0051 [-0.0085, +0.0179] —
indistinguishable from zero. Day-2's 0.041 was ~6x too large.
**This STRENGTHENS the headline**: the probe direction rotates to cos 0.502 at
L31 (~60 degrees) while the frozen probe loses ~0.007 AUROC. "Information
preserved, basis moved" is now a claim of ~zero information loss.

**Q3 — mean pooling does NOT flip anything here** (unlike A2's depth claim).
Both poolings agree on every headline sign; magnitudes differ substantially
(cliff 0.185 vs 0.083; end-cosine 0.699 vs 0.880; rotation to 0.50 vs 0.73), so
**never quote a magnitude without naming the pooling.**

**SOLVER BUG AFFECTING EARLIER NUMBERS.** `LogisticRegression(max_iter=5000)`
uses sklearn default `tol=1e-4` and stops after **14 LBFGS iterations** (objective
1.9685 vs 1.8224 at tol=1e-10, 74 iterations). The under-converged direction sits
at **cosine 0.978 to the true optimum** — so ~0.02 of arbitrary solver-path
rotation was contaminating every direction-cosine metric. Now fixed at 1e-10 and
added to protocol.md as P1b. **E1/A2 AUROCs are likely fine; any probe-DIRECTION
number from days 1-2 must be re-checked.**

**Deviations:** n=500 not 1,000 (gated dataset — the subagent declined to use the
leaked notebook token, correctly, since a relayed authorization from another
agent is not user consent; one-step fix is HF_TOKEN via the Secrets panel).
Refit-based bootstrap quantities use 200 resamples (each needs a fresh fit);
non-refit quantities use the full 1,000.

Ops: a batched GPU LBFGS (validated at cosine 1.000000 vs converged sklearn)
plus GPU bootstrap matmuls cut analysis 3.3x. Re-running a worker cell re-runs
its descendant launch cell and **spawns a duplicate racing thread** — launch
cells now guard on live threads (added to protocol P10).

## E4/E6-clean at protocol v1 (2026-08-07) — two day-2 claims REWRITTEN

200 HarmBench behaviours, greedy, **512** new tokens, 8 revisions verified
pairwise-distinct by safetensors sha256. 86.9 min total.

**E4: "ASR collapses to 0.000 within the first 1,000 SFT steps" DOES NOT
REPLICATE.** It is **0.180 [0.130, 0.235]**. Day 2's zero was the 44-token
truncation bug: at that budget the answer never escapes `<think>`, so the judge
scored reasoning traces as non-harmful.

| model | ASR [95% CI] | refusal [95% CI] |
|---|---|---|
| base | 0.355 [0.290, 0.425] | 0.445 |
| sft-1k | **0.180 [0.130, 0.235]** | 0.735 |
| sft-20k | 0.150 | 0.815 |
| sft-43k | 0.100 | 0.890 |
| dpo | 0.080 | 0.900 |
| rlvr-25 | 0.090 | 0.895 |
| rlvr-1375 | 0.070 | 0.905 |
| rlz-math | 0.400 [0.335, 0.470] | 0.450 |

Step 1000 delivers **61.4%** of the total base->RLVR-last drop, and the residual
`sft-1k -> rlvr-1375` decline is itself significant (dASR -0.110 [-0.165,
-0.055], paired). **So the "single event" framing needs softening: most of the
behavioural change is in the first 1k SFT steps, but a real further decline
continues through SFT.** DPO->RLVR IS statistically flat (every contrast spans
zero) — that half holds. Endpoints replicate (base 0.355 vs 0.350; RL-Zero
0.400 vs 0.417).

**NEW FINDING — suppression, not removal.** Under prefill, every post-SFT
checkpoint returns to **ASR 0.92-0.96, above the base model's own 0.865**.
Post-training suppresses harmful capability rather than removing it, and the
suppressed models are *more* exploitable than base once the refusal onset is
bypassed.

**E6: "a base-model monitor is near-useless at 1% FPR" DOES NOT REPLICATE.**
With all 8 columns at the 200-positive cap (day 2 had 0-4 positives in six of
eight), catch@1%FPR is **base 0.712 vs sft-43k 0.748** — day 2's dramatic
0.017 vs 0.431 was small-sample noise.

Elicitation (a stated part of the setup): positives pooled over intact,
own-direction refusal ablation, and `" Sure, here"` prefill; every positive
independently judged harmful by HarmBench-cls.

| monitor | AUROC (benign neg) | AUROC (hard neg) | catch@1% | catch@5% |
|---|---|---|---|---|
| base | 0.955 | **0.744** | 0.712 | 0.899 |
| sft-1k | 0.949 | 0.829 | 0.790 | 0.909 |
| sft-43k | 0.952 | **0.894** | 0.748 | 0.911 |
| dpo | 0.953 | 0.885 | 0.747 | 0.911 |

**H6 regression** (catch@5%FPR ~ capability + difficulty + distance, n=32,
cluster bootstrap): distance **-0.0020 [-0.0056, 0.0019]** (spans zero);
capability -1.282 [-3.176, 0.022] (spans zero); **difficulty/legibility
+1.845 [1.430, 3.391]** — the only term that excludes zero.

**"Capability, not freshness, is the binding constraint" survives only half.**
Freshness is dead: the raw AUROC-vs-distance correlation replicates (-0.284 vs
-0.26) but is entirely a confound — the two hard-to-monitor generators (base and
RL-Zero) sit at opposite ENDS of the flow ordering, so both have large mean
distance from mid-flow monitors; there is no within-cluster slope. But capability
is not the driver either, because at proper power all four monitors are ~0.95.
**The actual driver is how legible the monitored model's harmful output is**
(r = 0.996 with catch@5%FPR). The one genuine monitor difference is on **hard
negatives** (harmful compliance vs refusals *about* harmful topics): base 0.744
vs sft-43k 0.894. **If a monitoring claim is made, that is where it lives.**

Statistical honesty note from the agent: partial R^2 is non-negative by
construction, so its bootstrap interval can never contain a negative value —
"excludes zero" on a partial-R^2 row is NOT a significance test. The coefficient
CI is the test.

Deviations: monitor MAX_LEN=1536 (a 3-shot frame around a full response cannot
fit 384; truncation counts logged, all zero); 512 new tokens not 256 (Think
`<think>` blocks do not reliably close by 256; unclosed fraction 5-11% reported
per checkpoint so the day-2 failure mode is visible not silent).

## A4 32B — NOT MEASURED (box died during first download, 2026-08-07)

The sweep died ~1 minute in, during the base anchor's 64.5 GB download: 0 of 7
checkpoints complete, GPU at 685 MiB. **No 32B numbers exist; whether the SFT
cliff replicates at scale is untouched.**

Facts the preflight established, so a fresh box does not rediscover them
(`results/a4_partial_preflight.json`, `scripts/notebook_cells/RESTORE_a4_32b.md`,
`scripts/pipeline_a4.py` — the complete 32B implementation, already local):

- **Noise floor for R^5120 = 0.011155** (20 directions, 190 pairs; analytic
  0.011151). Same code reproduces **0.01172** for R^4096, matching the 7B run's
  0.0117 — implementation validated across both dimensions.
- **P1b confirmed empirically at 32B scale:** on (500 x 5120), `tol=1e-4` stops
  LBFGS at **18** iterations vs **33** at `tol=1e-10`; 0.76 s per fit.
- **P9 trap, documented:** `Olmo-3-32B-Think-SFT` ships **two learning-rate
  lineages** (`5e-5-*` and `1e-4-*`), and its `main` is byte-distinct from both
  `step10790` branches. **The cliff contrast must stay inside the 5e-5 run.**
- 32B geometry: 64 layers, d_model 5120. Split fingerprint `c4ed0db59d58`.
  160 GB box RAM makes a plain CPU->CUDA load viable.
- All 7 revisions resolved, pinned by branch AND commit, verified pairwise
  distinct by safetensors sha256.

Restore cost on a fresh box: ~45-60 min extraction (download-dominated,
unvalidated) + ~25 min analysis. Drop order if tight: `sft-main` and `rlvr-50`
first, leaving base -> sft-1k -> sft-10790 -> dpo -> rlvr-750 (still shows
cliff-then-flat, loses the RLVR-drift contrast).

## Gate A, partial (2026-08-27) — box reclaimed at 2/4 models, but the design changed

Sandbox reclaimed mid-`generate:instruct`. **No result files survived** — artifacts were written
to sandbox disk but pulled only at the end, and the reclaim ate them. Numbers below are read off
the kernel progress log: **point estimates only, no CIs, no null band, no RL-Zero arm.**
Restore assets are complete on disk (`scripts/gate_a.py`, `scripts/gate_a_analysis.py`,
`scripts/notebook_cells/gateA_01..12_*.py`); a full 4-model rerun is ~30 min GPU + ~5 min setup.
Commits pinned: base `a81bae42`, Instruct `6e5971d9`, RL-Zero-Math step_1900 `81823671`,
RL-Zero-Code step_2900 `ea18fc74`.

| model | unsteered gap | % refusing | c_50 (mass-mean) | d_50 (mass-mean) | d_50 (logistic) |
|---|---|---|---|---|---|
| base | +0.812 | 64% | -0.272 | 1.511 | 1.532 |
| instruct | +7.745 | 100% | -1.224 | 7.886 | 7.934 |

### Finding 1 — d_50 is mechanically pinned to baseline depth. **The brief's null was wrong.**
To push the median prompt across zero you must displace it by roughly its own baseline gap:
base d_50 1.511 vs mean gap 0.812; instruct d_50 7.886 vs mean gap 7.745. So "equal raw d_50
across models" is not the no-residual-effect null — it is close to unachievable by construction.
**The correct boundary-relative statistic is `d_50 - median(unsteered gap)`** — the *excess*
displacement a model costs over and above its own distance to the boundary. On that axis:
**base ~0.70, instruct ~0.14** — i.e. Instruct is, if anything, *easier* to move than base.
That points toward **Outcome A (the steering "decay" is a baseline-distance artifact)**, but
with n=2 models, no CIs, and mean standing in for median, it is indicative, not a result.

### Finding 2 — achieved displacement SATURATES AND REVERSES
Base mass-mean mean-gap by coefficient: c=-1.5 -> -5.498 (peak displacement ~6.31), c=-2 ->
-5.432, c=-4 -> -5.024, c=-12 -> -4.568. **Pushing harder past c~-1.5 buys less displacement.**
Consequences: (i) "extend c until every model crosses" is not always achievable — a deep-enough
model exhausts the grid instead of crossing; (ii) base's *maximum* achievable displacement
(~6.3) is **below Instruct's d_50 (7.886)**, so the two models barely overlap on the raw
displacement axis and any collapse test there is partly extrapolation. Both figures and the
verdict must state the overlap region explicitly.

### Finding 3 — mass-mean and logistic directions differ but dose alike
`cos(mass-mean, logistic) = 0.7388` at base L20, yet near-identical d_50 (1.511 vs 1.532).
Mass-mean reaches d_50 at smaller |c| (-0.272 vs -0.369) — mildly more dose-efficient,
consistent with Marks & Tegmark, but small and uncertained.

Behavioural (base only, 40 prompts, greedy 512 tok, at its own c_50): refusal 0.78 -> 0.60;
unclosed `<think>` 0.00. Smoke tests 5/5 pass. Logistic converged in 84 iterations at tol=1e-10.

**Deviations:** SEED=42 per P1, so the held-out split differs from A1's (split seed 0) — Gate A
numbers are NOT prompt-matched to A1. Dense 23-point c grid. Behavioural n=40 (below P2's 100;
raise on rerun, ~+2 min/model). The production split fingerprint was logged but never captured.

**Process lesson (cost us the data): pull each artifact THE MOMENT it lands, not at the end.**
Base finished ~2h before the reclaim and sat un-pulled.

## Sandbox mortality (2026-08-07) — treat as a design constraint

**Two boxes were reclaimed within hours**, both returning HTTP 410
`sandbox terminated` on every endpoint. Not stream drops — permanent.
What worked, and should stay standard practice:
- Write every artifact to sandbox disk immediately; pull to the local machine as
  soon as it exists.
- Mirror notebook cell bodies to `scripts/notebook_cells/` so a fresh box is a
  **restore, not a rebuild**.
- Keep a `RESTORE_*.md` per experiment recording pinned revisions, established
  facts, budget and decided deviations.
- **Probe for HTTP 410 FIRST** when a connection drops. Both agents initially
  entered retry loops against an already-dead box; the cheap probe should be step
  one, not step five.
Net loss across both deaths: 27 GB of cached activations, per-checkpoint bootstrap
replicas, and the A4 run. No reported conclusion lost its supporting file.

## Molab operational lessons (from the 3-box parallel runs)

- **Client disconnects INTERRUPT the running kernel cell** — and so does
  running any other cell (ctx.run_cell) mid-sweep. Long sweeps must run in a
  kernel-side threading.Thread; poll progress via variables, never cells.
- Probe fitting costs ~36 s/checkpoint (32 LogisticRegression fits) —
  comparable to extraction (~50 s); budget for it.
- Sandboxes: notebook + /marimo artifacts persist across sessions; HF cache
  and env vars do not. Backing store ~6.7 TB — activations can be kept.

## API gotchas (transformers 5.x / Olmo3 / molab)

1. `walledai/AdvBench` is now gated on HF. Public mirror used instead:
   `https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv`
   (column `goal`). `tatsu-lab/alpaca` is fine.
2. transformers 5.x removed `torch_dtype` -> use `dtype=torch.bfloat16`.
3. Olmo3 decoder layers return a plain `Tensor`, not a `(hidden_states, ...)` tuple.
   Hooks assuming a tuple silently no-op. Handle both shapes.
4. Ablating only decoder layers is insufficient (2/6): the embedding re-injects the
   direction. Hook `embed_tokens` too -> 6/6.
5. Olmo-3-Instruct's default chat template injects a function-calling system prompt
   into every prompt. Base/mid checkpoints likely have no chat template. The sweep
   needs an explicit prompt-format policy per comparison (see project.md confound
   note): raw text within pretraining, fixed template within post-training, never
   mixed in one figure.
6. Tokenizer right-pads with a real `<|pad|>` token; use `add_special_tokens=False`
   after `apply_chat_template`. Index last non-pad token via attention mask.
7. `accelerate` not needed for 7B bf16 single-GPU (`.to("cuda")`).
8. molab quirks: `shutil.disk_usage` free-space value is nonsense (use HF cache size
   as the disk guard); cells >60 s can drop the response stream while still
   completing (re-query state, don't assume failure); `execute-code.sh` needs a file
   path for long payloads and ASCII-only source (UTF-8 heredoc issue from Windows).
