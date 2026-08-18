# RESTORE: A4 - 32B scale replication (box lost mid-run)

Sandbox `sb-de00241482158d2a` was reclaimed (HTTP 410 "sandbox terminated")
while the A4 32B sweep was downloading the base anchor. **No 32B numbers
exist.** Everything needed to redo it is on this machine; this is a restore,
not a rebuild.

## What was lost

* 32B activation caches: **none completed**. Status at death was
  `load:base`, 0/7 checkpoints done, ~1 minute into a ~64.5 GB download.
* `/marimo/a4/analysis.json`: never produced.
* `fig_a4_32b_replication.png`: never produced.
* The live notebook (~110 cells, A1 + E4/E6 + A4 scaffolding). Cell bodies are
  rescued here; the kernel state is gone.

## What survived (nothing below needs re-deriving)

| Artifact | Path |
|---|---|
| A4 module (all 32B code) | `scripts/pipeline_a4.py` |
| Core pipeline (A1) | `scripts/pipeline.py` |
| E4/E6 module | `scripts/pipeline_e4e6.py` |
| 7B comparison curve for the A4 figure | `results/a4_ref7b_curve.json` |
| A1 results + figure | `results/a1_summary.json`, `figures/fig_a1_causal_power.png` |
| E4/E6 results + figures | `results/e4e6_clean_summary.json`, `figures/fig_e4_clean_behaviour.png`, `figures/fig_e6_clean_monitoring.png` |
| All notebook cell bodies | this directory |

## Facts already established (do NOT recompute)

* **Noise floor for R^5120 = 0.011155** (20 random unit vectors, 190 pairs;
  analytic sqrt(2/pi*d) = 0.011151). The same code reproduces **0.01172** for
  R^4096, matching the sibling 7B run's 0.0117 - implementation is validated.
* **32B geometry**: 64 layers, `d_model=5120`, vocab 100278, ~64.5 GB bf16.
* **Probe cost at P1b `tol=1e-10`**: 0.76 s per fit on (500 x 5120).
  `tol=1e-4` stops at 18 iterations, `tol=1e-10` at 33 - P1b confirmed on this
  data.
* **Box sizing**: 160 GB RAM, so a plain CPU load then `.to("cuda")` fits a
  32B checkpoint; no accelerate needed.
* **Contrast split fingerprint** (SEED=42, 500 AdvBench harmful / 500 Alpaca
  benign): `c4ed0db59d58`.

## Pinned revisions (branch AND commit; verified pairwise-distinct)

These were resolved with `HfApi().list_repo_refs` and their safetensors
sha256-hashed. `Olmo-3-32B-Think-SFT` ships TWO learning-rate lineages
(`5e-5-*` and `1e-4-*`) and its `main` is byte-distinct from both, so the cliff
contrast is kept inside the 5e-5 run.

| order | label | repo | revision | safetensors fp |
|---|---|---|---|---|
| 0 | base | allenai/Olmo-3-1125-32B | main | be0ef6a4f009aa7a |
| 1 | sft-1k | allenai/Olmo-3-32B-Think-SFT | 5e-5-step1000 | e7292f17021e01fa |
| 2 | sft-10790 | allenai/Olmo-3-32B-Think-SFT | 5e-5-step10790 | 91c593ba94cf300e |
| 3 | sft-main | allenai/Olmo-3-32B-Think-SFT | main | e6282dbc427d27de |
| 4 | dpo | allenai/Olmo-3-32B-Think-DPO | main | f65f29d56a00b876 |
| 5 | rlvr-50 | allenai/Olmo-3-32B-Think | step_050 | 636bbe53a3f164f1 |
| 6 | rlvr-750 | allenai/Olmo-3-32B-Think | step_750 | 211d82786f4babdd |

`Olmo-3-32B-Think` RLVR steps run step_050 .. step_750 (16 branches).
`Olmo-3-32B-Think-SFT` steps run 1000 .. 10790 per lineage.

## Restore procedure on a fresh box

1. Upload `scripts/pipeline.py` and `scripts/pipeline_a4.py` to `/marimo/`
   (base64 in <=6000-char chunks; there is a command-length cap).
2. Upload `results/a4_ref7b_curve.json` to `/marimo/a4/ref7b.json` - the A4
   figure overlays the 7B curve from it.
3. Run `32b_01_header_setup.py` then `32b_02_launch.py` through
   `marimo._code_mode`. The launcher is guarded: it refuses to spawn a second
   `a4-worker` while one is alive and skips checkpoints whose activation cache
   already exists, so re-running is safe.
4. Poll `/marimo/a4/log.txt` and `/marimo/a4/status.json` by READING FILES.
   Never poll by running cells.
5. Still to be written (was never reached): the A4 drift table cell, the
   `fig_a4_32b_replication.png` figure cell overlaying the 7B curve, and the
   verdict cell.

## Budget estimate (unvalidated - the run died before the first checkpoint)

7 checkpoints x 64.5 GB. Download dominates. At the 7B-observed ~420 MB/s that
is ~150 s per checkpoint of download, plus CPU->GPU load and ~1-2 min of
extraction over 1000 prompts at batch 8. Phase B analysis is CPU-only and was
budgeted at ~25 min (dominated by 60 refit bootstraps x 2 poolings x 7
checkpoints at 0.76 s per logistic fit). **Consider dropping `sft-main` and
`rlvr-50` first** if time is short: base -> sft-1k -> sft-10790 -> dpo ->
rlvr-750 still shows cliff-then-flat, though it costs the RLVR-drift contrast.

## Deviations already decided (carry them into the result file)

* `allenai/wildguardmix` is gated and no `HF_TOKEN` is present, so pools are
  AdvBench vs Alpaca. The direction work does not need the adversarial split;
  the frozen-vs-refit AUROC arm therefore runs on the EASIER split and its
  absolute AUROCs are not comparable to a WildGuard number - only the
  within-split frozen-vs-refit gap is.
* 500 per pool, not P2's 1000: AdvBench has only 520 harmful items total. This
  matches the sibling 7B run exactly, which is the point of the comparison.
* Probe-direction cosine CIs use 60 refit resamples (not 1000); every
  non-refit quantity uses the full 1000. Each refit is a fresh logistic fit.

## Note on the patch files

`*_patch.py` files call `ctx.edit_cell` with cell IDs from the dead session
(e.g. `EBee`, `xLLj`, `pbjG`). On a fresh box those IDs will not exist - fold
the replacement strings into the corresponding base builder instead, or re-run
the patch against the new IDs.
