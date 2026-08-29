# Box B E6 (32B stretch) - ran as an inline detached scratchpad worker, not a
# persisted cell. Mirrored here for reproducibility. Reuses e5_family's
# parameterized readout/mu/sweep/efficacy functions on OLMo-3-32B refusal at the
# proportional layer round(20/32*64)=40, neutral scaffold, R^5120 random-dir null
# on the |c|<=2 subgrid (efficacy only uses |c|<=2), NO behavioural generation.
# Base refusal direction (diff-in-means + logistic) fit once on base, carried
# unchanged. Commits resolved per checkpoint via HfApi and recorded.
# Checkpoints (32B pins from RESTORE_a4_32b.md; SFT stays in the 5e-5 lineage):
#   base      allenai/Olmo-3-1125-32B         main
#   sft-1000  allenai/Olmo-3-32B-Think-SFT    5e-5-step1000
#   dpo       allenai/Olmo-3-32B-Think-DPO    main
#   rlvr-last allenai/Olmo-3-32B-Think        step_750
# Writes /marimo/e6/e6_model_<lab>.json + status.json + log.txt; pulled per
# checkpoint. Aggregate E6_32b_scale.json + fig_E6_32b.png built after pull.
#
# Result: efficacy(mass-mean) 5.49/5.58/5.92/5.95 across base->rlvr-last
# (CV 0.041, z 9.7-12.6 vs R^5120 null); margin 0.95/0.84/2.09/2.15 (grows 2.3x).
# The 7B refusal shape (flat efficacy, growing margin) REPLICATES at 32B scale;
# margin growth is more gradual than 7B's 9.5x (only reaches RLVR-last, not a
# separate Instruct endpoint, and 32B SFT builds margin more slowly).
#
# The full worker body is the inline scratchpad code executed via execute-code.sh
# (see the session transcript); it references pipeline as P, gate_a as ga,
# e5_family as e5, and e1_causal.efficacy_from_gaps.
print("E6 32B: see e6_model_*.json + E6_32b_scale.json + fig_E6_32b.png")
