# Box B E4 honesty launch (marimo id fHMI). Detached worker: base truth direction
# (diff-in-means + logistic tol=1e-10) at L20 carried unchanged across
# base / think-sft-1000 / think-dpo / think-rlvr-last / instruct. Per checkpoint:
# margin, efficacy(|c|<=2) + bootstrap CI, 20-random-dir null z, c50, and a
# label-frame vs agreement-frame dissociation at c50.
# Poll /marimo/e4/status.json + log.txt; pull e4_model_*.json as they land.
print(e4_honesty.e4_launch())
