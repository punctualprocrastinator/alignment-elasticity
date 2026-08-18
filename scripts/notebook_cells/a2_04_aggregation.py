# A2 aggregation. Reads the per-checkpoint JSONs written by the detached sweep
# and reduces them to (i) a tidy long frame, (ii) MLP seed-averaged scores and
# (iii) the best-layer formation curve per readout.
import glob as _glob
import json as _json2
import re as _re2

import numpy as _np2
import pandas as _pd2

# Only trust result files whose recorded split fingerprint matches the split
# this notebook currently defines. Anything else is a leftover from an earlier
# parameterisation and is reported rather than silently averaged in.
_rows = []
A2_REJECTED = {}
for _f in sorted(_glob.glob(A2_DIR + "/res_*.json")):
    _p = _json2.load(open(_f))
    if _p.get("split_fp") != A2_SPLIT_FP:
        A2_REJECTED[_p.get("revision", _f)] = _p.get("split_fp")
        continue
    _rows.extend(_p["rows"])
A2_RAW = _pd2.DataFrame(_rows)
assert len(A2_RAW) > 0, "no result files match the current split fingerprint"


def _a2_step2(rev):
    _m = _re2.search(r"step(\d+)", rev)
    return int(_m.group(1)) if _m else -1


A2_RAW["step"] = A2_RAW["revision"].map(_a2_step2)
A2_RAW["stage"] = A2_RAW["revision"].map(lambda _r: _r.split("-")[0])
# "readout key": the three headline readouts plus the mean-pooled MLP bonus.
A2_RAW["rk"] = A2_RAW["readout"] + "/" + A2_RAW["pooling"]

# Average the MLP over its 3 initialisation seeds; linear has a single fit.
A2_AGG = (
    A2_RAW.groupby(["revision", "step", "stage", "rk", "cond", "layer"], as_index=False)
    .agg(
        ood_auroc=("ood_auroc", "mean"),
        ood_auroc_sd=("ood_auroc", "std"),
        ood_acc=("ood_acc", "mean"),
        id_auroc=("id_auroc", "mean"),
        id_acc=("id_acc", "mean"),
        n_seed=("ood_auroc", "size"),
    )
)

# Null band: distribution of shuffled-label OOD AUROC, per readout, pooled over
# every layer, checkpoint and seed. A real score only counts as signal if it
# clears the upper tail of its OWN readout's null.
A2_NULL = (
    A2_RAW[A2_RAW["cond"] == "shuffled"]
    .groupby("rk")["ood_auroc"]
    .agg(
        null_mean="mean",
        null_sd="std",
        null_lo=lambda _s: float(_np2.quantile(_s, 0.025)),
        null_hi=lambda _s: float(_np2.quantile(_s, 0.975)),
        null_max="max",
    )
    .reset_index()
)

# Best-layer formation curve: for each checkpoint x readout, the layer with the
# highest real OOD AUROC, and the matching shuffled score at the same layer.
_real = A2_AGG[A2_AGG["cond"] == "real"]
_shuf = A2_AGG[A2_AGG["cond"] == "shuffled"][
    ["revision", "rk", "layer", "ood_auroc"]
].rename(columns={"ood_auroc": "ctl_auroc"})
_best_idx = _real.groupby(["revision", "rk"])["ood_auroc"].idxmax()
A2_CURVE = (
    _real.loc[_best_idx]
    .merge(_shuf, on=["revision", "rk", "layer"], how="left")
    .merge(A2_NULL, on="rk", how="left")
    .sort_values(["rk", "step"])
    .reset_index(drop=True)
)
A2_CURVE["signal"] = A2_CURVE["ood_auroc"] > A2_CURVE["null_hi"]
A2_CURVE = A2_CURVE.rename(columns={"layer": "best_layer"})

A2_ORDER = ["linear/last", "linear/mean", "mlp/last", "mlp/mean"]
A2_LABEL = {
    "linear/last": "(a) linear, last token",
    "linear/mean": "(b) linear, mean-pooled",
    "mlp/last": "(c) MLP, last token",
    "mlp/mean": "(c2) MLP, mean-pooled",
}
A2_COLOR = {
    "linear/last": "#1f4e9c",
    "linear/mean": "#0f8a6a",
    "mlp/last": "#c1440e",
    "mlp/mean": "#8a6bbf",
}

{'rejected_files': A2_REJECTED, 'null': A2_NULL}
