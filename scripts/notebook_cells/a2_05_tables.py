# A2 headline table + the three key questions, computed rather than asserted.
import glob as _glob4
import json as _json4
import re as _re4

import numpy as _np4
import pandas as _pd4

# ---- (0) cross-check: does A2's linear/last reproduce the published sweep? --
_pub = []
for _f in sorted(_glob4.glob(A2_E1_DIR + "/results_*_scaffold.json")):
    _pub.extend(_json4.load(open(_f))["rows"])
_pub = _pd4.DataFrame(_pub)
_pub = _pub[_pub["pooling"] == "last"][["revision", "layer", "ood_auroc"]]
_pub = _pub.rename(columns={"ood_auroc": "published"})
_mine = A2_AGG[(A2_AGG["cond"] == "real") & (A2_AGG["rk"] == "linear/last")][
    ["revision", "layer", "ood_auroc"]
].rename(columns={"ood_auroc": "a2"})
_chk = _pub.merge(_mine, on=["revision", "layer"], how="inner")
A2_CROSSCHECK = {
    "n_compared": int(len(_chk)),
    "max_abs_diff": float((_chk["a2"] - _chk["published"]).abs().max()),
    "identical": bool((_chk["a2"] - _chk["published"]).abs().max() < 1e-9),
}

# ---- (1) onset per readout -------------------------------------------------
# Onset = first stage-1 step from which the best-layer OOD AUROC stays above
# the 97.5th percentile of that readout's own shuffled-label null, for every
# later checkpoint too (a single lucky checkpoint is not a formation event).
_s1c = A2_CURVE[A2_CURVE["stage"] == "stage1"].sort_values("step")
_onset = {}
for _rk in A2_ORDER:
    _sub = _s1c[_s1c["rk"] == _rk].reset_index(drop=True)
    _hit = None
    for _i in range(len(_sub)):
        if bool(_sub["signal"][_i:].all()):
            _hit = int(_sub["step"][_i])
            break
    _onset[_rk] = _hit
A2_ONSET = _onset

# ---- (2) layer profile -----------------------------------------------------
_late_rev = A2_STAGE1[-1]
_prof = A2_AGG[
    (A2_AGG["cond"] == "real") & (A2_AGG["revision"] == _late_rev)
].pivot(index="layer", columns="rk", values="ood_auroc")
A2_LAYER_PROFILE = _prof[[_c for _c in A2_ORDER if _c in _prof.columns]].round(3)
A2_L4_BESTFRAC = {
    _rk: float(
        _s1c[(_s1c["rk"] == _rk)]["best_layer"].pipe(lambda _s: (_s == 4).mean())
    )
    for _rk in A2_ORDER
}
_l4 = A2_AGG[(A2_AGG["cond"] == "real") & (A2_AGG["layer"] == 4)]
A2_L4_PEAK = (
    _l4[_l4["stage"] == "stage1"].groupby("rk")["ood_auroc"].max().round(3).to_dict()
)

# ---- (3) does the MLP beat the linear probe beyond its own control? --------
_wide = A2_CURVE.pivot_table(
    index=["revision", "step", "stage"], columns="rk",
    values=["ood_auroc", "ctl_auroc"],
).reset_index()
_wide.columns = [
    _c[0] if _c[1] == "" else _c[0] + ":" + _c[1] for _c in _wide.columns
]
_wide["d_mlp_minus_lin"] = _wide["ood_auroc:mlp/last"] - _wide["ood_auroc:linear/last"]
_wide["d_ctl"] = _wide["ctl_auroc:mlp/last"] - _wide["ctl_auroc:linear/last"]
_wide["d_meanpool_minus_last"] = (
    _wide["ood_auroc:linear/mean"] - _wide["ood_auroc:linear/last"]
)
A2_WIDE = _wide.sort_values(["stage", "step"]).reset_index(drop=True)

_s1w = A2_WIDE[A2_WIDE["stage"] == "stage1"]
A2_Q3 = {
    "mlp_minus_linear_mean": float(_s1w["d_mlp_minus_lin"].mean()),
    "mlp_minus_linear_max": float(_s1w["d_mlp_minus_lin"].max()),
    "mlp_minus_linear_max_at": str(
        _s1w.loc[_s1w["d_mlp_minus_lin"].idxmax(), "revision"]
    ),
    "n_ckpt_mlp_wins": int((_s1w["d_mlp_minus_lin"] > 0).sum()),
    "n_ckpt": int(len(_s1w)),
    "control_delta_mean": float(_s1w["d_ctl"].mean()),
    "meanpool_minus_last_mean": float(_s1w["d_meanpool_minus_last"].mean()),
}

# ---- headline table --------------------------------------------------------
A2_TABLE = A2_WIDE[A2_WIDE["stage"] == "stage1"][
    [
        "step",
        "ood_auroc:linear/last",
        "ood_auroc:linear/mean",
        "ood_auroc:mlp/last",
        "ood_auroc:mlp/mean",
        "ctl_auroc:mlp/last",
    ]
].round(3)
A2_TABLE.columns = [
    "step", "(a) linear/last", "(b) linear/mean", "(c) MLP/last",
    "(c2) MLP/mean", "MLP shuffled ctl",
]
A2_TABLE = A2_TABLE.reset_index(drop=True)
A2_TABLE
