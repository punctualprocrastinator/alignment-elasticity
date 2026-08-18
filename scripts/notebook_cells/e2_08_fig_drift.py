# E2-clean figure 1: refusal-direction drift along the post-training flow.
# One format only (neutral scaffold, P6). Both poolings (P5). Bootstrap 95%
# bands on every point (P4). Random-direction reference so a cosine of ~0.6 is
# read against the right scale, not against 0.
import os as _osf1

import numpy as _npf1

import matplotlib as _mplf1
import matplotlib.pyplot as _pltf1

_mplf1.rcParams.update({"font.size": 10.5, "axes.titlesize": 12,
                        "axes.labelsize": 11, "xtick.labelsize": 9.5,
                        "ytick.labelsize": 10})

E2_LABELS_X = {
    "sft_1000": "SFT 1k", "sft_15000": "SFT 15k", "sft_29000": "SFT 29k",
    "sft_43000": "SFT 43k", "dpo": "DPO", "rlvr_0025": "RL 25",
    "rlvr_0275": "RL 275", "rlvr_0550": "RL 550", "rlvr_0825": "RL 825",
    "rlvr_1100": "RL 1100", "rlvr_1375": "RL 1375", "rlzero_1900": "RLZero 1900",
}
E2_FIG1_PATH = "/marimo/figures/fig_e2_clean_drift.png"

_fig1, _ax1 = _pltf1.subplots(1, 2, figsize=(14.5, 5.6), facecolor="white", sharey=True)
for _k, _pool in enumerate(E2_POOLINGS):
    _ax = _ax1[_k]
    _sub = E2_DRIFT[E2_DRIFT.pooling == _pool].sort_values("order")
    _x = list(range(len(_sub)))
    _tags = list(_sub["tag"])

    _ax.fill_between(_x, _sub["lo"], _sub["hi"], color="#1f4e9c", alpha=0.22, lw=0)
    _ax.plot(_x, _sub["cos_mean"], "-o", color="#1f4e9c", lw=2.2, ms=6,
             label="diff-in-means refusal direction")

    _pc, _plo, _phi = [], [], []
    for _t in _tags:
        _r = E2_LONG[(E2_LONG.tag == _t) & (E2_LONG.pooling == _pool)]
        _pc.append(float(_r["cos_probe"].mean()))
        _bb = e2_layer_mean_boot(_t, _pool, "cos_probe")
        _plo.append(float(_npf1.percentile(_bb, 2.5)))
        _phi.append(float(_npf1.percentile(_bb, 97.5)))
    _ax.fill_between(_x, _plo, _phi, color="#c1440e", alpha=0.18, lw=0)
    _ax.plot(_x, _pc, "--s", color="#c1440e", lw=2.0, ms=5.5,
             label="logistic probe direction")

    _rr = E2_RANDREF["abs_cos_mean"]
    _ax.axhspan(-_rr, _rr, color="0.5", alpha=0.35, lw=0,
                label="random direction reference (|cos| ~ " + format(_rr, ".3f") + ")")
    _ax.axhline(1.0, color="0.3", lw=0.9, ls=":")

    # mark the SFT cliff
    if "sft_1000" in _tags and "sft_15000" in _tags:
        _i0, _i1 = _tags.index("sft_1000"), _tags.index("sft_15000")
        _ax.annotate(
            "", xy=(_i1, _sub["cos_mean"].iloc[_i1]), xytext=(_i0, _sub["cos_mean"].iloc[_i0]),
            arrowprops=dict(arrowstyle="<->", color="#7a1fa2", lw=2.0),
        )
        _mid = (float(_sub["cos_mean"].iloc[_i0]) + float(_sub["cos_mean"].iloc[_i1])) / 2.0
        _ax.text((_i0 + _i1) / 2.0 + 0.35, min(_mid + 0.055, 0.92),
                 "SFT cliff", color="#7a1fa2", fontsize=10.5, ha="left", weight="bold")

    _ax.set_xticks(_x)
    _ax.set_xticklabels([E2_LABELS_X.get(_t, _t) for _t in _tags], rotation=45, ha="right")
    _ax.set_title(("A. last token" if _k == 0 else "B. mean-pooled")
                  + "  (cosine to base anchor, mean over 8 layers)")
    _ax.grid(True, alpha=0.3)
    _ax.set_ylim(-0.08, 1.05)
    if _k == 0:
        _ax.set_ylabel("cosine to allenai/Olmo-3-1025-7B (main)")
    _ax.legend(fontsize=8.8, loc="lower left", framealpha=0.95)

_fig1.suptitle(
    "E2-clean: alignment geometry moves once, in early SFT, then stops "
    "(neutral scaffold; bands = bootstrap 95% CI; n=" + str(E2_CFG["n_per_pool"]) + " per pool)",
    fontsize=12.5, y=0.98,
)
_fig1.tight_layout(rect=[0, 0, 1, 0.93])
_fig1.savefig(E2_FIG1_PATH, dpi=200, facecolor="white", bbox_inches="tight")
E2_FIG1_BYTES = _osf1.path.getsize(E2_FIG1_PATH)
_fig1
