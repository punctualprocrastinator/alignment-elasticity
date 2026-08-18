# A2 figure: formation curves for the three readouts on one axis, with the
# shuffled-label control bands, plus the depth profile that shows the layer
# ordering is a property of the POOLING, not of the model.
import os as _os3

import numpy as _np3
import matplotlib as _mpl
import matplotlib.pyplot as _plt
from matplotlib.lines import Line2D as _L2D
from matplotlib.patches import Patch as _Patch

_mpl.rcParams.update(
    {"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5}
)

A2_FIG_PATH = "/marimo/figures/fig_a2_probe_family.png"
_fig, _axes = _plt.subplots(1, 2, figsize=(14.0, 5.8), facecolor="white")
_axA, _axB = _axes
_nullx = A2_NULL.set_index("rk")


def _band(ax, rk, color, alpha, label=None):
    ax.axhspan(
        float(_nullx.loc[rk, "null_lo"]), float(_nullx.loc[rk, "null_hi"]),
        color=color, alpha=alpha, zorder=0, lw=0, label=label,
    )


# ---- panel A: formation curves (stage-1 pretraining only) -----------------
# Two control bands, because the width of the null is itself a result: the
# non-linear head has a much wider shuffled-label null than the linear one.
_band(_axA, "mlp/last", "#c1440e", 0.11, "shuffled null, MLP (95%)")
_band(_axA, "linear/last", "0.45", 0.22, "shuffled null, linear (95%)")
_axA.axhline(0.5, color="0.3", lw=0.9, ls=":", zorder=1)

_s1 = A2_CURVE[A2_CURVE["stage"] == "stage1"].sort_values("step")
for _rk in A2_ORDER:
    _sub = _s1[_s1["rk"] == _rk]
    if len(_sub) == 0:
        continue
    _axA.plot(
        _sub["step"].clip(lower=300), _sub["ood_auroc"],
        "-o" if "last" in _rk else "--s",
        color=A2_COLOR[_rk], label=A2_LABEL[_rk], lw=2.1, ms=6,
        alpha=1.0 if _rk != "mlp/mean" else 0.6, zorder=3,
    )

_axA.set_xscale("log")
_axA.set_xlim(280, 2.2e6)
_axA.set_xticks([300, 1e3, 1e4, 1e5, 1e6])
_axA.set_xticklabels(["0", "1k", "10k", "100k", "1M"])
_axA.set_xlabel("stage-1 pretraining step (log; step 0 at the left tick)")
_axA.set_ylabel("OOD-hard AUROC (vanilla -> adversarial), best layer")
_axA.set_title("A. The formation curve is a property of the model," + chr(10) + "not of the readout")
_axA.set_ylim(0.44, 0.88)
_axA.grid(True, alpha=0.3)
_axA.legend(fontsize=9, loc="lower right", framealpha=0.95)

# ---- panel B: depth profile, early vs end of stage 1 ----------------------
_band(_axB, "linear/last", "0.45", 0.22)
_axB.axhline(0.5, color="0.3", lw=0.9, ls=":", zorder=1)
_early, _late = "stage1-step1000", A2_STAGE1[-1]
for _rk in ["linear/last", "linear/mean", "mlp/last"]:
    for _rev, _ls, _al, _lw in [(_late, "-", 1.0, 2.1), (_early, "--", 0.45, 1.4)]:
        _sub = A2_AGG[
            (A2_AGG["cond"] == "real")
            & (A2_AGG["rk"] == _rk)
            & (A2_AGG["revision"] == _rev)
        ].sort_values("layer")
        _axB.plot(
            _sub["layer"], _sub["ood_auroc"], _ls, marker="o", ms=5,
            color=A2_COLOR[_rk], alpha=_al, lw=_lw, zorder=3,
        )
_axB.annotate(
    "layer 4 is at chance at the last token (0.53)" + chr(10)
    + "but strongly decodable mean-pooled (0.72)",
    xy=(4, 0.719), xytext=(6.2, 0.60), fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color="0.25", lw=1.1), color="0.15",
)
_axB.set_xlabel("layer")
_axB.set_ylabel("OOD-hard AUROC")
_axB.set_title("B. 'Deep layers lead, layer 4 is at chance'" + chr(10) + "is a last-token artifact")
_axB.set_xticks(A2_LAYERS)
_axB.set_ylim(0.44, 0.88)
_axB.grid(True, alpha=0.3)
_axB.legend(
    handles=[
        _L2D([], [], color=A2_COLOR[_k], lw=2.1, marker="o", ms=5, label=A2_LABEL[_k])
        for _k in ["linear/last", "linear/mean", "mlp/last"]
    ]
    + [
        _L2D([], [], color="0.35", lw=2.1, label="end of stage 1"),
        _L2D([], [], color="0.35", lw=1.4, ls="--", alpha=0.6, label="step 1k"),
        _Patch(color="0.45", alpha=0.22, label="shuffled null, linear (95%)"),
    ],
    fontsize=9, loc="lower right", framealpha=0.95, ncol=1,
)

_fig.suptitle(
    "A2: is the harmfulness formation curve an artifact of linear, last-token "
    "decoding?  (Olmo-3-1025-7B, stage 1)",
    fontsize=13.5, y=0.985,
)
_fig.tight_layout(rect=[0, 0, 1, 0.93])
_fig.savefig(A2_FIG_PATH, dpi=200, facecolor="white", bbox_inches="tight")
A2_FIG_BYTES = _os3.path.getsize(A2_FIG_PATH)
_fig
