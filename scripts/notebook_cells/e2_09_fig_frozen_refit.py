# E2-clean figure 2: frozen base probe vs refit probe along the flow.
# The question this figure answers: when the harm subspace rotates, does the
# information survive (frozen probe still reads it) or is it lost?
import os as _osf2

import numpy as _npf2
import matplotlib as _mplf2
import matplotlib.pyplot as _pltf2

_mplf2.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5,
                        "axes.labelsize": 11, "xtick.labelsize": 9.5,
                        "ytick.labelsize": 10})

E2_FIG2_PATH = "/marimo/figures/fig_e2_clean_frozen_refit.png"
_fig2, _ax2 = _pltf2.subplots(2, 2, figsize=(14.5, 9.0), facecolor="white")

for _k, _pool in enumerate(E2_POOLINGS):
    _sub = E2_FROZEN[E2_FROZEN.pooling == _pool].sort_values("order")
    _tags = list(_sub["tag"])
    _x = list(range(len(_tags)))

    _fl, _fh, _rl, _rh, _gl, _gh, _gm = [], [], [], [], [], [], []
    for _t in _tags:
        _bf = e2_layer_mean_boot(_t, _pool, "auroc_frozen")
        _br = e2_layer_mean_boot(_t, _pool, "auroc_refit")
        _bg = _br - _bf
        _fl.append(float(_npf2.percentile(_bf, 2.5)))
        _fh.append(float(_npf2.percentile(_bf, 97.5)))
        _rl.append(float(_npf2.percentile(_br, 2.5)))
        _rh.append(float(_npf2.percentile(_br, 97.5)))
        _gl.append(float(_npf2.percentile(_bg, 2.5)))
        _gh.append(float(_npf2.percentile(_bg, 97.5)))
        _gm.append(float(_bg.mean()))

    _axA = _ax2[0][_k]
    _axA.fill_between(_x, _fl, _fh, color="#1f4e9c", alpha=0.20, lw=0)
    _axA.plot(_x, _sub["frozen_mean"], "-o", color="#1f4e9c", lw=2.2, ms=6,
              label="frozen base probe")
    _axA.fill_between(_x, _rl, _rh, color="#0f8a6a", alpha=0.20, lw=0)
    _axA.plot(_x, _sub["refit_mean"], "-s", color="#0f8a6a", lw=2.2, ms=6,
              label="refit probe")
    _axA.plot(_x, _sub["null_mean"], ":x", color="0.35", lw=1.4, ms=5,
              label="shuffled-label null")
    _axA.axhline(0.5, color="0.4", lw=0.8, ls=":")
    _axA.set_xticks(_x)
    _axA.set_xticklabels([E2_LABELS_X.get(_t, _t) for _t in _tags], rotation=45, ha="right")
    _axA.set_title(("A. last token" if _k == 0 else "B. mean-pooled")
                   + "  -  OOD-hard AUROC (mean over 8 layers)")
    _axA.set_ylim(0.42, 1.0)
    _axA.grid(True, alpha=0.3)
    if _k == 0:
        _axA.set_ylabel("OOD-hard AUROC (vanilla -> adversarial)")
    _axA.legend(fontsize=9, loc="lower left", framealpha=0.95)

    _axB = _ax2[1][_k]
    _axB.fill_between(_x, _gl, _gh, color="#7a1fa2", alpha=0.22, lw=0)
    _axB.plot(_x, _gm, "-o", color="#7a1fa2", lw=2.2, ms=6,
              label="refit minus frozen")
    _axB.axhline(0.0, color="0.2", lw=1.2)
    _axB.axhline(0.041, color="#c1440e", lw=1.4, ls="--",
                 label="day-2 claim (+0.041, n=500, no CI)")
    _axB.set_xticks(_x)
    _axB.set_xticklabels([E2_LABELS_X.get(_t, _t) for _t in _tags], rotation=45, ha="right")
    _axB.set_title(("C. last token" if _k == 0 else "D. mean-pooled")
                   + "  -  elasticity gap, bootstrap 95% CI")
    _axB.grid(True, alpha=0.3)
    _axB.set_ylim(-0.09, 0.09)
    if _k == 0:
        _axB.set_ylabel("AUROC(refit) - AUROC(frozen)")
    _axB.legend(fontsize=9, loc="upper left", framealpha=0.95)
    _cospb = float(
        E2_LONG[(E2_LONG.tag == "rlvr_1375") & (E2_LONG.pooling == _pool)
                & (E2_LONG.layer == 31)]["cos_probe"].iloc[0]
    )
    _axB.text(
        0.99, 0.06,
        "probe direction has rotated to cos " + format(_cospb, ".2f")
        + " of base (L31)" + chr(10) + "yet the frozen probe loses ~"
        + format(abs(_gm[-2]), ".3f") + " AUROC",
        transform=_axB.transAxes, ha="right", va="bottom", fontsize=9.5,
        color="0.15",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.95),
    )

_fig2.suptitle(
    "E2-clean: the harm subspace rotates but the information does not move "
    "(neutral scaffold; paired bootstrap 95% CI; n=" + str(E2_CFG["n_per_pool"]) + " per pool)",
    fontsize=12.5, y=0.985,
)
_fig2.tight_layout(rect=[0, 0, 1, 0.955])
_fig2.savefig(E2_FIG2_PATH, dpi=200, facecolor="white", bbox_inches="tight")
E2_FIG2_BYTES = _osf2.path.getsize(E2_FIG2_PATH)
_fig2
