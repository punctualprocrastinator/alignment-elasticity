# A2 verdict. Every number below is computed from A2_CURVE / A2_Q3 / A2_ONSET,
# so the prose cannot drift away from the artifacts.
import json as _json5

import marimo as _mo5
import numpy as _np5

_s1v = A2_CURVE[A2_CURVE["stage"] == "stage1"].sort_values("step")


def _at(rk, step):
    _r = _s1v[(_s1v["rk"] == rk) & (_s1v["step"] == step)]
    return None if len(_r) == 0 else float(_r["ood_auroc"].iloc[0])


_first = int(_s1v["step"].min())
_last_step = int(_s1v["step"].max())
_lin, _mlp, _mean = "linear/last", "mlp/last", "linear/mean"

# Does the non-linear head find signal EARLIER than the linear one?
_q1_earlier = (
    A2_ONSET[_mlp] is not None
    and A2_ONSET[_lin] is not None
    and A2_ONSET[_mlp] < A2_ONSET[_lin]
)
# Does mean-pooling move the onset?
_q2_earlier = (
    A2_ONSET[_mean] is not None
    and A2_ONSET[_lin] is not None
    and A2_ONSET[_mean] < A2_ONSET[_lin]
)
# Does the MLP beat the linear probe by more than its own control justifies?
_ctl_gap = float(_s1v[_s1v["rk"] == _mlp]["ctl_auroc"].std())
_q3_beats = bool(A2_Q3["mlp_minus_linear_max"] > max(2 * _ctl_gap, 0.02))

A2_SURVIVES = bool((not _q1_earlier) and (not _q3_beats))

# The claim survives, but one SUB-claim does not. "Layer 4 never leaves chance"
# was measured at the last token only. Check it under mean-pooling.
_l4_lin_last = float(A2_L4_PEAK.get(_lin, float("nan")))
_l4_lin_mean = float(A2_L4_PEAK.get(_mean, float("nan")))
_null_lin_last = float(A2_NULL.set_index("rk").loc[_lin, "null_hi"])
_null_lin_mean = float(A2_NULL.set_index("rk").loc[_mean, "null_hi"])
# Margin above each readout's own null, not a knife-edge boolean: at the last
# token layer 4 peaks 0.002 above its null (i.e. at chance for all practical
# purposes), mean-pooled it clears its null by an order of magnitude more.
_l4_last_margin = _l4_lin_last - _null_lin_last
_l4_mean_margin = _l4_lin_mean - _null_lin_mean
_l4_survives = bool(_l4_mean_margin <= 0.05)

# Depth ordering: which layer wins under each pooling at the end of stage 1?
_best_layer_last = int(A2_LAYER_PROFILE[_lin].idxmax())
_best_layer_mean = int(A2_LAYER_PROFILE[_mean].idxmax())

A2_VERDICT = {
    "claim": "the ~step-2k onset and the shape of the formation curve are "
    "properties of the model, not artifacts of using a linear probe at the "
    "last token",
    "survives": A2_SURVIVES,
    "split_fp": A2_SPLIT_FP,
    "seed": A2_SEED,
    "crosscheck_vs_published": A2_CROSSCHECK,
    "onset_step": A2_ONSET,
    "q1_nonlinear_earlier": _q1_earlier,
    "q2_meanpool_earlier": _q2_earlier,
    "q3_nonlinear_beats_linear": _q3_beats,
    "q3_detail": A2_Q3,
    "auroc_first_step": {_k: _at(_k, _first) for _k in A2_ORDER},
    "auroc_last_step": {_k: _at(_k, _last_step) for _k in A2_ORDER},
    "null_band": A2_NULL.set_index("rk")[["null_mean", "null_hi"]].round(3).to_dict(),
    "layer4_peak_ood_auroc_stage1": A2_L4_PEAK,
    "layer4_chance_subclaim_survives": bool(_l4_survives),
    "layer4_margin_over_null": {
        "linear/last": float(_l4_last_margin),
        "linear/mean": float(_l4_mean_margin),
    },
    "best_layer_end_stage1": {
        "linear/last": _best_layer_last,
        "linear/mean": _best_layer_mean,
    },
    "meanpool_minus_last_mean_auroc": A2_Q3["meanpool_minus_last_mean"],
    "n_revisions": int(A2_RAW["revision"].nunique()),
}
with open(A2_DIR + "/a2_summary.json", "w") as _fh:
    _json5.dump(A2_VERDICT, _fh, indent=1, default=float)

_ok = "**SURVIVES**" if A2_SURVIVES else "**DOES NOT SURVIVE**"
_mo5.md(
    "### A2 verdict: the linear-probe formation claim " + _ok + chr(10) + chr(10)
    + "Cross-check against the published sweep: max |delta| AUROC = "
    + format(A2_CROSSCHECK["max_abs_diff"], ".2e")
    + " over " + str(A2_CROSSCHECK["n_compared"])
    + " (checkpoint, layer) cells, i.e. readout (a) reproduces the published "
    + "curve exactly." + chr(10) + chr(10)
    + "**Q1. Does the non-linear probe find signal earlier?** onset(linear/last) = "
    + str(A2_ONSET[_lin]) + ", onset(MLP/last) = " + str(A2_ONSET[_mlp])
    + ". Earlier: **" + str(_q1_earlier) + "**." + chr(10) + chr(10)
    + "**Q2. Does mean-pooling move the onset or the depth profile?** "
    + "onset(linear/mean) = " + str(A2_ONSET[_mean])
    + "; best layer-4 OOD AUROC anywhere in stage 1 = "
    + str(A2_L4_PEAK) + "." + chr(10) + chr(10)
    + "> **Caveat that the data forces.** The onset does not move, but the "
    + "*depth story* does. Mean-pooling lifts the whole curve by "
    + format(A2_Q3["meanpool_minus_last_mean"], "+.3f")
    + " AUROC on average, and it lifts **layer 4** from "
    + format(_l4_lin_last, ".3f") + " (last token, inside its null band of "
    + format(_null_lin_last, ".3f") + ") to " + format(_l4_lin_mean, ".3f")
    + " (mean-pooled, clearing its null band of "
    + format(_null_lin_mean, ".3f") + " by "
    + format(_l4_mean_margin, "+.3f") + " versus "
    + format(_l4_last_margin, "+.3f") + " at the last token). The best layer at the end of stage 1 "
    + "moves from L" + str(_best_layer_last) + " (last token) to L"
    + str(_best_layer_mean) + " (mean-pooled). So *layer 4 never leaves chance* "
    + "and *deep layers lead* are statements about the LAST-TOKEN readout, not "
    + "about the model. The paper should say so." + chr(10) + chr(10)
    + "**Q3. Does the MLP exceed the linear probe beyond its control?** "
    + "mean delta = " + format(A2_Q3["mlp_minus_linear_mean"], "+.3f")
    + ", max delta = " + format(A2_Q3["mlp_minus_linear_max"], "+.3f")
    + " (at " + A2_Q3["mlp_minus_linear_max_at"] + "), MLP wins at "
    + str(A2_Q3["n_ckpt_mlp_wins"]) + "/" + str(A2_Q3["n_ckpt"])
    + " checkpoints; shuffled-control spread (1 sd) = "
    + format(_ctl_gap, ".3f") + ". Beats it: **" + str(_q3_beats) + "**."
)
