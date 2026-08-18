# E2-clean verdict. Every claim below is computed from the artifacts, with the
# protocol deviations stated up front rather than buried.
import json as _jv

import marimo as _mov
import numpy as _npv

_END = "rlvr_1375"
_V = {"protocol": "v1", "seed": E2_SEED, "split_fp": E2_SPLIT_FP,
      "n_per_pool": E2_CFG["n_per_pool"], "anchor": E2_ANCHOR,
      "solver_tol": E2_TOL, "deviations": E2_DEVIATIONS,
      "random_direction_reference": E2_RANDREF, "by_pooling": {}}

for _pool in E2_POOLINGS:
    _t = E2_TESTS[_pool]
    _fz = E2_FROZEN[E2_FROZEN.pooling == _pool].set_index("tag")
    _cliff = _t["sft_cliff_1000_to_15000"]
    _rlvr = _t["rlvr_total_0025_to_1375"]
    _end = _fz.loc[_END]
    _V["by_pooling"][_pool] = {
        "sft_cliff_delta_cos": _cliff,
        "sft_15000_to_43000": _t["sft_15000_to_43000"],
        "sft_end_to_dpo": _t["sft_end_to_dpo"],
        "rlvr_total_delta_cos": _rlvr,
        "rlvr_indistinguishable_from_zero": bool(_rlvr["lo"] <= 0 <= _rlvr["hi"]),
        "cliff_excludes_zero": bool(not (_cliff["lo"] <= 0 <= _cliff["hi"])),
        "cliff_over_rlvr_ratio": float(abs(_cliff["mean"]) / max(abs(_rlvr["mean"]), 1e-9)),
        "end_frozen_auroc": float(_end["frozen_mean"]),
        "end_refit_auroc": float(_end["refit_mean"]),
        "end_gap": float(_end["gap_mean"]),
        "end_gap_ci": [float(_end["gap_lo"]), float(_end["gap_hi"])],
        "end_gap_excludes_zero": bool(_end["gap_excludes_zero"]),
        "end_best_layer": int(_end["best_layer"]),
        "end_gap_best_layer": float(_end["gap_bl"]),
        "end_gap_best_layer_ci": [float(_end["gap_bl_lo"]), float(_end["gap_bl_hi"])],
        "end_probe_cos_L31": float(_end["cos_probe_L31"]),
        "end_cos_dim_mean": float(
            E2_DRIFT[(E2_DRIFT.pooling == _pool) & (E2_DRIFT.tag == _END)]["cos_mean"].iloc[0]
        ),
        "rlzero_cos_dim_mean": float(
            E2_DRIFT[(E2_DRIFT.pooling == _pool) & (E2_DRIFT.tag == "rlzero_1900")]["cos_mean"].iloc[0]
        ),
    }

_L, _M = _V["by_pooling"]["last"], _V["by_pooling"]["mean"]
_V["q1_cliff_survives"] = bool(_L["cliff_excludes_zero"] and _M["cliff_excludes_zero"])
_V["q1_rlvr_inert"] = bool(
    _L["rlvr_indistinguishable_from_zero"] or abs(_L["rlvr_total_delta_cos"]["mean"]) < 0.01
)
_V["q2_gap_is_004"] = bool(
    abs(_L["end_gap"] - 0.041) < 0.02 and _L["end_gap_excludes_zero"]
)
_V["q2_gap_defensible"] = bool(_L["end_gap_excludes_zero"] and _L["end_gap"] > 0)
_V["q3_pooling_agrees"] = bool(
    _L["cliff_excludes_zero"] == _M["cliff_excludes_zero"]
    and (_L["end_gap"] > 0) == (_M["end_gap"] > 0)
)
E2_VERDICT = _V
with open(E2_DIR + "/e2_clean_summary.json", "w") as _fh:
    _jv.dump(E2_VERDICT, _fh, indent=1, default=float)


def _fmt(d):
    return format(d["mean"], "+.4f") + " [" + format(d["lo"], "+.4f") + ", " + format(d["hi"], "+.4f") + "]"


_md = [
    "### E2-clean verdict (protocol v1, seed " + str(E2_SEED)
    + ", split " + E2_SPLIT_FP + ", n=" + str(E2_CFG["n_per_pool"]) + " per pool)",
    "",
    "**Deviation from protocol.md, stated up front:** P2 requires 1,000 prompts "
    "per pool. This run uses **500**, because resampling WildGuardMix needs the "
    "gated repo and no HF_TOKEN is available to this sandbox through a "
    "legitimate channel. Everything else is at protocol strength.",
    "",
    "**Q1. Does the SFT cliff survive with CIs, and is RLVR drift zero?**",
    "",
    "| pooling | SFT cliff (1k->15k) | RLVR total (25->1375) | cliff / RLVR |",
    "|---|---|---|---|",
]
for _pool in E2_POOLINGS:
    _b = _V["by_pooling"][_pool]
    _md.append(
        "| " + _pool + " | " + _fmt(_b["sft_cliff_delta_cos"]) + " | "
        + _fmt(_b["rlvr_total_delta_cos"]) + " | "
        + format(_b["cliff_over_rlvr_ratio"], ".0f") + "x |"
    )
_md += [
    "",
    "Cliff excludes zero in both poolings: **" + str(_V["q1_cliff_survives"])
    + "**. RLVR total drift indistinguishable from zero (last token): **"
    + str(_L["rlvr_indistinguishable_from_zero"]) + "**.",
    "",
    "**Q2. Is the frozen-vs-refit gap still ~0.04?**",
    "",
    "| pooling | frozen | refit | gap (layer-mean) | gap 95% CI | excludes 0 |",
    "|---|---|---|---|---|---|",
]
for _pool in E2_POOLINGS:
    _b = _V["by_pooling"][_pool]
    _md.append(
        "| " + _pool + " | " + format(_b["end_frozen_auroc"], ".4f") + " | "
        + format(_b["end_refit_auroc"], ".4f") + " | "
        + format(_b["end_gap"], "+.4f") + " | ["
        + format(_b["end_gap_ci"][0], "+.4f") + ", "
        + format(_b["end_gap_ci"][1], "+.4f") + "] | "
        + str(_b["end_gap_excludes_zero"]) + " |"
    )
_md += [
    "",
    "Day-2 reported frozen 0.888 vs refit 0.929, gap **+0.041**, with no CI and "
    "an under-converged solver. At protocol strength the gap at the end of the "
    "flow is " + format(_L["end_gap"], "+.4f") + " "
    + str([round(_x, 4) for _x in _L["end_gap_ci"]]) + " (last token). "
    + ("The 0.041 figure does NOT replicate." if not _V["q2_gap_is_004"]
       else "The 0.041 figure replicates."),
    "",
    "**Q3. Does mean-pooling change the story (as it did for depth in A2)?** "
    "Poolings agree on both headline signs: **" + str(_V["q3_pooling_agrees"])
    + "**. Cosine to base at the end of the flow: last "
    + format(_L["end_cos_dim_mean"], ".3f") + " vs mean-pooled "
    + format(_M["end_cos_dim_mean"], ".3f") + ".",
    "",
    "**Scale reference (P3).** Two independent random unit vectors in R^4096 "
    "have |cos| ~ " + format(E2_RANDREF["abs_cos_mean"], ".4f")
    + ", so a cosine of 0.6 is a *large* retained overlap, not near-orthogonality; "
    "and RL-Zero-Math at cos "
    + format(_L["rlzero_cos_dim_mean"], ".3f")
    + " is essentially the base model's geometry untouched.",
]
_mov.md(chr(10).join(_md))
