# E2-clean aggregation. Reads the per-checkpoint JSONs, refuses any file whose
# split fingerprint disagrees (P1), and builds the headline quantities with
# PAIRED bootstrap CIs. Pairing is real: every checkpoint reuses the same
# bootstrap count matrices, so replica b of checkpoint A and replica b of
# checkpoint B resample the same prompts.
import glob as _glob6
import json as _json6

import numpy as _np6
import pandas as _pd6

_files = sorted(_glob6.glob(E2_DIR + "/res_*.json"))
E2_REJECTED = {}
_payloads = []
for _f in _files:
    _p = _json6.load(open(_f))
    if (_p.get("split_fp") != E2_SPLIT_FP or _p.get("schema") != 2
            or _p.get("n_per_pool") != E2_CFG["n_per_pool"]):
        E2_REJECTED[_p.get("tag", _f)] = {
            "split_fp": _p.get("split_fp"), "schema": _p.get("schema"),
            "n_per_pool": _p.get("n_per_pool"),
        }
        continue
    _payloads.append(_p)

_rows = []
E2_BOOT = {}
for _p in _payloads:
    for _r in _p["rows"]:
        _key = (_r["tag"], _r["pooling"], _r["layer"])
        E2_BOOT[_key] = {
            "cos_dim": _np6.array(_r.pop("cos_dim_boot"), dtype=float),
            "cos_probe": _np6.array(_r.pop("cos_probe_boot"), dtype=float),
            "auroc_frozen": _np6.array(_r.pop("auroc_frozen_boot"), dtype=float),
            "auroc_refit": _np6.array(_r.pop("auroc_refit_boot"), dtype=float),
        }
        _r["n_per_pool"] = _p["n_per_pool"]
        _rows.append(_r)
E2_LONG = _pd6.DataFrame(_rows)

E2_ORDER_TAGS = [_r[0] for _r in E2_PLAN if _r[0] != E2_ANCHOR]
E2_TAG_ORDER = {_t: _i for _i, _t in enumerate(E2_ORDER_TAGS)}
E2_STAGE_OF = {_r[0]: _r[3] for _r in E2_PLAN}


def e2_layer_mean_boot(tag, pooling, field):
    """Bootstrap replicas of the across-layer mean, preserving pairing."""
    _M = _np6.stack([
        E2_BOOT[(tag, pooling, _L)][field] for _L in E2_LAYERS
        if (tag, pooling, _L) in E2_BOOT
    ])
    return _M.mean(axis=0)


def e2_summ(v):
    _a = _np6.asarray(v, dtype=float)
    return {"mean": float(_a.mean()),
            "lo": float(_np6.percentile(_a, 2.5)),
            "hi": float(_np6.percentile(_a, 97.5))}


# --- drift curve: across-layer mean cosine to base, per pooling -------------
_drift = []
for _tag in E2_ORDER_TAGS:
    for _pool in E2_POOLINGS:
        if (_tag, _pool, E2_LAYERS[0]) not in E2_BOOT:
            continue
        _b = e2_layer_mean_boot(_tag, _pool, "cos_dim")
        _pt = float(
            E2_LONG[(E2_LONG.tag == _tag) & (E2_LONG.pooling == _pool)]["cos_dim"].mean()
        )
        _s = e2_summ(_b)
        _drift.append({"tag": _tag, "stage": E2_STAGE_OF[_tag], "pooling": _pool,
                       "order": E2_TAG_ORDER[_tag], "cos_mean": _pt,
                       "lo": _s["lo"], "hi": _s["hi"]})
E2_DRIFT = _pd6.DataFrame(_drift).sort_values(["pooling", "order"]).reset_index(drop=True)


def e2_paired_delta(tag_a, tag_b, pooling, field="cos_dim"):
    """cos(tag_b) - cos(tag_a) with a PAIRED bootstrap CI and a two-sided
    bootstrap p-value for 'the difference is zero'."""
    _a = e2_layer_mean_boot(tag_a, pooling, field)
    _b = e2_layer_mean_boot(tag_b, pooling, field)
    _d = _b - _a
    _p = 2.0 * min(float((_d <= 0).mean()), float((_d >= 0).mean()))
    _s = e2_summ(_d)
    _s["p_boot"] = min(1.0, _p)
    _s["point"] = float(
        E2_LONG[(E2_LONG.tag == tag_b) & (E2_LONG.pooling == pooling)][field].mean()
        - E2_LONG[(E2_LONG.tag == tag_a) & (E2_LONG.pooling == pooling)][field].mean()
    )
    return _s


# --- the two claims under test ---------------------------------------------
# 1. the SFT cliff: how much of the total rotation happens sft_1000 -> sft_15000
# 2. RLVR inertness: total drift across the whole RLVR run
E2_TESTS = {}
for _pool in E2_POOLINGS:
    E2_TESTS[_pool] = {
        "sft_cliff_1000_to_15000": e2_paired_delta("sft_1000", "sft_15000", _pool),
        "sft_15000_to_43000": e2_paired_delta("sft_15000", "sft_43000", _pool),
        "sft_end_to_dpo": e2_paired_delta("sft_43000", "dpo", _pool),
        "dpo_to_rlvr_start": e2_paired_delta("dpo", "rlvr_0025", _pool),
        "rlvr_total_0025_to_1375": e2_paired_delta("rlvr_0025", "rlvr_1375", _pool),
    }

# --- frozen vs refit --------------------------------------------------------
_fr = []
for _tag in E2_ORDER_TAGS:
    for _pool in E2_POOLINGS:
        _sub = E2_LONG[(E2_LONG.tag == _tag) & (E2_LONG.pooling == _pool)]
        if len(_sub) == 0:
            continue
        _bf = e2_layer_mean_boot(_tag, _pool, "auroc_frozen")
        _br = e2_layer_mean_boot(_tag, _pool, "auroc_refit")
        _bg = _br - _bf
        _sg = e2_summ(_bg)
        # the layer where the refit probe is strongest = the honest headline layer
        _bl = int(_sub.loc[_sub["auroc_refit"].idxmax(), "layer"])
        _blr = _sub[_sub.layer == _bl].iloc[0]
        _fr.append({
            "tag": _tag, "stage": E2_STAGE_OF[_tag], "pooling": _pool,
            "order": E2_TAG_ORDER[_tag],
            "frozen_mean": float(_sub["auroc_frozen"].mean()),
            "refit_mean": float(_sub["auroc_refit"].mean()),
            "gap_mean": float(_sub["auroc_refit"].mean() - _sub["auroc_frozen"].mean()),
            "gap_lo": _sg["lo"], "gap_hi": _sg["hi"],
            "gap_excludes_zero": bool(_sg["lo"] > 0 or _sg["hi"] < 0),
            "best_layer": _bl,
            "frozen_bl": float(_blr["auroc_frozen"]), "refit_bl": float(_blr["auroc_refit"]),
            "gap_bl": float(_blr["gap_refit_minus_frozen"]),
            "gap_bl_lo": float(_blr["gap_ci"][0]), "gap_bl_hi": float(_blr["gap_ci"][1]),
            "null_mean": float(_sub["null_auroc_mean"].mean()),
            "cos_probe_L31": float(_sub[_sub.layer == 31]["cos_probe"].iloc[0]),
        })
E2_FROZEN = _pd6.DataFrame(_fr).sort_values(["pooling", "order"]).reset_index(drop=True)

E2_RANDREF = {
    "abs_cos_mean": float(E2_LONG["rand_cos_absmean"].mean()),
    "n_directions": E2_NRAND,
    "note": "cosine between two independent random unit vectors in R^4096",
}

{"n_ckpt": int(E2_LONG.tag.nunique()), "rejected": E2_REJECTED,
 "rand_ref": E2_RANDREF}
