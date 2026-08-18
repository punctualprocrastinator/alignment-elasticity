# E2-clean analysis sweep: one JSON per checkpoint, every headline number with
# a bootstrap 95% CI, both poolings, nulls and references included.
import json as _json5
import os as _os5
import threading as _th5
import time as _time5

import numpy as _np5
import torch as _torch5
from sklearn.metrics import accuracy_score as _acc5, roc_auc_score as _auc5


def e2_load_acts(cfg, tag):
    return _torch5.load(
        cfg["act_dir"] + "/acts_" + tag + ".pt", map_location="cpu",
        weights_only=False,
    )


def e2_base_cache(cfg):
    """Anchor quantities, computed once and shared (paired) across every target
    checkpoint: directions, bootstrap directions, frozen probe."""
    _y = _np5.array(cfg["labels"])
    _tr = _np5.array(cfg["van_tr"])
    _adv = _np5.array(cfg["adv_idx"])
    _pos = _np5.where(_y == 1)[0]
    _neg = _np5.where(_y == 0)[0]

    _C_dim = e2_count_matrix(_pos, _neg, len(_y), cfg["n_boot"], cfg["seed"])
    _tr_pos = _np5.where(_y[_tr] == 1)[0]
    _tr_neg = _np5.where(_y[_tr] == 0)[0]
    _C_fit = e2_count_matrix(_tr_pos, _tr_neg, len(_tr), cfg["n_boot_refit"], cfg["seed"] + 1)
    _rng = _np5.random.default_rng(cfg["seed"] + 2)
    _adv_pos = _np5.where(_y[_adv] == 1)[0]
    _adv_neg = _np5.where(_y[_adv] == 0)[0]
    _boot_eval = [
        _np5.concatenate([
            _rng.choice(_adv_pos, len(_adv_pos), replace=True),
            _rng.choice(_adv_neg, len(_adv_neg), replace=True),
        ])
        for _ in range(cfg["n_boot"])
    ]

    _acts = e2_load_acts(cfg, cfg["anchor"])
    _cache = {"C_dim": _C_dim, "C_fit": _C_fit, "boot_eval": _boot_eval,
              "y": _y, "tr": _tr, "adv": _adv, "per": {}}
    for _pool in cfg["poolings"]:
        for _L in cfg["layers"]:
            _X = _acts[_pool][_L].numpy().astype(_np5.float64)
            _d = e2_dim_dir(_X, _y)
            _bd = e2_boot_dim_dirs(_X, _y, _C_dim).astype(_np5.float32)
            _sc, _clf = e2_fit_probe(_X[_tr], _y[_tr])
            _pd = e2_probe_dir(_sc, _clf)
            _Z = _sc.transform(_X[_tr])
            _Wb, _ = e2_gpu_logreg(_Z, _y[_tr].astype(float), _C_fit)
            _Wb = _Wb / _sc.scale_
            _Wb = _Wb / (_np5.linalg.norm(_Wb, axis=1, keepdims=True) + 1e-12)
            _cache["per"][(_pool, _L)] = {
                "dim_dir": _d, "boot_dim": _bd, "sc": _sc, "clf": _clf,
                "probe_dir": _pd, "boot_probe": _Wb.astype(_np5.float32),
            }
    del _acts
    return _cache


def e2_analyze_one(cfg, cache, tag, stage, order):
    _path = cfg["dir"] + "/res_n" + str(cfg["n_per_pool"]) + "_" + tag + ".json"
    if _os5.path.exists(_path):
        try:
            _old = _json5.load(open(_path))
            if _old.get("split_fp") == cfg["split_fp"] and _old.get("schema") == 2:
                return "skipped"
        except Exception:
            pass
    _t0 = _time5.time()
    _y, _tr, _adv = cache["y"], cache["tr"], cache["adv"]
    _y_adv = _y[_adv]
    _acts = e2_load_acts(cfg, tag)
    _rows = []
    for _pi, _pool in enumerate(cfg["poolings"]):
        for _L in cfg["layers"]:
            _B = cache["per"][(_pool, _L)]
            _X = _acts[_pool][_L].numpy().astype(_np5.float64)

            _d = e2_dim_dir(_X, _y)
            _cos_dim = float(_d @ _B["dim_dir"])
            _bd = e2_boot_dim_dirs(_X, _y, cache["C_dim"]).astype(_np5.float32)
            _cos_dim_b = (_bd * _B["boot_dim"]).sum(1)

            _rr = _np5.random.default_rng(cfg["seed"] + 7 + _L + 100 * _pi)
            _R = _rr.normal(size=(cfg["n_rand"], _X.shape[1]))
            _R = _R / _np5.linalg.norm(_R, axis=1, keepdims=True)
            _rand_cos = _R @ _B["dim_dir"]

            _s_frozen = e2_scores(_B["sc"], _B["clf"], _X[_adv])
            _sc_t, _clf_t = e2_fit_probe(_X[_tr], _y[_tr])
            _s_refit = e2_scores(_sc_t, _clf_t, _X[_adv])
            _a_fro = float(_auc5(_y_adv, _s_frozen))
            _a_ref = float(_auc5(_y_adv, _s_refit))
            _bf, _br, _bg = e2_auc_boot(_s_frozen, _s_refit, _y_adv, cache["boot_eval"])

            _pd_t = e2_probe_dir(_sc_t, _clf_t)
            _cos_probe = float(_pd_t @ _B["probe_dir"])
            _Zt = _sc_t.transform(_X[_tr])
            _Wt, _ = e2_gpu_logreg(_Zt, _y[_tr].astype(float), cache["C_fit"])
            _Wt = _Wt / _sc_t.scale_
            _Wt = (_Wt / (_np5.linalg.norm(_Wt, axis=1, keepdims=True) + 1e-12)).astype(_np5.float32)
            _cos_probe_b = (_Wt * _B["boot_probe"]).sum(1)

            _Yn = _np5.stack([
                _np5.random.default_rng([cfg["seed"], _L, _pi, _s]).permutation(_y[_tr])
                for _s in cfg["null_seeds"]
            ], axis=1).astype(float)
            _Cn = _np5.ones((_Yn.shape[1], len(_tr)), dtype=_np5.float64)
            _Wn, _bn = e2_gpu_logreg(_Zt, _Yn, _Cn)
            _Zadv = _sc_t.transform(_X[_adv])
            _null = [
                float(_auc5(_y_adv, _Zadv @ _Wn[_k] + _bn[_k]))
                for _k in range(_Wn.shape[0])
            ]

            _rows.append({
                "tag": tag, "stage": stage, "order": order,
                "pooling": _pool, "layer": _L,
                "cos_dim": _cos_dim, "cos_dim_ci": e2_ci(_cos_dim_b),
                "cos_probe": _cos_probe, "cos_probe_ci": e2_ci(_cos_probe_b),
                "auroc_frozen": _a_fro, "auroc_frozen_ci": e2_ci(_bf),
                "auroc_refit": _a_ref, "auroc_refit_ci": e2_ci(_br),
                "gap_refit_minus_frozen": _a_ref - _a_fro,
                "gap_ci": e2_ci(_bg),
                "gap_frac_le0": float(_np5.mean(_np5.array(_bg) <= 0)),
                "acc_frozen": float(_acc5(_y_adv, (_s_frozen > 0).astype(int))),
                "acc_refit": float(_acc5(_y_adv, (_s_refit > 0).astype(int))),
                "null_auroc": _null,
                "null_auroc_mean": float(_np5.mean(_null)),
                "rand_cos_absmean": float(_np5.abs(_rand_cos).mean()),
                "rand_cos_ci": e2_ci(_rand_cos),
                "cos_dim_boot": [round(float(_v), 5) for _v in _cos_dim_b],
                "cos_probe_boot": [round(float(_v), 5) for _v in _cos_probe_b],
                "auroc_frozen_boot": [round(float(_v), 5) for _v in _bf],
                "auroc_refit_boot": [round(float(_v), 5) for _v in _br],
            })
    del _acts
    _payload = {
        "tag": tag, "stage": stage, "order": order,
        "split_fp": cfg["split_fp"], "seed": cfg["seed"],
        "commit": cfg["commits"].get(tag), "anchor": cfg["anchor"],
        "n_per_pool": cfg["n_per_pool"], "n_boot": cfg["n_boot"],
        "n_boot_refit": cfg["n_boot_refit"], "solver_tol": E2_TOL,
        "schema": 2,
        "deviations": cfg["deviations"],
        "elapsed_s": _time5.time() - _t0, "rows": _rows,
    }
    _tmp = _path + ".tmp"
    with open(_tmp, "w") as _fh:
        _json5.dump(_payload, _fh, default=float)
    _os5.replace(_tmp, _path)
    return "ok"


def e2_analysis_worker(cfg):
    _prog = cfg["dir"] + "/_analysis_progress.json"
    _state = {"started": _time5.time(), "n_total": len(cfg["plan"]) - 1,
              "done": [], "failed": {}, "current": "base-cache"}

    def _flush():
        _tmp = _prog + ".tmp"
        with open(_tmp, "w") as _fh:
            _json5.dump(_state, _fh)
        _os5.replace(_tmp, _prog)

    _flush()
    try:
        _cache = e2_base_cache(cfg)
    except Exception as _e:
        _state["failed"]["__base_cache__"] = type(_e).__name__ + ": " + str(_e)[:400]
        _state["finished"] = _time5.time()
        _flush()
        return
    for _row in cfg["plan"]:
        _tag, _stage, _order = _row[0], _row[3], _row[4]
        if _tag == cfg["anchor"]:
            continue
        _state["current"] = _tag
        _flush()
        try:
            _st = e2_analyze_one(cfg, _cache, _tag, _stage, _order)
            _state["done"].append([_tag, _st, round(_time5.time() - _state["started"], 1)])
        except Exception as _e:
            _state["failed"][_tag] = type(_e).__name__ + ": " + str(_e)[:400]
        _flush()
    _state["current"] = None
    _state["finished"] = _time5.time()
    _flush()


def e2_launch_analysis(cfg):
    _t = _th5.Thread(
        target=e2_analysis_worker, args=(dict(cfg),),
        name="e2-analysis-" + cfg["split_fp"], daemon=True,
    )
    _t.start()
    return _t


E2_ANALYSIS_READY = True
E2_ANALYSIS_READY
