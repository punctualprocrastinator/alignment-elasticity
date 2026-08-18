# E2-clean analysis worker: drift, frozen-vs-refit, probe-direction rotation,
# every headline number with a bootstrapped 95% CI (P4).
#
# P10: cfg is a thread argument. P1: SEED + split fingerprint in every file.
# P3: shuffled null with an INDEPENDENT permutation per (layer, pooling, seed),
#     plus a random-direction reference so a cosine of 0.68 is contextualised.
# Solver note: sklearn's default tol=1e-4 stops this problem after ~14 LBFGS
# iterations, well short of the L2 optimum (objective 1.9685 vs 1.8224), and
# the resulting direction sits at cosine 0.978 to the converged one. Since the
# E2 headline IS a direction cosine, every probe here is fit to tol=1e-10.
import json as _json4
import os as _os4
import threading as _th4
import time as _time4

import numpy as _np4
import torch as _torch4
from sklearn.linear_model import LogisticRegression as _LR4
from sklearn.metrics import accuracy_score as _acc4, roc_auc_score as _auc4
from sklearn.preprocessing import StandardScaler as _SS4

E2_TOL = 1e-10


def e2_fit_probe(X, y, tol=E2_TOL):
    """Converged L2 logistic probe. Returns (scaler, clf)."""
    _sc = _SS4().fit(X)
    _clf = _LR4(max_iter=20000, C=1.0, tol=tol).fit(_sc.transform(X), y)
    return _sc, _clf


def e2_scores(sc, clf, X):
    return clf.decision_function(sc.transform(X))


def e2_probe_dir(sc, clf):
    """Probe direction in RAW activation space (undo the scaler), unit norm."""
    _w = clf.coef_[0] / sc.scale_
    return _w / _np4.linalg.norm(_w)


def e2_unit(v):
    return v / (_np4.linalg.norm(v) + 1e-12)


def e2_dim_dir(X, y):
    """Diff-in-means refusal direction, unit norm."""
    return e2_unit(X[y == 1].mean(0) - X[y == 0].mean(0))


def e2_count_matrix(idx_pos, idx_neg, n_total, n_boot, seed):
    """Stratified bootstrap weights over prompts: resample within each class so
    class balance is preserved. Returns [n_boot, n_total] float counts."""
    _rng = _np4.random.default_rng(seed)
    _C = _np4.zeros((n_boot, n_total), dtype=_np4.float64)
    for _idx in (idx_pos, idx_neg):
        _draws = _rng.integers(0, len(_idx), size=(n_boot, len(_idx)))
        for _b in range(n_boot):
            _np4.add.at(_C[_b], _idx[_draws[_b]], 1.0)
    return _C


def e2_boot_dim_dirs(X, y, C):
    """Bootstrap diff-in-means directions via count matmul (no gathering).
    Runs on the GPU: the CPU float64 version is ~16 GFLOP per call and was the
    dominant cost of the whole sweep."""
    _Xg = _torch4.tensor(X, device="cuda", dtype=_torch4.float32)
    _Cg = _torch4.tensor(C, device="cuda", dtype=_torch4.float32)
    _pos = _torch4.tensor((y == 1).astype("float32"), device="cuda")
    _neg = _torch4.tensor((y == 0).astype("float32"), device="cuda")
    _Cp, _Cn = _Cg * _pos, _Cg * _neg
    _mp = (_Cp @ _Xg) / _Cp.sum(1, keepdim=True)
    _mn = (_Cn @ _Xg) / _Cn.sum(1, keepdim=True)
    _D = _mp - _mn
    _D = _D / (_D.norm(dim=1, keepdim=True) + 1e-12)
    _out = _D.cpu().numpy()
    del _Xg, _Cg, _Cp, _Cn, _D
    _torch4.cuda.empty_cache()
    return _out


def e2_gpu_logreg(Z, y, C, iters=300):
    """Batched L2 logistic on GPU, one replica per row of the count matrix C.
    Matches sklearn at tol=1e-10 (validated: cosine 1.000000).

    `y` may be 1-D (shared labels, bootstrap replicas) or 2-D [n, B] (one label
    vector per replica, used to batch the shuffled-label nulls into one call).
    Returns (W [B, d], b [B])."""
    _Zt = _torch4.tensor(Z, device="cuda", dtype=_torch4.float64)
    _yt = _torch4.tensor(_np4.asarray(y), device="cuda", dtype=_torch4.float64)
    _Ct = _torch4.tensor(C, device="cuda", dtype=_torch4.float64)
    _n, _d = _Zt.shape
    _B = _Ct.shape[0]
    _W = _torch4.zeros(_B, _d, device="cuda", dtype=_torch4.float64, requires_grad=True)
    _b = _torch4.zeros(_B, device="cuda", dtype=_torch4.float64, requires_grad=True)
    _opt = _torch4.optim.LBFGS(
        [_W, _b], max_iter=iters, tolerance_grad=1e-12, tolerance_change=1e-16,
        history_size=20, line_search_fn="strong_wolfe",
    )
    _lf = _torch4.nn.BCEWithLogitsLoss(reduction="none")
    _Y = _yt if _yt.dim() == 2 else _yt.unsqueeze(1).expand(_n, _B)

    def _closure():
        _opt.zero_grad()
        _L = _lf(_Zt @ _W.T + _b, _Y)
        _loss = ((_Ct.T * _L).sum(0) + 0.5 * (_W * _W).sum(1)).sum()
        _loss.backward()
        return _loss

    _opt.step(_closure)
    _outW = _W.detach().cpu().numpy()
    _outb = _b.detach().cpu().numpy()
    del _Zt, _Ct, _W, _b
    _torch4.cuda.empty_cache()
    return _outW, _outb


def e2_ci(v, lo=2.5, hi=97.5):
    _a = _np4.asarray(v, dtype=float)
    return [float(_np4.percentile(_a, lo)), float(_np4.percentile(_a, hi))]


def e2_auc_boot(scores_a, scores_b, y_eval, boot_idx):
    """Paired bootstrap of two AUROCs on the same resampled eval prompts."""
    _a, _b, _d = [], [], []
    for _ix in boot_idx:
        _yy = y_eval[_ix]
        if _yy.min() == _yy.max():
            continue
        _ua = float(_auc4(_yy, scores_a[_ix]))
        _ub = float(_auc4(_yy, scores_b[_ix]))
        _a.append(_ua)
        _b.append(_ub)
        _d.append(_ub - _ua)
    return _a, _b, _d
