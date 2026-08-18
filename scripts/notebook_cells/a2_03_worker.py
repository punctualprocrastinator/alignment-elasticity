# A2 probe-family machinery.
#
# Three readouts on the SAME cached activation tensors, so any difference is
# the readout and nothing else:
#   (a) linear  / last-token   logistic regression  (the published protocol)
#   (b) linear  / mean-pooled  logistic regression
#   (c) mlp     / last-token   4096 -> 256 ReLU -> 1, early stopping
#   (c2) mlp    / mean-pooled  (same head, reported as a bonus)
# plus a shuffled-label control for every one of them.
#
# Protocol invariants copied from section 7c/8:
#   fit   = A2_VAN_TR   (800 vanilla prompts)
#   ID    = A2_VAN_TE   (200 held-out vanilla)
#   OOD   = A2_ADV_IDX  (1000 adversarial)  <- the primary metric
#   features standardised on the fit split only.
# The MLP additionally carves an internal 80/20 fit/val split out of the 800
# for early stopping, so it never sees ID or OOD data during training.
import hashlib as _hashlib
import json as _json
import os as _os
import time as _time
import threading as _threading

import numpy as _np
import torch as _torch
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.metrics import accuracy_score as _acc, roc_auc_score as _auc
from sklearn.model_selection import train_test_split as _tts2
from sklearn.pipeline import make_pipeline as _mkpipe
from sklearn.preprocessing import StandardScaler as _SS

# Generation tag. The launch cell names its thread after this, and
# a2_run_revision refuses to do work from a thread of an older generation, so
# re-running this cell cleanly retires an in-flight sweep instead of racing it.
A2_GEN = "g2-seed42"
# Fingerprint of the fit split. Written into every result file; anything that
# does not match the current split is recomputed rather than trusted.
A2_SPLIT_FP = _hashlib.sha1(
    A2_VAN_TR.tobytes() + A2_VAN_TE.tobytes() + A2_ADV_IDX.tobytes()
).hexdigest()[:12]

A2_MLP_HIDDEN = 256
A2_MLP_MAX_EPOCHS = 120
A2_MLP_PATIENCE = 12
A2_MLP_BATCH = 64
A2_MLP_LR = 1e-3
A2_MLP_WD = 1e-4


def a2_linear_scores(X, y):
    _clf = _mkpipe(_SS(), _LR(max_iter=5000, C=1.0))
    _clf.fit(X[A2_VAN_TR], y[A2_VAN_TR])
    _out = {}
    for _tag, _idx in [("id", A2_VAN_TE), ("ood", A2_ADV_IDX)]:
        _s = _clf.decision_function(X[_idx])
        _out[_tag + "_acc"] = float(_acc(A2_LABELS[_idx], _clf.predict(X[_idx])))
        _out[_tag + "_auroc"] = float(_auc(A2_LABELS[_idx], _s))
    return _out


def a2_mlp_scores(X, y, seed):
    _torch.manual_seed(seed)
    _fit, _val = _tts2(
        A2_VAN_TR, test_size=0.2, stratify=y[A2_VAN_TR], random_state=seed
    )
    _sc = _SS().fit(X[_fit])

    def _T(idx):
        return _torch.tensor(
            _sc.transform(X[idx]), dtype=_torch.float32, device="cuda"
        )

    _Xf, _Xv = _T(_fit), _T(_val)
    _yf = _torch.tensor(y[_fit], dtype=_torch.float32, device="cuda")
    _yv = y[_val]

    _net = _torch.nn.Sequential(
        _torch.nn.Linear(X.shape[1], A2_MLP_HIDDEN),
        _torch.nn.ReLU(),
        _torch.nn.Linear(A2_MLP_HIDDEN, 1),
    ).cuda()
    _opt = _torch.optim.Adam(
        _net.parameters(), lr=A2_MLP_LR, weight_decay=A2_MLP_WD
    )
    _lossf = _torch.nn.BCEWithLogitsLoss()

    _best = -1.0
    _best_ep = 0
    _best_state = None
    _bad = 0
    for _ep in range(A2_MLP_MAX_EPOCHS):
        _net.train()
        _perm = _torch.randperm(len(_Xf), device="cuda")
        for _i in range(0, len(_Xf), A2_MLP_BATCH):
            _b = _perm[_i : _i + A2_MLP_BATCH]
            _opt.zero_grad()
            _lossf(_net(_Xf[_b]).squeeze(1), _yf[_b]).backward()
            _opt.step()
        _net.eval()
        with _torch.no_grad():
            _sv = _net(_Xv).squeeze(1).float().cpu().numpy()
        _v = float(_auc(_yv, _sv)) if len(set(_yv.tolist())) > 1 else 0.5
        if _v > _best + 1e-4:
            _best = _v
            _best_ep = _ep
            _bad = 0
            _best_state = {
                _k: _t.detach().clone() for _k, _t in _net.state_dict().items()
            }
        else:
            _bad += 1
            if _bad >= A2_MLP_PATIENCE:
                break
    if _best_state is not None:
        _net.load_state_dict(_best_state)

    _net.eval()
    _out = {"val_auroc": _best, "stop_epoch": int(_best_ep)}
    with _torch.no_grad():
        for _tag, _idx in [("id", A2_VAN_TE), ("ood", A2_ADV_IDX)]:
            _s = _net(_T(_idx)).squeeze(1).float().cpu().numpy()
            _out[_tag + "_acc"] = float(_acc(A2_LABELS[_idx], (_s > 0).astype(int)))
            _out[_tag + "_auroc"] = float(_auc(A2_LABELS[_idx], _s))
    del _Xf, _Xv, _net
    _torch.cuda.empty_cache()
    return _out


def a2_run_revision(revision):
    if not _threading.current_thread().name.endswith(A2_GEN):
        return "aborted-stale-generation"
    _path = A2_DIR + "/res_" + revision + ".json"
    if _os.path.exists(_path):
        try:
            if _json.load(open(_path)).get("split_fp") == A2_SPLIT_FP:
                return "skipped"
        except Exception:
            pass
    _t0 = _time.time()
    _obj = _torch.load(
        A2_ACTS_DIR + "/acts_" + revision + "_scaffold.pt",
        map_location="cpu",
        weights_only=False,
    )
    # Every shuffled-label control gets its OWN permutation, keyed on
    # (layer, pooling, draw). One shared permutation would make the null band a
    # single correlated draw -- and the per-layer spread of a shuffled probe on
    # the OOD pools is large, so that would badly mis-state the band.
    def _shuffled_y(*key):
        _yy = A2_LABELS.copy()
        _yy[A2_VAN_TR] = _np.random.default_rng(
            [A2_SEED] + [int(_k) for _k in key]
        ).permutation(A2_LABELS[A2_VAN_TR])
        return _yy

    _rows = []
    for _pi, _pool in enumerate(["last", "mean"]):
        for _L in A2_LAYERS:
            _X = _obj[_pool][_L].numpy()
            for _cond in ["real", "shuffled"]:
                _y_lin = (
                    A2_LABELS if _cond == "real" else _shuffled_y(_L, _pi, 900)
                )
                _r = a2_linear_scores(_X, _y_lin)
                _r.update(
                    revision=revision, layer=_L, pooling=_pool,
                    readout="linear", cond=_cond, seed=-1,
                )
                _rows.append(_r)
                for _sd in A2_MLP_SEEDS:
                    _y_mlp = (
                        A2_LABELS if _cond == "real" else _shuffled_y(_L, _pi, _sd)
                    )
                    _m = a2_mlp_scores(_X, _y_mlp, _sd)
                    _m.update(
                        revision=revision, layer=_L, pooling=_pool,
                        readout="mlp", cond=_cond, seed=_sd,
                    )
                    _rows.append(_m)
    del _obj
    _payload = {
        "revision": revision,
        "split_fp": A2_SPLIT_FP,
        "seed": A2_SEED,
        "gen": A2_GEN,
        "elapsed_s": _time.time() - _t0,
        "rows": _rows,
    }
    _tmp = _path + ".tmp"
    with open(_tmp, "w") as _fh:
        _json.dump(_payload, _fh)
    _os.replace(_tmp, _path)
    return "ok"


def a2_worker(revisions):
    _prog = A2_DIR + "/_progress.json"
    _state = {
        "started": _time.time(),
        "todo": list(revisions),
        "done": [],
        "failed": {},
    }

    def _flush():
        _tmp = _prog + ".tmp"
        with open(_tmp, "w") as _fh:
            _json.dump(_state, _fh)
        _os.replace(_tmp, _prog)

    _flush()
    for _rev in revisions:
        _state["current"] = _rev
        _flush()
        try:
            _status = a2_run_revision(_rev)
            _state["done"].append(
                [_rev, _status, round(_time.time() - _state["started"], 1)]
            )
        except Exception as _e:
            _state["failed"][_rev] = type(_e).__name__ + ": " + str(_e)[:300]
        _flush()
    _state["current"] = None
    _state["finished"] = _time.time()
    _flush()


def a2_launch(revisions=None):
    # Long jobs MUST live in a kernel-side daemon thread: a client disconnect
    # or any ctx.run_cell interrupts a running notebook cell.
    _revs = list(A2_REVS) if revisions is None else list(revisions)
    _th = _threading.Thread(
        target=a2_worker, args=(_revs,), name="a2-probe-family-" + A2_GEN,
        daemon=True,
    )
    _th.start()
    return _th


A2_READY = {
    "gen": A2_GEN,
    "split_fp": A2_SPLIT_FP,
    "seed": A2_SEED,
    "revisions": len(A2_REVS),
    "fits_per_revision": len(A2_LAYERS) * 2 * 2 * (1 + len(A2_MLP_SEEDS)),
    "mlp": {
        "hidden": A2_MLP_HIDDEN,
        "max_epochs": A2_MLP_MAX_EPOCHS,
        "patience": A2_MLP_PATIENCE,
        "lr": A2_MLP_LR,
        "wd": A2_MLP_WD,
        "seeds": A2_MLP_SEEDS,
    },
}
A2_READY
