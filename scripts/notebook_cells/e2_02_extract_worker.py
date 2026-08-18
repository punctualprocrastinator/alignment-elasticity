# E2-clean extraction worker.
#
# P10: every function below takes `cfg` explicitly. Nothing reads a notebook
# global, so re-running any cell cannot kill a running sweep.
# P7: MAX_LEN=384, LEFT truncation (readout is the last token), last NON-PAD
#     token indexed via the attention mask, fraction hitting MAX_LEN logged,
#     per-layer direction norms checked so a degenerate layer is rejected.
# P9: exact revision strings pinned; commit hash recorded in every result file.
import gc as _gc
import json as _json3
import os as _os3
import threading as _th3
import time as _time3

import numpy as _np3
import torch as _torch3
from transformers import AutoModelForCausalLM as _AMLM, AutoTokenizer as _ATok


def e2_purge(repo, revision):
    """Drop this revision's weights from the HF cache (P10). Returns bytes."""
    from huggingface_hub import scan_cache_dir

    try:
        _cache = scan_cache_dir()
    except Exception:
        return 0
    _freed = 0
    for _r in _cache.repos:
        if _r.repo_id != repo:
            continue
        _hashes = [_rev.commit_hash for _rev in _r.revisions if revision in _rev.refs]
        if not _hashes:
            _hashes = [_rev.commit_hash for _rev in _r.revisions]
        if _hashes:
            _st = _cache.delete_revisions(*_hashes)
            _freed += _st.expected_freed_size
            _st.execute()
    return _freed


def e2_extract_one(cfg, tag, repo, revision):
    """Download -> extract last+mean activations -> save -> purge weights."""
    _path = cfg["act_dir"] + "/acts_" + tag + ".pt"
    _meta_path = cfg["act_dir"] + "/meta_" + tag + ".json"
    if _os3.path.exists(_path) and _os3.path.exists(_meta_path):
        _m = _json3.load(open(_meta_path))
        if _m.get("max_len") == cfg["max_len"] and _m.get("commit") == cfg["commits"].get(tag):
            return "skipped"

    _t0 = _time3.time()
    _tk = _ATok.from_pretrained(repo, revision=revision)
    _tk.truncation_side = "left"
    _tk.padding_side = "right"
    if _tk.pad_token is None:
        _tk.pad_token = _tk.eos_token
    _texts = [cfg["scaffold"].format(prompt=_p) for _p in cfg["prompts"]]
    # untruncated lengths, for the P7 truncation audit
    _raw_len = [
        len(_tk(_t, add_special_tokens=False)["input_ids"]) for _t in _texts
    ]
    _n_trunc = int(sum(1 for _l in _raw_len if _l > cfg["max_len"]))
    _t_dl = _time3.time() - _t0

    _model = _AMLM.from_pretrained(repo, revision=revision, dtype=_torch3.bfloat16)
    _model = _model.to("cuda").eval()
    _t_load = _time3.time() - _t0

    _LY = list(cfg["layers"])
    _last = {_L: [] for _L in _LY}
    _mean = {_L: [] for _L in _LY}
    _bs = int(cfg["batch_size"])
    _t1 = _time3.time()
    for _i in range(0, len(_texts), _bs):
        _enc = _tk(
            _texts[_i : _i + _bs], return_tensors="pt", padding=True,
            truncation=True, max_length=cfg["max_len"], add_special_tokens=False,
        )
        _enc = {_k: _v.to("cuda") for _k, _v in _enc.items()}
        with _torch3.no_grad():
            _out = _model(**_enc, output_hidden_states=True)
        _mask = _enc["attention_mask"]
        _lastpos = _mask.sum(dim=1) - 1
        _rows = _torch3.arange(_mask.size(0), device="cuda")
        for _L in _LY:
            _hs = _out.hidden_states[_L]
            _last[_L].append(_hs[_rows, _lastpos].float().cpu())
            _mm = _mask.unsqueeze(-1).to(_hs.dtype)
            _mean[_L].append(((_hs * _mm).sum(dim=1) / _mm.sum(dim=1)).float().cpu())
        del _out
    _elapsed = _time3.time() - _t1
    _last = {_L: _torch3.cat(_last[_L]) for _L in _LY}
    _mean = {_L: _torch3.cat(_mean[_L]) for _L in _LY}

    del _model
    _gc.collect()
    _torch3.cuda.empty_cache()
    _freed = e2_purge(repo, revision)

    _torch3.save({"last": _last, "mean": _mean}, _path)

    # P7 degenerate-layer guard: mass-mean direction norm relative to activation
    # scale. A layer whose direction norm is ~0 must not enter a cosine average.
    _y = _np3.array(cfg["labels"])
    _norms = {}
    for _pool, _dd in [("last", _last), ("mean", _mean)]:
        for _L in _LY:
            _X = _dd[_L].numpy()
            _d = _X[_y == 1].mean(0) - _X[_y == 0].mean(0)
            _norms[_pool + "_L" + str(_L)] = {
                "dir_norm": float(_np3.linalg.norm(_d)),
                "act_norm": float(_np3.linalg.norm(_X, axis=1).mean()),
                "rel": float(_np3.linalg.norm(_d) / (_np3.linalg.norm(_X, axis=1).mean() + 1e-9)),
            }
    _degenerate = [_k for _k, _v in _norms.items() if _v["rel"] < 0.02]

    _meta = {
        "tag": tag, "repo": repo, "revision": revision,
        "commit": cfg["commits"].get(tag), "split_fp": cfg["split_fp"],
        "seed": cfg["seed"], "max_len": cfg["max_len"],
        "truncation_side": "left", "padding_side": "right",
        "add_special_tokens": False, "format": "neutral_scaffold",
        "n_prompts": len(_texts), "n_per_pool": cfg["n_per_pool"],
        "n_truncated": _n_trunc,
        "frac_truncated": round(_n_trunc / len(_texts), 5),
        "median_tokens": int(_np3.median(_raw_len)),
        "max_tokens": int(max(_raw_len)),
        "download_s": round(_t_dl, 1), "load_s": round(_t_load, 1),
        "extract_s": round(_elapsed, 1), "cache_freed_bytes": int(_freed),
        "direction_norms": _norms, "degenerate_layers": _degenerate,
        "deviations": cfg["deviations"],
    }
    _tmp = _meta_path + ".tmp"
    with open(_tmp, "w") as _fh:
        _json3.dump(_meta, _fh, indent=1)
    _os3.replace(_tmp, _meta_path)
    return "ok"


def e2_extract_worker(cfg):
    """Detached body. cfg is a THREAD ARGUMENT (P10), never a notebook global."""
    _prog = cfg["dir"] + "/_extract_progress.json"
    _state = {"started": _time3.time(), "n_total": len(cfg["plan"]),
              "done": [], "failed": {}, "current": None}

    def _flush():
        _tmp = _prog + ".tmp"
        with open(_tmp, "w") as _fh:
            _json3.dump(_state, _fh)
        _os3.replace(_tmp, _prog)

    _flush()
    for _row in cfg["plan"]:
        _tag, _repo, _rev = _row[0], _row[1], _row[2]
        _state["current"] = _tag
        _flush()
        try:
            _st = e2_extract_one(cfg, _tag, _repo, _rev)
            _state["done"].append([_tag, _st, round(_time3.time() - _state["started"], 1)])
        except Exception as _e:
            _state["failed"][_tag] = type(_e).__name__ + ": " + str(_e)[:400]
        _flush()
    _state["current"] = None
    _state["finished"] = _time3.time()
    _flush()


def e2_launch_extract(cfg):
    _t = _th3.Thread(
        target=e2_extract_worker, args=(dict(cfg),),
        name="e2-extract-" + cfg["split_fp"], daemon=True,
    )
    _t.start()
    return _t


E2_EXTRACT_READY = {"n_ckpt": len(E2_CFG["plan"]), "split_fp": E2_CFG["split_fp"]}
E2_EXTRACT_READY
