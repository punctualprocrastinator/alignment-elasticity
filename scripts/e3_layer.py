"""E3 - layer robustness (Box A, attack (iii): only layer 20).

Extract base train activations at layers {8,12,16,20,24,28}, refit the base
diff-in-means refusal direction PER LAYER, and compute steering efficacy +
margin for base / sft-1000 / instruct at each layer (steering applied at that
layer with that layer's base direction). Efficacy uses the paper definition
(OLS slope of mean refusal-gap displacement vs |c|, |c|<=2). Margin is the mean
unsteered refusal gap (layer-independent, computed once per checkpoint).

Detached-thread safe (P10). ASCII only.
"""

import gc
import json
import os
import time

import numpy as np
import torch

import pipeline as P
import gate_a as ga

E3_ART = "/marimo/e3"
GA_ART = "/marimo/gateA"

LAYERS = [8, 12, 16, 20, 24, 28]
MAX_LEN = ga.MAX_LEN
SEED = ga.SEED
BATCH = ga.BATCH

# Reduced c-grid: only the near-zero region needed for efficacy (|c|<=2),
# matching the |c|<=2 subset of the Gate A grid so the slope is comparable.
C_GRID = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]

E3_CKPTS = [
    ("base",     "allenai/Olmo-3-1025-7B",     "main", "a81bae42db3975be1671e27b9c9a56da1a9f980f"),
    ("sft-1000", "allenai/Olmo-3-7B-Think-SFT", "step1000", "9a45447cd55efd41e6eada0da28407277e818c63"),
    ("instruct", "allenai/Olmo-3-7B-Instruct",  "main", "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"),
]


def paths(art=E3_ART):
    P.ensure_dir(art)
    return {
        "art": art,
        "log": os.path.join(art, "log.txt"),
        "status": os.path.join(art, "status.json"),
        "dirs": os.path.join(art, "e3_base_dirs.npz"),
        "acts": os.path.join(art, "base_train_acts.npz"),
        "model": lambda lab: os.path.join(art, "e3_model_%s.json" % lab),
        "final": os.path.join(art, "E3_layer_robustness.json"),
    }


def efficacy_from_gaps(gaps, c_values, zero_index, cmax=2.0):
    gaps = np.asarray(gaps, dtype=np.float64)
    g0 = gaps[zero_index]
    cs = np.asarray(c_values, dtype=np.float64)
    xs, ys = [], []
    for i, cv in enumerate(cs):
        if abs(cv) <= cmax:
            xs.append(abs(cv))
            ys.append(float((g0 - gaps[i]).mean()))
    xs = np.asarray(xs); ys = np.asarray(ys)
    o = np.argsort(xs, kind="stable")
    slope, intercept = np.polyfit(xs[o], ys[o], 1)
    return {"efficacy": float(slope), "intercept": float(intercept),
            "n_points": int(xs.size)}


def mu_per_layer(model, tok, prompts, layers, batch_size=BATCH, max_len=MAX_LEN):
    """Mean all-token residual-stream L2 norm at each of `layers`, one pass."""
    blocks = P.decoder_layers(model)
    store = {}
    handles = []
    for L in layers:
        def mk(L):
            def hook(_m, _i, out):
                store[L] = P.hidden_of(out).detach()
            return hook
        handles.append(blocks[int(L)].register_forward_hook(mk(L)))
    device = next(model.parameters()).device
    tot = {L: 0.0 for L in layers}
    cnt = 0
    try:
        for s in range(0, len(prompts), batch_size):
            chunk = prompts[s:s + batch_size]
            enc, _last, _raw = P.encode_batch(tok, chunk, max_len=max_len,
                                              device=device, side="right")
            with torch.no_grad():
                model(**enc)
            mask = enc["attention_mask"].bool()
            for L in layers:
                nrm = store[L].float().norm(dim=-1)
                tot[L] += float(nrm[mask].sum())
            cnt += int(mask.sum())
            store.clear()
    finally:
        P.remove_hooks(handles)
    return {L: tot[L] / max(1, cnt) for L in layers}


def e3_worker(art=E3_ART, ga_art=GA_ART, ckpts=None, layers=None, batch_size=BATCH):
    ckpts = ckpts or E3_CKPTS
    layers = layers or LAYERS
    pth = paths(art)
    ga_pth = ga.paths(ga_art)

    def log(msg):
        return P.elog(pth["log"], msg)

    t_all = time.time()
    status = {"stage": "start", "t0": t_all, "done": [], "error": None}
    P.write_json(pth["status"], status)
    try:
        P.set_seed(SEED)
        data = P.load_prompts(seed=ga.SPLIT_SEED, cache_path=ga_pth["prompts"])
        fp = ga.split_fingerprint(data)
        log("=== E3 start; split fingerprint %s; layers=%s" % (fp, layers))
        if fp != "99a7ac88967302166d6e1698d1eebae8d2fd9576":
            raise RuntimeError("split fingerprint mismatch: %s (STOP)" % fp)
        harm_held = data["harm_held"]
        zero_index = C_GRID.index(0.0)

        # Fit base diff-in-means direction per layer (needs base model once).
        dirs = None
        if os.path.exists(pth["dirs"]):
            with np.load(pth["dirs"]) as b:
                dirs = {int(k.split("_")[1]): np.asarray(b[k], dtype=np.float32)
                        for k in b.files if k.startswith("L_")}
            if set(dirs.keys()) != set(layers):
                dirs = None

        for label, repo, branch, commit in ckpts:
            out_path = pth["model"](label)
            if os.path.exists(out_path) and json.load(open(out_path)).get("complete"):
                log("skip %s (complete)" % label)
                status["done"].append(label); P.write_json(pth["status"], status)
                continue

            status["stage"] = "load:" + label; P.write_json(pth["status"], status)
            t0 = time.time()
            log("loading %s @ %s (%s)" % (repo, branch, commit[:10]))
            tok = P.load_tokenizer(repo, revision=commit)
            model = P.load_model(repo, revision=commit)
            log("loaded %s in %.1fs" % (label, time.time() - t0))
            r_ids, _rs, _rd = P.onset_token_ids(tok, P.REFUSAL_STRS)
            c_ids, _cs, _cd = P.onset_token_ids(tok, P.COMPLY_STRS)

            if label == "base" and dirs is None:
                status["stage"] = "fit_dirs"; P.write_json(pth["status"], status)
                fit_prompts = list(data["harm_train"]) + list(data["ben_train"])
                fit_labels = np.array([1]*len(data["harm_train"]) + [0]*len(data["ben_train"]))
                pack = P.extract_activations(model, tok, fit_prompts, P.PROBE_LAYERS,
                                             cache_path=pth["acts"], batch_size=batch_size)
                dirs = {}
                for L in layers:
                    li = P.PROBE_LAYERS.index(L)
                    v, nrm = P.refusal_direction(pack["acts"], fit_labels, layer_index=li)
                    dirs[L] = np.asarray(v, dtype=np.float32)
                    log("fit base dir L%d (massmean raw norm %.3f)" % (L, nrm))
                np.savez_compressed(pth["dirs"], **{"L_%d" % L: dirs[L] for L in layers})
                del pack; gc.collect()
            if dirs is None:
                raise RuntimeError("base directions missing; include base first")

            status["stage"] = "mu:" + label; P.write_json(pth["status"], status)
            mus = mu_per_layer(model, tok, harm_held, layers, batch_size=batch_size)
            log("%s mu per layer: %s" % (label, {L: round(mus[L],2) for L in layers}))

            rec = {"label": label, "repo": repo, "branch": branch, "commit": commit,
                   "split_fingerprint": fp, "seed": SEED, "layers": layers,
                   "c_grid": C_GRID, "zero_index": zero_index, "mu_by_layer": mus,
                   "per_layer": {}}
            margin = None
            for L in layers:
                status["stage"] = "sweep:%s:L%d" % (label, L); P.write_json(pth["status"], status)
                t0 = time.time()
                gaps = ga.swept_gaps(model, tok, harm_held, dirs[L], mus[L], C_GRID,
                                     r_ids, c_ids, layer=L, batch_size=batch_size, log=None)
                if margin is None:
                    margin = float(gaps[zero_index].mean())
                eff = efficacy_from_gaps(gaps, C_GRID, zero_index)
                rec["per_layer"][str(L)] = {"layer": L, "mu": mus[L],
                    "efficacy": eff["efficacy"], "intercept": eff["intercept"],
                    "n_points": eff["n_points"], "gaps": gaps.tolist()}
                log("%s L%2d efficacy=%.4f (%.1fs)" % (label, L, eff["efficacy"], time.time()-t0))
            rec["margin"] = margin
            rec["complete"] = True
            P.write_json(out_path, rec)
            log("%s DONE margin=%+.3f eff_by_layer=%s" % (label, margin,
                {L: round(rec["per_layer"][str(L)]["efficacy"],3) for L in layers}))
            status["done"].append(label); P.write_json(pth["status"], status)
            P.free_model(model); del tok
            freed = P.purge_hf_cache(repo)
            log("%s freed %.2f GB" % (label, freed/1e9))

        status["stage"] = "aggregate"; P.write_json(pth["status"], status)
        aggregate(art)
        status["stage"] = "done"; status["total_seconds"] = time.time()-t_all
        P.write_json(pth["status"], status)
        log("=== E3 done in %.1fs" % (time.time()-t_all))
    except Exception as exc:
        import traceback
        status["stage"] = "error"; status["error"] = repr(exc)
        P.write_json(pth["status"], status)
        P.elog(pth["log"], "ERROR " + repr(exc)); P.elog(pth["log"], traceback.format_exc())


def aggregate(art=E3_ART):
    pth = paths(art)
    labels = [lab for lab, *_ in E3_CKPTS]
    models = {}
    for lab in labels:
        p = pth["model"](lab)
        if os.path.exists(p) and json.load(open(p)).get("complete"):
            models[lab] = json.load(open(p))
    eff = {lab: {int(L): models[lab]["per_layer"][str(L)]["efficacy"] for L in LAYERS}
           for lab in models}
    # per-layer CV across checkpoints
    layer_cv = {}
    for L in LAYERS:
        vals = np.array([eff[lab][L] for lab in models], dtype=np.float64)
        m = float(vals.mean())
        layer_cv[int(L)] = {"mean": m, "sd": float(vals.std(ddof=1)) if vals.size>1 else 0.0,
                            "cv": float(vals.std(ddof=1)/m) if m else None,
                            "by_model": {lab: eff[lab][L] for lab in models}}
    out = {
        "experiment": "E3 - layer robustness of steering efficacy",
        "definition": {"efficacy": "OLS slope of mean refusal-gap displacement vs |c|, |c|<=2",
                       "margin": "mean unsteered refusal gap (layer-independent)"},
        "layers": LAYERS, "checkpoints": list(models.keys()),
        "split_fingerprint": (list(models.values())[0]["split_fingerprint"] if models else None),
        "margin": {lab: models[lab]["margin"] for lab in models},
        "efficacy_by_model_layer": {lab: {str(L): eff[lab][L] for L in LAYERS} for lab in models},
        "layer_cv_across_checkpoints": layer_cv,
    }
    P.write_json(pth["final"], out)
    return out


def e3_launch(art=E3_ART, **kw):
    import threading
    for th in threading.enumerate():
        if th.name == "e3-worker" and th.is_alive():
            return {"launched": False, "reason": "e3-worker already alive"}
    P.run_detached(lambda: e3_worker(art=art, **kw), name="e3-worker")
    return {"launched": True}
