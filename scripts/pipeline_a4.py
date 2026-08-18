"""A4 - 32B scale replication of the refusal-direction elasticity result.

Answers the "one lineage, one scale" objection by re-running the E2-clean
measurement on the Olmo-3 32B flow. Builds on pipeline.py. ASCII only.

Protocol v1 bindings:
  P1   SEED = 42, SHA-1 split fingerprint recorded in the result file.
  P1b  logistic probes fit at tol=1e-10 (sklearn's default 1e-4 stops LBFGS
       after ~14 iterations and leaves the DIRECTION at cosine 0.978 to the
       optimum - fatal when the headline metric IS a direction cosine).
  P2   500 per pool (see deviations), 1000 bootstrap resamples.
  P3   shuffled-label null with an INDEPENDENT permutation per
       (layer, pooling, seed); random-direction floor recomputed for R^5120.
  P5   both last-token and mean pooling everywhere.
  P7   MAX_LEN=384, left truncation, last non-pad token via attention mask,
       truncation fraction logged.
  P9   revisions pinned by branch AND commit; safetensors hashed.
  P10  detached thread, config as arguments, launch guarded against duplicate
       threads, idempotent and disk-backed, HF cache purged per checkpoint.

Scale notes: 32B is 64 layers, d_model 5120, ~64.5 GB of bf16 weights. The box
has 160 GB RAM and a ~102 GB card, so a plain CPU load followed by .to("cuda")
fits, but batch sizes are kept well below the 7B settings and the layer grid is
proportional (8/64 .. 63/64) rather than reusing the 7B indices.
"""

import hashlib
import json
import os
import threading
import time

import numpy as np
import torch

import pipeline as pl

SEED_A4 = 42
A4_DIR = "/marimo/a4"
A4_LAYERS = [8, 16, 24, 32, 40, 48, 56, 63]
A4_POOLINGS = ["last", "mean"]
N_PER_POOL = 500

# Pinned by branch AND commit (P9). SFT main is byte-distinct from BOTH
# learning-rate lineages, so the cliff contrast stays inside the 5e-5 run.
CKPTS_A4 = [
    ("allenai/Olmo-3-1125-32B", "main", "be0ef6a4f009aa7a", "base", 0),
    ("allenai/Olmo-3-32B-Think-SFT", "5e-5-step1000", "e7292f17021e01fa", "sft-1k", 1),
    ("allenai/Olmo-3-32B-Think-SFT", "5e-5-step10790", "91c593ba94cf300e", "sft-10790", 2),
    ("allenai/Olmo-3-32B-Think-DPO", "main", "f65f29d56a00b876", "dpo", 4),
    ("allenai/Olmo-3-32B-Think", "step_050", "636bbe53a3f164f1", "rlvr-50", 5),
    ("allenai/Olmo-3-32B-Think", "step_750", "211d82786f4babdd", "rlvr-750", 6),
    ("allenai/Olmo-3-32B-Think-SFT", "main", "e6282dbc427d27de", "sft-main", 3),
]


def split_fp(items):
    return hashlib.sha1("|".join(str(x) for x in items).encode()).hexdigest()[:12]


def load_contrast(n_per_pool=N_PER_POOL, seed=SEED_A4, cache_path=None):
    """Harmful = AdvBench goals, benign = alpaca rows with empty input.

    WildGuardMix is gated and no HF_TOKEN is present on this box, so the
    adversarial split is unavailable; the direction work does not need it.
    """
    cache_path = cache_path or os.path.join(A4_DIR, "contrast.json")
    pl.ensure_dir(os.path.dirname(cache_path))
    if os.path.exists(cache_path):
        return pl.read_json(cache_path)
    import io
    import random as _random

    import pandas as pd
    import requests

    adv = pd.read_csv(io.StringIO(requests.get(pl.HARM_URL, timeout=120).text))
    harm = [str(g).strip() for g in adv["goal"].tolist() if str(g).strip()]
    from datasets import load_dataset

    alp = load_dataset("tatsu-lab/alpaca", split="train")
    ben = [str(i).strip() for i, j in zip(alp["instruction"], alp["input"])
           if str(j).strip() == "" and str(i).strip()]
    rng = _random.Random(seed)
    rng.shuffle(harm)
    rng.shuffle(ben)
    if len(harm) < n_per_pool:
        raise RuntimeError("harmful pool too small: %d" % len(harm))
    blob = {
        "source_harmful": pl.HARM_URL, "source_benign": "tatsu-lab/alpaca",
        "seed": seed, "n_per_pool": n_per_pool,
        "harmful": harm[:n_per_pool], "benign": ben[:n_per_pool],
        "harm_pool": len(harm), "ben_pool": len(ben),
    }
    blob["fingerprint"] = split_fp(blob["harmful"] + blob["benign"])
    pl.write_json(cache_path, blob)
    return blob


def extract_both_poolings(model, tokenizer, prompts, layers, cache_path=None,
                          batch_size=8, max_len=pl.MAX_LEN):
    """Last-token AND mean-pooled residual activations in one forward pass (P5).

    Mean pooling averages over non-pad positions only, using the attention mask;
    last-token indexes the last non-pad position (P7).
    """
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path) as b:
            return {"last": b["last"], "mean": b["mean"], "layers": b["layers"].tolist(),
                    "lengths": b["lengths"], "n_trunc": int(b["n_trunc"]),
                    "frac_trunc": float(b["frac_trunc"])}
    layers = [int(x) for x in layers]
    blocks = pl.decoder_layers(model)
    store, handles = {}, []

    def mk(li):
        def hook(_m, _i, out):
            store[li] = pl.hidden_of(out).detach()
        return hook

    for li in layers:
        handles.append(blocks[li].register_forward_hook(mk(li)))
    d = int(model.config.hidden_size)
    out_last = np.zeros((len(layers), len(prompts), d), dtype=np.float32)
    out_mean = np.zeros((len(layers), len(prompts), d), dtype=np.float32)
    lengths = []
    device = next(model.parameters()).device
    try:
        for s in range(0, len(prompts), batch_size):
            chunk = prompts[s : s + batch_size]
            enc, last_idx, raw = pl.encode_batch(tokenizer, chunk, max_len=max_len,
                                                 device=device, side="right")
            lengths.extend(raw)
            with torch.no_grad():
                model(**enc)
            mask = enc["attention_mask"].to(torch.float32)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            rows = torch.arange(len(chunk), device=device)
            for j, li in enumerate(layers):
                h = store[li].float()
                out_last[j, s : s + len(chunk)] = h[rows, last_idx].cpu().numpy()
                out_mean[j, s : s + len(chunk)] = (
                    (h * mask.unsqueeze(-1)).sum(dim=1) / denom).cpu().numpy()
            store.clear()
            del enc
    finally:
        for h in handles:
            h.remove()
    lengths = np.asarray(lengths, dtype=np.int32)
    n_trunc = int((lengths >= max_len).sum())
    res = {"last": out_last, "mean": out_mean, "layers": layers, "lengths": lengths,
           "n_trunc": n_trunc, "frac_trunc": float(n_trunc) / max(1, lengths.size)}
    if cache_path:
        np.savez_compressed(cache_path, last=out_last, mean=out_mean,
                            layers=np.asarray(layers), lengths=lengths,
                            n_trunc=n_trunc, frac_trunc=res["frac_trunc"])
    return res


def a4_worker(art_dir=A4_DIR, ckpts=None, layers=None, n_per_pool=N_PER_POOL,
              batch_size=8, max_len=pl.MAX_LEN, seed=SEED_A4):
    """Phase A: activations only. Models are huge, so every checkpoint is
    reduced to a ~330 MB activation cache and then purged; all statistics are
    computed afterwards from disk with no GPU."""
    ckpts = ckpts or CKPTS_A4
    layers = layers or A4_LAYERS
    pl.ensure_dir(art_dir)
    log_path = os.path.join(art_dir, "log.txt")
    status_path = os.path.join(art_dir, "status.json")

    def log(m):
        return pl.elog(log_path, m)

    t_all = time.time()
    status = {"phase": "a4", "stage": "start", "done": [], "error": None, "cost": {}}
    pl.write_json(status_path, status)
    try:
        pl.set_seed(seed)
        data = load_contrast(n_per_pool=n_per_pool, seed=seed,
                             cache_path=os.path.join(art_dir, "contrast.json"))
        prompts = list(data["harmful"]) + list(data["benign"])
        log("=== A4 start; %d ckpts, %d prompts, layers=%s, fp=%s"
            % (len(ckpts), len(prompts), layers, data["fingerprint"]))
        for repo, rev, fp8, label, order in ckpts:
            out_path = os.path.join(art_dir, "acts_%s.npz" % label)
            if os.path.exists(out_path):
                log("skip %s (exists)" % label)
                status["done"].append(label)
                pl.write_json(status_path, status)
                continue
            status["stage"] = "load:" + label
            pl.write_json(status_path, status)
            t0 = time.time()
            tok = pl.load_tokenizer(repo, revision=rev)
            model = pl.load_model(repo, revision=rev)
            t_load = time.time() - t0
            log("loaded %s (%s@%s) in %.1fs; layers=%d d_model=%d"
                % (label, repo, rev, t_load, len(pl.decoder_layers(model)),
                   model.config.hidden_size))
            status["stage"] = "extract:" + label
            pl.write_json(status_path, status)
            t1 = time.time()
            pack = extract_both_poolings(model, tok, prompts, layers,
                                         cache_path=out_path, batch_size=batch_size,
                                         max_len=max_len)
            t_ext = time.time() - t1
            log("%s activations %s in %.1fs; trunc %d/%d"
                % (label, str(pack["last"].shape), t_ext, pack["n_trunc"], len(prompts)))
            pl.free_model(model)
            del tok, pack
            t2 = time.time()
            freed = pl.purge_hf_cache(repo)
            status["cost"][label] = {"load_s": round(t_load, 1), "extract_s": round(t_ext, 1),
                                     "purge_s": round(time.time() - t2, 1),
                                     "gb": round(freed / 1e9, 2)}
            status["done"].append(label)
            pl.write_json(status_path, status)
            log("%s done; purged %.2f GB in %.1fs (total %.1f min)"
                % (label, freed / 1e9, time.time() - t2, (time.time() - t_all) / 60))
        status["stage"] = "done"
        status["total_seconds"] = time.time() - t_all
        pl.write_json(status_path, status)
        log("=== A4 activations done in %.1f min" % ((time.time() - t_all) / 60))
    except Exception as exc:
        import traceback
        status["stage"] = "error"
        status["error"] = repr(exc)
        pl.write_json(status_path, status)
        pl.elog(log_path, "ERROR " + repr(exc))
        pl.elog(log_path, traceback.format_exc())


def a4_launch(art_dir=A4_DIR, **kw):
    """Guarded launcher (P10): refuses to start a second thread while one is
    alive, and refuses to start at all when every artifact already exists.
    Re-running this cell is therefore safe."""
    for t in threading.enumerate():
        if t.name == "a4-worker" and t.is_alive():
            return {"launched": False, "reason": "a4-worker already alive"}
    ckpts = kw.get("ckpts") or CKPTS_A4
    missing = [lab for _r, _v, _f, lab, _o in ckpts
               if not os.path.exists(os.path.join(art_dir, "acts_%s.npz" % lab))]
    if not missing:
        return {"launched": False, "reason": "all activation caches present"}
    th = pl.run_detached(lambda: a4_worker(art_dir=art_dir, **kw), name="a4-worker")
    return {"launched": True, "missing": missing, "thread": th.name}


# ---------------------------------------------------------------- statistics


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def dim_direction(acts, labels):
    """Difference-in-means direction (unit)."""
    a = np.asarray(acts, dtype=np.float64)
    lab = np.asarray(labels)
    return unit(a[lab == 1].mean(axis=0) - a[lab == 0].mean(axis=0))


def boot_count_matrix(n, n_boot, rng):
    """Multinomial count matrix so bootstrap means are a single BLAS matmul
    instead of materialising an [n_boot, n, d] gather."""
    idx = rng.integers(0, n, size=(n_boot, n))
    counts = np.zeros((n_boot, n), dtype=np.float32)
    for b in range(n_boot):
        np.add.at(counts[b], idx[b], 1.0)
    return counts


def paired_cos_ci(acts_a, acts_b, labels, n_boot=1000, seed=SEED_A4, alpha=0.05):
    """CI for cos(dir(acts_a), dir(acts_b)) using the SAME resampled prompt
    indices on both checkpoints - they are measured on identical prompts."""
    lab = np.asarray(labels)
    rng = np.random.default_rng(seed)
    hi_, bi_ = np.where(lab == 1)[0], np.where(lab == 0)[0]
    ch = boot_count_matrix(hi_.size, n_boot, rng)
    cb = boot_count_matrix(bi_.size, n_boot, rng)
    out = np.empty(n_boot, dtype=np.float64)
    A = np.asarray(acts_a, dtype=np.float32)
    B = np.asarray(acts_b, dtype=np.float32)
    da = (ch @ A[hi_]) / hi_.size - (cb @ A[bi_]) / bi_.size
    db = (ch @ B[hi_]) / hi_.size - (cb @ B[bi_]) / bi_.size
    da /= (np.linalg.norm(da, axis=1, keepdims=True) + 1e-12)
    db /= (np.linalg.norm(db, axis=1, keepdims=True) + 1e-12)
    out = (da * db).sum(axis=1)
    point = float(np.dot(dim_direction(A, lab), dim_direction(B, lab)))
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def random_cos_floor(d_model, k=20, seed=SEED_A4):
    """Noise floor for a cosine in R^d (P10). MUST be recomputed per d_model:
    7B used R^4096 (|cos| ~ 0.0117); 32B is R^5120."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(k, d_model))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cs = []
    for i in range(k):
        for j in range(i + 1, k):
            cs.append(abs(float(np.dot(v[i], v[j]))))
    cs = np.asarray(cs)
    return {"abs_cos_mean": float(cs.mean()), "abs_cos_p95": float(np.percentile(cs, 95)),
            "abs_cos_max": float(cs.max()), "n_pairs": int(cs.size),
            "d_model": int(d_model), "k": int(k),
            "analytic_sqrt_2_over_pi_d": float(np.sqrt(2.0 / (np.pi * d_model)))}


# ---------------------------------------------------------------- probes


def fit_probe(X, y, seed=SEED_A4, tol=1e-10, max_iter=5000, C=1.0):
    """P1b: tol=1e-10. sklearn's default 1e-4 stops LBFGS ~14 iterations short
    and leaves the fitted DIRECTION at cosine ~0.978 to the true optimum."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, tol=tol, max_iter=max_iter, solver="lbfgs",
                             random_state=seed)
    clf.fit(np.asarray(X, dtype=np.float64), np.asarray(y))
    return clf


def probe_auroc(clf, X, y):
    s = np.asarray(X, dtype=np.float64) @ clf.coef_[0] + clf.intercept_[0]
    return _auroc(s[np.asarray(y) == 1], s[np.asarray(y) == 0]), s


def _auroc(pos, neg):
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    if p.size == 0 or n.size == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    order = allv.argsort()
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1, dtype=np.float64)
    srt = allv[order]
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[:p.size].sum() - p.size * (p.size + 1) / 2.0) / (p.size * n.size))


def train_test_idx(n_per_pool, seed=SEED_A4, frac=0.5):
    """Half of each pool trains the probe, half evaluates it."""
    rng = np.random.default_rng(seed)
    h = rng.permutation(n_per_pool)
    b = rng.permutation(n_per_pool) + n_per_pool
    k = int(n_per_pool * frac)
    tr = np.concatenate([h[:k], b[:k]])
    te = np.concatenate([h[k:], b[k:]])
    return tr, te


def eval_boot_auroc_pair(s_frozen, s_refit, y, n_boot=1000, seed=SEED_A4, alpha=0.05):
    """Bootstrap the EVALUATION set only (no refit), so frozen and refit AUROC
    and their paired difference share resampled indices."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    fr, rf, gp = [], [], []
    for _ in range(n_boot):
        a = rng.integers(0, pi.size, pi.size)
        b = rng.integers(0, ni.size, ni.size)
        p_, n_ = pi[a], ni[b]
        f = _auroc(s_frozen[p_], s_frozen[n_])
        r = _auroc(s_refit[p_], s_refit[n_])
        fr.append(f), rf.append(r), gp.append(r - f)
    q = lambda v: (float(np.percentile(v, 100 * alpha / 2)),
                   float(np.percentile(v, 100 * (1 - alpha / 2))))
    return {
        "auroc_frozen": _auroc(s_frozen[y == 1], s_frozen[y == 0]),
        "auroc_frozen_ci": list(q(fr)),
        "auroc_refit": _auroc(s_refit[y == 1], s_refit[y == 0]),
        "auroc_refit_ci": list(q(rf)),
        "gap": float(np.mean(gp)), "gap_ci": list(q(gp)),
        "gap_frac_le0": float(np.mean(np.asarray(gp) <= 0)),
    }


def shuffled_null_auroc(X_tr, y_tr, X_te, y_te, n_perm=3, seed=SEED_A4, tol=1e-10):
    """P3: an INDEPENDENT permutation per (layer, pooling, seed). A single
    shared permutation inflated a sibling's null band to 0.658."""
    out = []
    for s in range(n_perm):
        rng = np.random.default_rng(seed + 9173 * (s + 1))
        yp = rng.permutation(np.asarray(y_tr))
        clf = fit_probe(X_tr, yp, seed=seed + s, tol=tol)
        a, _ = probe_auroc(clf, X_te, y_te)
        out.append(float(a))
    return out


def a4_analyze(art_dir=A4_DIR, ckpts=None, layers=None, n_per_pool=N_PER_POOL,
               seed=SEED_A4, n_boot=1000, n_boot_refit=100, n_perm=3,
               anchor="base", tol=1e-10):
    """Phase B: all statistics from the cached activations. No GPU, no models.

    Headline metric is the difference-in-means direction cosine to the base
    anchor, which gets the full 1000-resample paired bootstrap. The probe
    direction cosine is a secondary operationalisation; its CI needs a fresh
    logistic fit per resample, so it is bootstrapped only at the best layer of
    each pooling (see deviations).
    """
    ckpts = ckpts or CKPTS_A4
    layers = layers or A4_LAYERS
    log_path = os.path.join(art_dir, "analyze_log.txt")

    def log(m):
        return pl.elog(log_path, m)

    t0 = time.time()
    labels = np.array([1] * n_per_pool + [0] * n_per_pool)
    tr, te = train_test_idx(n_per_pool, seed=seed)
    packs = {}
    for _r, _v, _f, lab, _o in ckpts:
        p = os.path.join(art_dir, "acts_%s.npz" % lab)
        if not os.path.exists(p):
            log("MISSING acts for %s" % lab)
            continue
        with np.load(p) as b:
            packs[lab] = {"last": b["last"], "mean": b["mean"],
                          "n_trunc": int(b["n_trunc"]), "layers": b["layers"].tolist()}
    if anchor not in packs:
        raise RuntimeError("anchor %s missing" % anchor)
    d_model = packs[anchor]["last"].shape[2]
    floor = random_cos_floor(d_model, k=20, seed=seed)
    log("analyzing %d ckpts, d_model=%d, floor |cos|=%.5f"
        % (len(packs), d_model, floor["abs_cos_mean"]))

    base_probe, best_layer = {}, {}
    for pool in A4_POOLINGS:
        for li, L in enumerate(layers):
            X = packs[anchor][pool][li]
            base_probe[(pool, L)] = fit_probe(X[tr], labels[tr], seed=seed, tol=tol)
        aur = {}
        for li, L in enumerate(layers):
            X = packs[anchor][pool][li]
            a, _ = probe_auroc(base_probe[(pool, L)], X[te], labels[te])
            aur[L] = a
        best_layer[pool] = max(aur, key=aur.get)
        log("anchor probes fitted for pooling=%s; best layer %d (AUROC %.3f)"
            % (pool, best_layer[pool], aur[best_layer[pool]]))

    rows = []
    for repo, rev, fp8, lab, order in ckpts:
        if lab not in packs:
            continue
        for pool in A4_POOLINGS:
            for li, L in enumerate(layers):
                Xa = packs[anchor][pool][li]
                Xc = packs[lab][pool][li]
                cos_d, lo_d, hi_d = paired_cos_ci(Xc, Xa, labels, n_boot=n_boot, seed=seed)
                clf_c = fit_probe(Xc[tr], labels[tr], seed=seed, tol=tol)
                clf_b = base_probe[(pool, L)]
                cos_p = float(np.dot(unit(clf_c.coef_[0]), unit(clf_b.coef_[0])))
                s_fr = Xc[te] @ clf_b.coef_[0] + clf_b.intercept_[0]
                s_rf = Xc[te] @ clf_c.coef_[0] + clf_c.intercept_[0]
                m = eval_boot_auroc_pair(s_fr, s_rf, labels[te], n_boot=n_boot, seed=seed)
                nulls = shuffled_null_auroc(Xc[tr], labels[tr], Xc[te], labels[te],
                                            n_perm=n_perm, seed=seed + 31 * L, tol=tol)
                row = {"tag": lab, "repo": repo, "revision": rev, "commit_fp": fp8,
                       "order": order, "pooling": pool, "layer": int(L),
                       "cos_dim": cos_d, "cos_dim_ci": [lo_d, hi_d],
                       "cos_probe": cos_p, "cos_probe_ci": None,
                       "null_auroc": nulls, "null_auroc_mean": float(np.mean(nulls))}
                row.update(m)
                rows.append(row)
        log("%s analysed (%.1f min elapsed)" % (lab, (time.time() - t0) / 60))
        pl.write_json(os.path.join(art_dir, "rows_partial.json"), rows)

    if n_boot_refit > 0:
        rng = np.random.default_rng(seed)
        for row in rows:
            if row["layer"] != best_layer[row["pooling"]]:
                continue
            li = layers.index(row["layer"])
            Xa = packs[anchor][row["pooling"]][li]
            Xc = packs[row["tag"]][row["pooling"]][li]
            vals = []
            for _b in range(n_boot_refit):
                idx = rng.integers(0, tr.size, tr.size)
                sub = tr[idx]
                ca = fit_probe(Xa[sub], labels[sub], seed=seed, tol=tol)
                cc = fit_probe(Xc[sub], labels[sub], seed=seed, tol=tol)
                vals.append(float(np.dot(unit(cc.coef_[0]), unit(ca.coef_[0]))))
            row["cos_probe_ci"] = [float(np.percentile(vals, 2.5)),
                                   float(np.percentile(vals, 97.5))]
            log("cos_probe CI for %s/%s L%d done" % (row["tag"], row["pooling"], row["layer"]))

    out = {"rows": rows, "random_direction_reference": floor,
           "best_layer": best_layer, "d_model": int(d_model),
           "n_boot": n_boot, "n_boot_refit": n_boot_refit, "n_perm": n_perm,
           "seed": seed, "solver_tol": tol, "layers": list(layers),
           "anchor": anchor, "elapsed_s": time.time() - t0}
    pl.write_json(os.path.join(art_dir, "analysis.json"), out)
    log("=== analysis done in %.1f min" % ((time.time() - t0) / 60))
    return out


def a4_run_all(worker_kw=None, analyze_kw=None):
    a4_worker(**(worker_kw or {}))
    a4_analyze(**(analyze_kw or {}))


def a4_launch_all(worker_kw=None, analyze_kw=None):
    """Guarded (P10): one live a4-worker at most."""
    for t in threading.enumerate():
        if t.name == "a4-worker" and t.is_alive():
            return {"launched": False, "reason": "a4-worker already alive"}
    w = dict(worker_kw or {})
    a = dict(analyze_kw or {})
    th = pl.run_detached(lambda: a4_run_all(w, a), name="a4-worker")
    return {"launched": True, "thread": th.name}
