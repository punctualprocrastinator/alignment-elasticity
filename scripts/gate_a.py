"""Gate A: does the steering collapse survive boundary-matched dosing?

Carries the BASE model's layer-20 refusal direction to every checkpoint and
sweeps a steering coefficient using the persona-vector parameterisation
(arXiv 2605.13329):

    h_l <- h_l + c * mu_l * v_hat        mu_l = mean residual-stream norm

At every (model, direction, c) we record the per-prompt refusal logit gap on a
held-out harmful set, so downstream analysis can express crossing rate against
BOTH the input-side dose c and the ACHIEVED DISPLACEMENT of the gap.

ASCII only. Importable as a plain module so it outlives the sandbox.
"""

import gc
import hashlib
import json
import os
import time

import numpy as np
import torch

import pipeline as P


# --------------------------------------------------------------- configuration

SEED = 42          # protocol P1; pipeline.py's own SEED=20260805 is NOT used
SPLIT_SEED = 42    # prompt-shuffle seed, recorded with a fingerprint
FIT_LAYER = 20
STEER_LAYER = 20
MAX_LEN = 384

ART = "/marimo/gateA"

# Pinned by branch AND commit: `main` can differ from the last-step branch.
CKPTS = [
    ("base",     "allenai/Olmo-3-1025-7B",         "main",      "a81bae42db3975be1671e27b9c9a56da1a9f980f"),
    ("instruct", "allenai/Olmo-3-7B-Instruct",     "main",      "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"),
    ("rlz-math", "allenai/Olmo-3-7B-RL-Zero-Math", "step_1900", "8182367150cef52ddf00dd5259ea94eaa330918e"),
    ("rlz-code", "allenai/Olmo-3-7B-RL-Zero-Code", "step_2900", "ea18fc74cee487438aa052ea5bf8aada8a04f4af"),
]

# Negative c = toward compliance. The protocol's starting set
# {-4,-3,-2,-1.5,-1,-0.5,0,0.5,1} is a subset; the grid is densified because a
# coarse grid quantises the interpolated c_50/d_50 and inflates their CIs.
C_GRID = [
    -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5,
    -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0,
]
# Random-direction null swept over the identical dose range, matched norm.
C_GRID_NULL = list(C_GRID)
# Appended one at a time if a model has still not reached 50% crossing.
EXTRA_C = [-16.0, -20.0, -24.0, -32.0, -48.0]

N_TRAIN = 200
N_HELD = 200
N_RAND = 20
N_GEN = 40
GEN_NEW_TOKENS = 512   # RL-Zero checkpoints emit <think>; report unclosed frac
BATCH = 32
GEN_BATCH = 8


def paths(art=ART):
    P.ensure_dir(art)
    return {
        "art": art,
        "log": os.path.join(art, "log.txt"),
        "status": os.path.join(art, "status.json"),
        "prompts": os.path.join(art, "prompts.json"),
        "acts": os.path.join(art, "base_train_acts.npz"),
        "dirs": os.path.join(art, "directions.json"),
        "dirs_npz": os.path.join(art, "directions.npz"),
        "smoke": os.path.join(art, "smoke.json"),
        "sweep": lambda lab: os.path.join(art, "gateA_sweep_%s.json" % lab),
        "gen": lambda lab: os.path.join(art, "gateA_gen_%s.json" % lab),
    }


# --------------------------------------------------------------- fingerprints


def split_fingerprint(data):
    """SHA-1 over the exact prompt lists, so a result file can be rejected if
    the split silently moved (protocol P1)."""
    blob = json.dumps(
        {k: data[k] for k in ("harm_train", "ben_train", "harm_held", "ben_held")},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def provenance(data):
    return {
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "split_fingerprint": split_fingerprint(data),
        "n_train": len(data["harm_train"]),
        "n_held": len(data["harm_held"]),
        "max_len": MAX_LEN,
        "fit_layer": FIT_LAYER,
        "steer_layer": STEER_LAYER,
        "scaffold": "User: {p}" + chr(10) + "Assistant:",
        "refusal_strings": P.REFUSAL_STRS,
        "comply_strings": P.COMPLY_STRS,
    }


# --------------------------------------------------------------- directions


def fit_logistic_direction(acts2d, labels):
    """Logistic-regression direction at tol=1e-10.

    sklearn's default tol=1e-4 stops LBFGS at ~14 iterations and leaves the
    direction at cosine ~0.978 to the L2 optimum (protocol P1b). Returns
    (unit_direction, info).
    """
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(acts2d, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", tol=1e-10, max_iter=50000,
        fit_intercept=True, random_state=SEED,
    )
    clf.fit(X, y)
    w = np.asarray(clf.coef_[0], dtype=np.float64)
    nrm = float(np.linalg.norm(w))
    n_it = int(np.asarray(clf.n_iter_).ravel()[0])
    info = {
        "n_iter": n_it,
        "converged": bool(n_it < 50000),
        "train_acc": float(clf.score(X, y)),
        "raw_norm": nrm,
        "tol": 1e-10,
        "C": 1.0,
    }
    return (w / (nrm + 1e-12)).astype(np.float32), info


# --------------------------------------------------------------- steering


def steer_hook(model, direction, layer, coeff, mu):
    """h_l <- h_l + c * mu_l * v_hat at ONE decoder layer.

    This is 2605.13329's parameterisation verbatim, so the cross-checkpoint
    comparison is like-for-like on the input side. Olmo3 blocks return a bare
    Tensor, hence hidden_of/rewrap.
    """
    vec = torch.as_tensor(np.asarray(direction, dtype=np.float32))
    vec = vec / (vec.norm() + 1e-12)
    scale = float(coeff) * float(mu)

    def hook(_mod, _inp, out):
        hid = P.hidden_of(out)
        dv = vec.to(device=hid.device, dtype=hid.dtype)
        return P.rewrap(out, hid + scale * dv)

    return [P.decoder_layers(model)[int(layer)].register_forward_hook(hook)]


def residual_norm_stats(model, tokenizer, prompts, layer, batch_size=BATCH,
                        max_len=MAX_LEN):
    """mu_l = mean L2 norm of the layer-l residual stream.

    Reported both over every non-pad position (the value used for dosing, and
    what 2605.13329 means by 'mean residual norm') and at the readout token.
    """
    blocks = P.decoder_layers(model)
    store = {}

    def hook(_mod, _inp, out):
        store["h"] = P.hidden_of(out).detach()

    handle = blocks[int(layer)].register_forward_hook(hook)
    device = next(model.parameters()).device
    tot, cnt, last_norms = 0.0, 0, []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            enc, last_index, _raw = P.encode_batch(
                tokenizer, chunk, max_len=max_len, device=device, side="right"
            )
            with torch.no_grad():
                model(**enc)
            hid = store["h"].float()
            mask = enc["attention_mask"].bool()
            nrm = hid.norm(dim=-1)
            tot += float(nrm[mask].sum())
            cnt += int(mask.sum())
            rows = torch.arange(len(chunk), device=device)
            last_norms.extend(nrm[rows, last_index].cpu().numpy().tolist())
            store.clear()
    finally:
        handle.remove()
    return {
        "mu_alltoken": tot / max(1, cnt),
        "mu_lasttoken": float(np.mean(last_norms)),
        "n_tokens": cnt,
        "layer": int(layer),
    }


def swept_gaps(model, tokenizer, prompts, direction, mu, c_values,
               r_ids, c_ids, layer=STEER_LAYER, batch_size=BATCH, log=None):
    """Per-prompt refusal logit gap at each steering coefficient.

    Returns array [len(c_values), n_prompts]. c == 0 runs with no hook at all,
    which doubles as a check that a zero-coefficient hook is a no-op.
    """
    out = np.zeros((len(c_values), len(prompts)), dtype=np.float64)
    for i, cval in enumerate(c_values):
        t0 = time.time()
        if abs(float(cval)) < 1e-12:
            handles = []
        else:
            handles = steer_hook(model, direction, layer, cval, mu)
        try:
            out[i] = P.logit_readout(model, tokenizer, prompts, r_ids, c_ids,
                                     batch_size=batch_size, max_len=MAX_LEN)
        finally:
            P.remove_hooks(handles)
        if log is not None:
            log("    c=%+.2f mean_gap=%+.4f (%.1fs)"
                % (cval, float(out[i].mean()), time.time() - t0))
    return out


# --------------------------------------------------------------- curve stats


def sweep_with_extension(model, tokenizer, prompts, direction, mu, c_values,
                         r_ids, c_ids, extra=None, target=0.5,
                         batch_size=BATCH, log=None):
    """Sweep the grid, then push further out while the model has still not been
    driven past its own boundary.

    d_50 is undefined for any model whose crossing rate never reaches 0.5, and
    a missing d_50 would silently drop that model from the headline comparison.
    """
    extra = list(EXTRA_C if extra is None else extra)
    gaps = swept_gaps(model, tokenizer, prompts, direction, mu, c_values,
                      r_ids, c_ids, batch_size=batch_size, log=log)
    c_used = list(c_values)
    for ce in extra:
        rates, _d, _n = curve_from_gaps(gaps, c_used, c_used.index(0.0))
        best = float(np.nanmax(rates)) if np.isfinite(rates).any() else 0.0
        if best >= target:
            break
        if log is not None:
            log("    extending grid to c=%.1f (max crossing %.3f < %.2f)"
                % (ce, best, target))
        gk = swept_gaps(model, tokenizer, prompts, direction, mu, [ce],
                        r_ids, c_ids, batch_size=batch_size, log=log)
        gaps = np.vstack([gk, gaps])
        c_used = [ce] + c_used
    return gaps, c_used


def crossing_rate(gaps_c, gaps_0):
    """Fraction of BASELINE-REFUSING prompts driven to compliance-leaning.

    Denominator is the prompts that can actually cross (gap_0 > 0); the count
    is reported alongside so the rate is never read without it.
    """
    base_pos = gaps_0 > 0
    n = int(base_pos.sum())
    if n == 0:
        return float("nan"), 0
    crossed = np.logical_and(base_pos, gaps_c < 0)
    return float(crossed.sum()) / n, n


def curve_from_gaps(gaps, c_values, zero_index):
    """Crossing rate and achieved displacement at every coefficient.

    displacement = mean(gap_unsteered - gap_c); positive = pushed toward
    compliance, matching A1's `delta = intact - ablated` convention.
    """
    g0 = gaps[zero_index]
    rates, disps, n_elig = [], [], 0
    for i in range(gaps.shape[0]):
        r, n = crossing_rate(gaps[i], g0)
        rates.append(r)
        n_elig = n
        disps.append(float((g0 - gaps[i]).mean()))
    return np.asarray(rates), np.asarray(disps), n_elig


def _interp_at_half(x_vals, rates, target=0.5):
    """x at which the crossing rate first reaches `target`, by linear
    interpolation after sorting on x."""
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(rates, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return float("nan")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.nanmax(y) < target or np.nanmin(y) > target:
        return float("nan")
    for i in range(x.size - 1):
        y0, y1 = y[i], y[i + 1]
        if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
            return float(x[i] + (target - y0) * (x[i + 1] - x[i]) / (y1 - y0))
    return float("nan")


def fifty_points(gaps, c_values, zero_index, target=0.5):
    """(c_50, d_50): the coefficient and the achieved displacement at which
    half of the eligible prompts have crossed the boundary."""
    rates, disps, _n = curve_from_gaps(gaps, c_values, zero_index)
    # c is swept negative-toward-compliance, so order on -c for monotonicity
    c50_neg = _interp_at_half([-c for c in c_values], rates, target)
    d50 = _interp_at_half(disps, rates, target)
    c50 = float("nan") if not np.isfinite(c50_neg) else -c50_neg
    return c50, d50


def bootstrap_fifty(gaps, c_values, zero_index, n_boot=1000, seed=SEED,
                    target=0.5):
    """Percentile CI for c_50 and d_50, resampling PROMPTS (the unit of
    replication) and recomputing the whole curve each time."""
    rng = np.random.default_rng(seed)
    n = gaps.shape[1]
    c50s, d50s = [], []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        c50, d50 = fifty_points(gaps[:, idx], c_values, zero_index, target)
        c50s.append(c50)
        d50s.append(d50)
    return np.asarray(c50s, dtype=np.float64), np.asarray(d50s, dtype=np.float64)


def pct_ci(arr, alpha=0.05):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"lo": None, "hi": None, "n_finite": 0, "median": None}
    lo, hi = np.percentile(a, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    return {"lo": float(lo), "hi": float(hi), "n_finite": int(a.size),
            "median": float(np.median(a))}


# --------------------------------------------------------------- generation


def unclosed_think_fraction(texts):
    """Think-family blocks do not reliably close inside the token budget; the
    refusal classifier is only meaningful on the post-</think> body."""
    if not texts:
        return {"frac_with_think": 0.0, "frac_unclosed_think": 0.0, "n": 0}
    opened = [("<think>" in t) for t in texts]
    unclosed = [("<think>" in t and "</think>" not in t) for t in texts]
    return {
        "frac_with_think": float(np.mean(opened)),
        "frac_unclosed_think": float(np.mean(unclosed)),
        "n": len(texts),
    }


def generate_steered(model, tokenizer, prompts, direction, mu, coeff,
                     layer=STEER_LAYER, max_new_tokens=GEN_NEW_TOKENS,
                     batch_size=GEN_BATCH):
    """Greedy continuations with the steering hook live for the whole rollout."""
    if abs(float(coeff)) < 1e-12:
        handles = []
    else:
        handles = steer_hook(model, direction, layer, coeff, mu)
    try:
        return P.generate_texts(model, tokenizer, prompts,
                                max_new_tokens=max_new_tokens,
                                batch_size=batch_size, max_len=MAX_LEN)
    finally:
        P.remove_hooks(handles)


# --------------------------------------------------------------- worker


def _fit_base_directions(model, tok, data, pth, log, batch_size=BATCH):
    """Fit the BASE layer-20 refusal direction BOTH ways on the train split.

    Marks & Tegmark report mass-mean directions are more causally implicated
    than logistic ones, so both are carried to every checkpoint unchanged.
    """
    fit_prompts = list(data["harm_train"]) + list(data["ben_train"])
    fit_labels = np.array([1] * len(data["harm_train"]) + [0] * len(data["ben_train"]))
    t0 = time.time()
    pack = P.extract_activations(model, tok, fit_prompts, P.PROBE_LAYERS,
                                 cache_path=pth["acts"], batch_size=batch_size)
    log("train activations %s in %.1fs; trunc frac %.4f"
        % (str(pack["acts"].shape), time.time() - t0, pack["frac_trunc"]))

    li = P.PROBE_LAYERS.index(FIT_LAYER)
    v_mm, mm_norm = P.refusal_direction(pack["acts"], fit_labels, layer_index=li)
    t0 = time.time()
    v_lr, lr_info = fit_logistic_direction(pack["acts"][li], fit_labels)
    lr_info["fit_seconds"] = time.time() - t0
    cos = float(np.dot(v_mm.astype(np.float64), v_lr.astype(np.float64)))
    log("massmean norm %.3f | logistic n_iter %d acc %.3f | cos(mm,lr) %.4f"
        % (mm_norm, lr_info["n_iter"], lr_info["train_acc"], cos))

    np.savez_compressed(pth["dirs_npz"], massmean=v_mm, logistic=v_lr,
                        layer=FIT_LAYER)
    meta = {
        "fit_layer": FIT_LAYER,
        "probe_layers": P.PROBE_LAYERS,
        "massmean_raw_norm": float(mm_norm),
        "logistic": lr_info,
        "cos_massmean_logistic": cos,
        "d_model": int(v_mm.shape[0]),
        "train_trunc_frac": float(pack["frac_trunc"]),
        "provenance": provenance(data),
    }
    P.write_json(pth["dirs"], meta)

    smoke = {
        "tests": [
            P.smoke_hook_liveness(model, tok, v_mm, data["harm_held"][:8]),
            P.smoke_layer_variation(pack["acts"], P.PROBE_LAYERS),
            P.smoke_truncation_audit(pack["lengths"], MAX_LEN),
            P.smoke_degenerate_directions(pack["acts"], fit_labels, P.PROBE_LAYERS),
            P.smoke_random_control(int(v_mm.shape[0]), k=N_RAND),
        ],
    }
    smoke["all_pass"] = bool(all(t["pass"] for t in smoke["tests"]))
    P.write_json(pth["smoke"], smoke)
    log("smoke: " + ", ".join("%s=%s" % (t["name"], "PASS" if t["pass"] else "FAIL")
                              for t in smoke["tests"]))
    del pack
    gc.collect()
    return v_mm, v_lr, meta


def gate_a_worker(art=ART, ckpts=None, n_train=N_TRAIN, n_held=N_HELD,
                  n_rand=N_RAND, n_gen=N_GEN, batch_size=BATCH,
                  gen_batch_size=GEN_BATCH, gen_new_tokens=GEN_NEW_TOKENS,
                  c_grid=None, c_grid_null=None, do_generation=True):
    """Full Gate A sweep. Detached-thread safe: every argument is passed in, no
    notebook global is read, and every stage writes its artifact to disk."""
    ckpts = ckpts or CKPTS
    c_grid = list(c_grid or C_GRID)
    c_grid_null = list(c_grid_null or C_GRID_NULL)
    pth = paths(art)

    def log(msg):
        return P.elog(pth["log"], msg)

    t_all = time.time()
    status = {"stage": "start", "t0": t_all, "done": [], "error": None,
              "c_grid": c_grid, "c_grid_null": c_grid_null}
    P.write_json(pth["status"], status)

    try:
        P.set_seed(SEED)
        log("=== GATE A start; seed=%d ckpts=%d c_grid=%s" % (SEED, len(ckpts), c_grid))
        data = P.load_prompts(n_train=n_train, n_held=n_held, seed=SPLIT_SEED,
                              cache_path=pth["prompts"])
        prov = provenance(data)
        log("split fingerprint %s | train %d/%d | held harmful %d"
            % (prov["split_fingerprint"], len(data["harm_train"]),
               len(data["ben_train"]), len(data["harm_held"])))
        harm_held = data["harm_held"]
        zero_index = c_grid.index(0.0)

        v_mm = v_lr = None
        for label, repo, branch, commit in ckpts:
            out_path = pth["sweep"](label)
            if os.path.exists(out_path) and os.path.exists(pth["dirs_npz"]):
                log("skip %s (artifact exists)" % label)
                status["done"].append(label)
                P.write_json(pth["status"], status)
                continue

            status["stage"] = "load:" + label
            P.write_json(pth["status"], status)
            t0 = time.time()
            log("loading %s @ %s (%s)" % (repo, branch, commit[:10]))
            tok = P.load_tokenizer(repo, revision=commit)
            model = P.load_model(repo, revision=commit)
            t_load = time.time() - t0
            log("loaded %s in %.1fs" % (label, t_load))

            r_ids, r_single, r_dec = P.onset_token_ids(tok, P.REFUSAL_STRS)
            c_ids, c_single, c_dec = P.onset_token_ids(tok, P.COMPLY_STRS)
            d_model = int(model.config.hidden_size)

            if label == "base" and not os.path.exists(pth["dirs_npz"]):
                status["stage"] = "fit_directions"
                P.write_json(pth["status"], status)
                v_mm, v_lr, _meta = _fit_base_directions(model, tok, data, pth,
                                                         log, batch_size)
            if v_mm is None:
                with np.load(pth["dirs_npz"]) as blob:
                    v_mm = blob["massmean"]
                    v_lr = blob["logistic"]

            status["stage"] = "mu:" + label
            P.write_json(pth["status"], status)
            t0 = time.time()
            mu_stats = residual_norm_stats(model, tok, harm_held, STEER_LAYER,
                                           batch_size=batch_size)
            mu = mu_stats["mu_alltoken"]
            log("%s mu_L%d = %.3f (all-token) / %.3f (last-token) in %.1fs"
                % (label, STEER_LAYER, mu, mu_stats["mu_lasttoken"],
                   time.time() - t0))

            rec = {
                "label": label, "repo": repo, "branch": branch, "commit": commit,
                "d_model": d_model, "n_layers": len(P.decoder_layers(model)),
                "load_seconds": t_load, "provenance": prov,
                "mu_stats": mu_stats, "mu_used": mu,
                "c_grid": c_grid, "c_grid_null": c_grid_null,
                "zero_index": zero_index,
                "refusal_ids": r_ids, "refusal_single_token": r_single,
                "refusal_decoded": r_dec,
                "comply_ids": c_ids, "comply_single_token": c_single,
                "comply_decoded": c_dec,
                "pad_token_id": int(tok.pad_token_id),
                "steer_param": "h_l <- h_l + c * mu_l * v_hat (layer %d, all positions)" % STEER_LAYER,
            }

            status["stage"] = "sweep_massmean:" + label
            P.write_json(pth["status"], status)
            t0 = time.time()
            log("%s sweeping MASS-MEAN direction" % label)
            gaps_mm, c_mm = sweep_with_extension(
                model, tok, harm_held, v_mm, mu, c_grid, r_ids, c_ids,
                batch_size=batch_size, log=log)
            zi_mm = c_mm.index(0.0)
            log("%s massmean sweep in %.1fs (%d coefficients)"
                % (label, time.time() - t0, len(c_mm)))

            status["stage"] = "sweep_logistic:" + label
            P.write_json(pth["status"], status)
            t0 = time.time()
            log("%s sweeping LOGISTIC direction" % label)
            gaps_lr, c_lr = sweep_with_extension(
                model, tok, harm_held, v_lr, mu, c_grid, r_ids, c_ids,
                batch_size=batch_size, log=log)
            zi_lr = c_lr.index(0.0)
            log("%s logistic sweep in %.1fs (%d coefficients)"
                % (label, time.time() - t0, len(c_lr)))

            g0 = gaps_mm[zi_mm]
            rec["unsteered_gap_mean"] = float(g0.mean())
            rec["unsteered_gap_per_prompt"] = g0.tolist()
            rec["frac_refusing_unsteered"] = float((g0 > 0).mean())
            rec["gaps_massmean"] = gaps_mm.tolist()
            rec["gaps_logistic"] = gaps_lr.tolist()
            rec["c_grid_massmean"], rec["zero_index_massmean"] = c_mm, zi_mm
            rec["c_grid_logistic"], rec["zero_index_logistic"] = c_lr, zi_lr
            # sanity: c=0 must be identical between the two sweeps (no hook runs)
            rec["zero_coeff_max_abs_diff"] = float(
                np.abs(gaps_mm[zi_mm] - gaps_lr[zi_lr]).max())

            rates_mm, disps_mm, n_elig = curve_from_gaps(gaps_mm, c_mm, zi_mm)
            rates_lr, disps_lr, _ = curve_from_gaps(gaps_lr, c_lr, zi_lr)
            rec["n_eligible"] = int(n_elig)
            rec["curve_massmean"] = {"c": c_mm, "rates": rates_mm.tolist(),
                                     "disps": disps_mm.tolist()}
            rec["curve_logistic"] = {"c": c_lr, "rates": rates_lr.tolist(),
                                     "disps": disps_lr.tolist()}
            c50_mm, d50_mm = fifty_points(gaps_mm, c_mm, zi_mm)
            c50_lr, d50_lr = fifty_points(gaps_lr, c_lr, zi_lr)
            rec["c50_massmean"], rec["d50_massmean"] = c50_mm, d50_mm
            rec["c50_logistic"], rec["d50_logistic"] = c50_lr, d50_lr
            log("%s massmean: c50=%.3f d50=%.3f | logistic: c50=%.3f d50=%.3f | "
                "unsteered gap %+.3f (%.0f%% refusing)"
                % (label, c50_mm, d50_mm, c50_lr, d50_lr,
                   rec["unsteered_gap_mean"], 100 * rec["frac_refusing_unsteered"]))
            P.write_json(out_path, rec)

            status["stage"] = "null:" + label
            P.write_json(pth["status"], status)
            t0 = time.time()
            rands = P.random_unit_directions(d_model, n_rand, seed=SEED)
            null_gaps = []
            for k in range(n_rand):
                gk = swept_gaps(model, tok, harm_held, rands[k], mu, c_grid_null,
                                r_ids, c_ids, batch_size=batch_size, log=None)
                null_gaps.append(gk.tolist())
                if (k + 1) % 5 == 0:
                    log("%s null %d/%d (%.1fs)" % (label, k + 1, n_rand, time.time() - t0))
            rec["gaps_random"] = null_gaps
            rec["c_grid_random"] = c_grid_null
            rec["random_seed"] = SEED
            rec["random_reference"] = "unsteered_gap_per_prompt (c=0 is direction-free)"
            log("%s %d random directions x %d coefficients in %.1fs"
                % (label, n_rand, len(c_grid_null), time.time() - t0))
            P.write_json(out_path, rec)

            if do_generation and np.isfinite(c50_mm):
                status["stage"] = "generate:" + label
                P.write_json(pth["status"], status)
                t0 = time.time()
                gen_prompts = harm_held[:n_gen]
                txt_un = generate_steered(model, tok, gen_prompts, v_mm, mu, 0.0,
                                          max_new_tokens=gen_new_tokens,
                                          batch_size=gen_batch_size)
                txt_st = generate_steered(model, tok, gen_prompts, v_mm, mu, c50_mm,
                                          max_new_tokens=gen_new_tokens,
                                          batch_size=gen_batch_size)
                grec = {
                    "label": label, "repo": repo, "commit": commit,
                    "provenance": prov, "n": len(gen_prompts),
                    "max_new_tokens": gen_new_tokens,
                    "coeff": float(c50_mm),
                    "coeff_source": "c50 of the mass-mean crossing-rate curve for this model",
                    "mu_used": mu,
                    "prompts": gen_prompts,
                    "unsteered_text": txt_un, "steered_text": txt_st,
                    "unsteered_refusal": [bool(P.is_refusal(t)) for t in txt_un],
                    "steered_refusal": [bool(P.is_refusal(t)) for t in txt_st],
                    "unsteered_think": unclosed_think_fraction(txt_un),
                    "steered_think": unclosed_think_fraction(txt_st),
                    "seconds": time.time() - t0,
                }
                grec["unsteered_refusal_rate"] = float(np.mean(grec["unsteered_refusal"]))
                grec["steered_refusal_rate"] = float(np.mean(grec["steered_refusal"]))
                P.write_json(pth["gen"](label), grec)
                log("%s generations in %.1fs; refusal %.2f -> %.2f at c=%.3f; "
                    "unclosed <think> %.2f -> %.2f"
                    % (label, grec["seconds"], grec["unsteered_refusal_rate"],
                       grec["steered_refusal_rate"], c50_mm,
                       grec["unsteered_think"]["frac_unclosed_think"],
                       grec["steered_think"]["frac_unclosed_think"]))
                rec["behavioural"] = {
                    "coeff": float(c50_mm), "n": len(gen_prompts),
                    "max_new_tokens": gen_new_tokens,
                    "unsteered_refusal_rate": grec["unsteered_refusal_rate"],
                    "steered_refusal_rate": grec["steered_refusal_rate"],
                    "unsteered_think": grec["unsteered_think"],
                    "steered_think": grec["steered_think"],
                }
            elif do_generation:
                log("%s SKIP generation: c50 not reached on the swept grid" % label)
                rec["behavioural"] = {"skipped": "c50 not reached on the swept grid"}

            rec["elapsed_seconds"] = time.time() - t_all
            P.write_json(out_path, rec)
            status["done"].append(label)
            P.write_json(pth["status"], status)

            P.free_model(model)
            del tok
            freed = P.purge_hf_cache(repo)
            log("%s finished; purged %.2f GB of HF cache" % (label, freed / 1e9))

        status["stage"] = "done"
        status["total_seconds"] = time.time() - t_all
        P.write_json(pth["status"], status)
        log("=== GATE A done in %.1fs" % (time.time() - t_all))
    except Exception as exc:
        import traceback
        status["stage"] = "error"
        status["error"] = repr(exc)
        P.write_json(pth["status"], status)
        P.elog(pth["log"], "ERROR " + repr(exc))
        P.elog(pth["log"], traceback.format_exc())


def gate_a_launch(art=ART, **kw):
    """Idempotent, duplicate-safe launcher.

    Re-running a worker cell re-runs its descendant launch cell; without the
    live-thread guard that spawns a SECOND sweep and the two race on the GPU.
    """
    import threading
    for th in threading.enumerate():
        if th.name == "gateA-worker" and th.is_alive():
            return {"launched": False, "reason": "gateA-worker already alive"}
    pth = paths(art)
    ckpts = kw.get("ckpts") or CKPTS
    missing = [lab for lab, _r, _b, _c in ckpts if not os.path.exists(pth["sweep"](lab))]
    if not missing:
        return {"launched": False, "reason": "all sweep artifacts present"}
    P.run_detached(lambda: gate_a_worker(art=art, **kw), name="gateA-worker")
    return {"launched": True, "missing": missing}
