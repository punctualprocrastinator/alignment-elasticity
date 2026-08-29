"""E1 - supervised causal steering direction (Box A, attack (i)).

Fit the L20 refusal steering direction THREE ways on BASE:
  (a) diff-in-means      (reused from gateA/directions.npz: 'massmean')
  (b) logistic tol=1e-10 (reused from gateA/directions.npz: 'logistic')
  (c) gradient-trained   (this module): optimise a single unit L20 vector v to
      minimise the mean refusal logit gap on 150 harmful TRAIN prompts when
      added as h += alpha*mu*v (alpha fixed), Adam, 50-prompt held-out early
      stopping, kept unit-norm.

Then compute margin (mean unsteered refusal gap) + steering efficacy (OLS slope
of mean per-prompt displacement of the refusal gap vs |c| for |c|<=2) for all
three directions on base / sft-1000 / dpo / instruct, on the SAME 200 held-out
prompts and the SAME c-grid as Gate A. Detached-thread safe (P10). ASCII only.
"""

import gc
import json
import os
import time

import numpy as np
import torch

import pipeline as P
import gate_a as ga

E1_ART = "/marimo/e1"
GA_ART = "/marimo/gateA"

FIT_LAYER = ga.FIT_LAYER
STEER_LAYER = ga.STEER_LAYER
MAX_LEN = ga.MAX_LEN
SEED = ga.SEED
BATCH = ga.BATCH

C_GRID = list(ga.C_GRID)

N_TRAIN = 150
N_VAL = 50
ALPHA = 1.0
TRAIN_BATCH = 30
LR = 5e-2
MAX_STEPS = 300
EVAL_EVERY = 10
PATIENCE = 6

E1_CKPTS = [
    ("base",     "allenai/Olmo-3-1025-7B",      "main",     "a81bae42db3975be1671e27b9c9a56da1a9f980f"),
    ("sft-1000", "allenai/Olmo-3-7B-Think-SFT",  "step1000", "9a45447cd55efd41e6eada0da28407277e818c63"),
    ("dpo",      "allenai/Olmo-3-7B-Think-DPO",  "main",     "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38"),
    ("instruct", "allenai/Olmo-3-7B-Instruct",   "main",     "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"),
]


def paths(art=E1_ART):
    P.ensure_dir(art)
    return {
        "art": art,
        "log": os.path.join(art, "log.txt"),
        "status": os.path.join(art, "status.json"),
        "vgrad": os.path.join(art, "v_grad.npz"),
        "vgrad_meta": os.path.join(art, "v_grad_meta.json"),
        "model": lambda lab: os.path.join(art, "e1_model_%s.json" % lab),
        "final": os.path.join(art, "E1_causal_direction.json"),
    }


def efficacy_from_gaps(gaps, c_values, zero_index, cmax=2.0):
    """Steering efficacy = OLS slope of mean per-prompt displacement of the
    refusal gap against |c|, over grid points with |c|<=cmax (paper definition).
    displacement(c) = mean_i (gap_unsteered_i - gap_c_i)."""
    gaps = np.asarray(gaps, dtype=np.float64)
    g0 = gaps[zero_index]
    cs = np.asarray(c_values, dtype=np.float64)
    xs, ys = [], []
    for i, cv in enumerate(cs):
        if abs(cv) <= cmax:
            xs.append(abs(cv))
            ys.append(float((g0 - gaps[i]).mean()))
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    order = np.argsort(xs, kind="stable")
    slope, intercept = np.polyfit(xs[order], ys[order], 1)
    return {"efficacy": float(slope), "intercept": float(intercept),
            "n_points": int(xs.size), "cmax": float(cmax),
            "abs_c": xs[order].tolist(), "disp": ys[order].tolist()}


def _diff_gap(model, tok, prompts, raw_vec, mu, alpha, r_ids, c_ids,
              layer=STEER_LAYER):
    """Differentiable mean refusal gap over one batch with the steering hook
    h += alpha*mu*(raw_vec/||raw_vec||) live at `layer`."""
    device = next(model.parameters()).device
    vec = raw_vec / (raw_vec.norm() + 1e-12)
    scale = float(alpha) * float(mu)

    def hook(_m, _i, out):
        hid = P.hidden_of(out)
        dv = vec.to(device=hid.device, dtype=hid.dtype)
        return P.rewrap(out, hid + scale * dv)

    handle = P.decoder_layers(model)[int(layer)].register_forward_hook(hook)
    try:
        enc, last_index, _raw = P.encode_batch(
            tok, prompts, max_len=MAX_LEN, device=device, side="right")
        out = model(**enc)
        rows = torch.arange(len(prompts), device=device)
        logits = out.logits[rows, last_index].float()
        logp = torch.log_softmax(logits, dim=-1)
        r_ids_t = torch.as_tensor(r_ids, dtype=torch.long, device=device)
        c_ids_t = torch.as_tensor(c_ids, dtype=torch.long, device=device)
        gap = logp[:, r_ids_t].mean(dim=-1) - logp[:, c_ids_t].mean(dim=-1)
        return gap.mean()
    finally:
        handle.remove()


def _eval_val_gap(model, tok, prompts, unit_vec, mu, alpha, r_ids, c_ids,
                  layer=STEER_LAYER, batch_size=BATCH):
    """Mean refusal gap on val prompts at the current unit vector (no grad)."""
    with torch.no_grad():
        tot, n = 0.0, 0
        for s in range(0, len(prompts), batch_size):
            chunk = prompts[s:s + batch_size]
            g = _diff_gap(model, tok, chunk, unit_vec, mu, alpha, r_ids, c_ids,
                          layer)
            tot += float(g) * len(chunk)
            n += len(chunk)
    return tot / max(1, n)


def train_gradient_direction(model, tok, data, mu, r_ids, c_ids, d_model, log,
                             seed=SEED):
    """Adam-optimise a unit L20 vector to minimise the mean refusal gap on the
    harmful TRAIN prompts; early-stop on a held-out 50-prompt harmful subset."""
    device = next(model.parameters()).device
    train_prompts = list(data["harm_train"])[:N_TRAIN]
    val_prompts = list(data["harm_train"])[N_TRAIN:N_TRAIN + N_VAL]
    if len(val_prompts) < N_VAL:
        raise RuntimeError("need >= %d harm_train prompts" % (N_TRAIN + N_VAL))

    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(d_model, generator=g, dtype=torch.float32)
    raw = (raw / raw.norm()).to(device=device, dtype=torch.float32)
    raw.requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=LR)

    rng = np.random.default_rng(seed)
    best_val = float("inf")
    best_vec = (raw.detach() / raw.detach().norm()).cpu().numpy().astype(np.float32)
    best_step = -1
    since_improve = 0
    history = []

    init_val = _eval_val_gap(model, tok, val_prompts,
                             raw.detach() / raw.detach().norm(), mu, ALPHA,
                             r_ids, c_ids)
    log("grad-train init val_gap=%+.4f (n_train=%d n_val=%d alpha=%.2f mu=%.3f lr=%.3g)"
        % (init_val, len(train_prompts), len(val_prompts), ALPHA, mu, LR))

    t0 = time.time()
    step = 0
    while step < MAX_STEPS:
        order = rng.permutation(len(train_prompts))
        for bs in range(0, len(train_prompts), TRAIN_BATCH):
            idx = order[bs:bs + TRAIN_BATCH]
            chunk = [train_prompts[i] for i in idx]
            opt.zero_grad()
            loss = _diff_gap(model, tok, chunk, raw, mu, ALPHA, r_ids, c_ids)
            loss.backward()
            opt.step()
            step += 1
            if step % EVAL_EVERY == 0 or step == 1:
                unit = raw.detach() / raw.detach().norm()
                vg = _eval_val_gap(model, tok, val_prompts, unit, mu, ALPHA,
                                   r_ids, c_ids)
                history.append({"step": step, "train_loss": float(loss),
                                "val_gap": vg})
                if vg < best_val - 1e-4:
                    best_val = vg
                    best_vec = unit.cpu().numpy().astype(np.float32)
                    best_step = step
                    since_improve = 0
                else:
                    since_improve += 1
                log("  step %3d train_loss=%+.4f val_gap=%+.4f best=%+.4f@%d (%.1fs)"
                    % (step, float(loss), vg, best_val, best_step, time.time() - t0))
                if since_improve >= PATIENCE:
                    log("  early stop at step %d" % step)
                    step = MAX_STEPS
                    break
            if step >= MAX_STEPS:
                break

    best_vec = best_vec / (np.linalg.norm(best_vec) + 1e-12)
    meta = {"init_val_gap": init_val, "best_val_gap": best_val,
            "best_step": best_step, "max_steps": MAX_STEPS, "lr": LR,
            "alpha": ALPHA, "n_train": len(train_prompts),
            "n_val": len(val_prompts), "train_batch": TRAIN_BATCH,
            "mu_train": float(mu), "seed": seed, "history": history,
            "seconds": time.time() - t0}
    return best_vec.astype(np.float32), meta


def e1_worker(art=E1_ART, ga_art=GA_ART, ckpts=None, batch_size=BATCH):
    ckpts = ckpts or E1_CKPTS
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
        log("=== E1 start; split fingerprint %s" % fp)
        if fp != "99a7ac88967302166d6e1698d1eebae8d2fd9576":
            raise RuntimeError("split fingerprint mismatch: %s (STOP)" % fp)

        with np.load(ga_pth["dirs_npz"]) as blob:
            v_mm = np.asarray(blob["massmean"], dtype=np.float32)
            v_lr = np.asarray(blob["logistic"], dtype=np.float32)
        d_model = int(v_mm.shape[0])
        harm_held = data["harm_held"]
        zero_index = C_GRID.index(0.0)

        for label, repo, branch, commit in ckpts:
            out_path = pth["model"](label)
            if os.path.exists(out_path):
                prior = json.load(open(out_path))
                if prior.get("complete"):
                    log("skip %s (complete)" % label)
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

            r_ids, _rs, r_dec = P.onset_token_ids(tok, P.REFUSAL_STRS)
            c_ids, _cs, c_dec = P.onset_token_ids(tok, P.COMPLY_STRS)

            status["stage"] = "mu:" + label
            P.write_json(pth["status"], status)
            mu_stats = ga.residual_norm_stats(model, tok, harm_held, STEER_LAYER,
                                              batch_size=batch_size)
            mu = mu_stats["mu_alltoken"]
            log("%s mu_L%d = %.3f" % (label, STEER_LAYER, mu))

            if label == "base":
                if os.path.exists(pth["vgrad"]):
                    with np.load(pth["vgrad"]) as b:
                        v_grad = np.asarray(b["grad"], dtype=np.float32)
                    log("loaded existing v_grad")
                else:
                    status["stage"] = "train_grad"
                    P.write_json(pth["status"], status)
                    v_grad, gmeta = train_gradient_direction(
                        model, tok, data, mu, r_ids, c_ids, d_model, log)
                    # Orient toward REFUSAL (same convention as diff-in-means /
                    # logistic) so negative c pushes toward compliance and the
                    # shared |c|<=2 efficacy slope is directly comparable across
                    # families. Gradient training minimises the gap, so the raw
                    # vector points toward compliance (cos to massmean < 0).
                    if float(np.dot(v_grad.astype(np.float64), v_mm.astype(np.float64))) < 0:
                        v_grad = (-v_grad).astype(np.float32)
                        log("oriented v_grad toward refusal (flipped sign)")
                    gmeta["oriented_to_refusal"] = True
                    gmeta["cos_to_massmean"] = float(np.dot(
                        v_grad.astype(np.float64), v_mm.astype(np.float64)))
                    gmeta["cos_to_logistic"] = float(np.dot(
                        v_grad.astype(np.float64), v_lr.astype(np.float64)))
                    gmeta["d_model"] = d_model
                    np.savez_compressed(pth["vgrad"], grad=v_grad,
                                        massmean=v_mm, logistic=v_lr)
                    P.write_json(pth["vgrad_meta"], gmeta)
                    log("v_grad best_val=%+.4f cos(mm)=%.4f cos(lr)=%.4f"
                        % (gmeta["best_val_gap"], gmeta["cos_to_massmean"],
                           gmeta["cos_to_logistic"]))
            else:
                with np.load(pth["vgrad"]) as b:
                    v_grad = np.asarray(b["grad"], dtype=np.float32)

            dirs = {"massmean": v_mm, "logistic": v_lr, "grad": v_grad}
            rec = {
                "label": label, "repo": repo, "branch": branch, "commit": commit,
                "d_model": d_model, "mu_stats": mu_stats, "mu_used": mu,
                "c_grid": C_GRID, "zero_index": zero_index,
                "split_fingerprint": fp, "seed": SEED,
                "refusal_ids": r_ids, "refusal_decoded": r_dec,
                "comply_ids": c_ids, "comply_decoded": c_dec,
                "load_seconds": t_load,
                "steer_param": "h_l <- h_l + c*mu_l*v_hat (layer %d)" % STEER_LAYER,
                "families": {},
            }
            margin = None
            for fam in ("massmean", "logistic", "grad"):
                status["stage"] = "sweep_%s:%s" % (fam, label)
                P.write_json(pth["status"], status)
                t0 = time.time()
                gaps = ga.swept_gaps(model, tok, harm_held, dirs[fam], mu,
                                     C_GRID, r_ids, c_ids, batch_size=batch_size,
                                     log=None)
                g0 = gaps[zero_index]
                if margin is None:
                    margin = float(g0.mean())
                eff = efficacy_from_gaps(gaps, C_GRID, zero_index)
                rec["families"][fam] = {
                    "efficacy": eff["efficacy"], "intercept": eff["intercept"],
                    "eff_n_points": eff["n_points"],
                    "abs_c": eff["abs_c"], "disp": eff["disp"],
                    "gaps": gaps.tolist()}
                log("%s %-8s efficacy=%.4f (int=%+.3f n=%d) %.1fs"
                    % (label, fam, eff["efficacy"], eff["intercept"],
                       eff["n_points"], time.time() - t0))

            rec["margin"] = margin
            rec["unsteered_gap_mean"] = margin
            rec["complete"] = True
            P.write_json(out_path, rec)
            log("%s DONE margin=%+.3f | eff mm=%.3f lr=%.3f grad=%.3f"
                % (label, margin, rec["families"]["massmean"]["efficacy"],
                   rec["families"]["logistic"]["efficacy"],
                   rec["families"]["grad"]["efficacy"]))
            status["done"].append(label)
            P.write_json(pth["status"], status)
            P.free_model(model)
            del tok
            freed = P.purge_hf_cache(repo)
            log("%s freed %.2f GB cache" % (label, freed / 1e9))

        status["stage"] = "aggregate"
        P.write_json(pth["status"], status)
        aggregate(art)
        status["stage"] = "done"
        status["total_seconds"] = time.time() - t_all
        P.write_json(pth["status"], status)
        log("=== E1 done in %.1fs" % (time.time() - t_all))
    except Exception as exc:
        import traceback
        status["stage"] = "error"
        status["error"] = repr(exc)
        P.write_json(pth["status"], status)
        P.elog(pth["log"], "ERROR " + repr(exc))
        P.elog(pth["log"], traceback.format_exc())


def aggregate(art=E1_ART):
    """Collate per-model records into E1_causal_direction.json with efficacy CV
    across checkpoints per direction family."""
    pth = paths(art)
    labels = [lab for lab, _r, _b, _c in E1_CKPTS]
    models = {}
    for lab in labels:
        p = pth["model"](lab)
        if os.path.exists(p):
            r = json.load(open(p))
            if r.get("complete"):
                models[lab] = r
    per_family = {}
    for fam in ("massmean", "logistic", "grad"):
        effs = {lab: models[lab]["families"][fam]["efficacy"] for lab in models}
        vals = np.asarray([effs[lab] for lab in models], dtype=np.float64)
        mean = float(vals.mean()) if vals.size else float("nan")
        sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        per_family[fam] = {
            "efficacy_by_model": effs, "mean": mean, "sd": sd,
            "cv": float(sd / mean) if mean else float("nan"),
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
            "range_ratio": (float(vals.max() / vals.min())
                            if vals.size and vals.min() != 0 else None)}
    vgm = json.load(open(pth["vgrad_meta"])) if os.path.exists(pth["vgrad_meta"]) else {}
    out = {
        "experiment": "E1 - supervised causal steering direction (three families)",
        "definition": {
            "margin": "mean unsteered refusal logit gap",
            "efficacy": "OLS slope of mean per-prompt displacement of the "
                        "refusal gap vs |c|, grid points |c|<=2",
            "steer_param": "h_l <- h_l + c*mu_l*v_hat at layer %d" % STEER_LAYER},
        "split_fingerprint": (list(models.values())[0]["split_fingerprint"]
                              if models else None),
        "seed": SEED, "c_grid": C_GRID,
        "checkpoints": [lab for lab in labels if lab in models],
        "margins": {lab: models[lab]["margin"] for lab in models},
        "efficacy_by_family": per_family,
        "gradient_direction": {
            "cos_to_massmean": vgm.get("cos_to_massmean"),
            "cos_to_logistic": vgm.get("cos_to_logistic"),
            "best_val_gap": vgm.get("best_val_gap"),
            "init_val_gap": vgm.get("init_val_gap"),
            "best_step": vgm.get("best_step"),
            "alpha": vgm.get("alpha"), "lr": vgm.get("lr"),
            "n_train": vgm.get("n_train"), "n_val": vgm.get("n_val")}}
    P.write_json(pth["final"], out)
    return out


def e1_launch(art=E1_ART, **kw):
    import threading
    for th in threading.enumerate():
        if th.name == "e1-worker" and th.is_alive():
            return {"launched": False, "reason": "e1-worker already alive"}
    P.run_detached(lambda: e1_worker(art=art, **kw), name="e1-worker")
    return {"launched": True}
