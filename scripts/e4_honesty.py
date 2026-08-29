"""E4 - honesty concept (Box B): efficacy/margin trajectory for a truth direction.

Base honesty (truth) direction fit ONCE on base at L20 over 200 true + 200 false
Azaria-Mitchell statements (diff-in-means and logistic tol=1e-10), carried
unchanged across base / think-sft-1000 / think-dpo / think-rlvr-last / instruct.
Readout gap g = mean logp(true-onset) - mean logp(false-onset) at the answer
position under the neutral scaffold ending with the primer. Held set = TRUE
statements: margin = mean g0; negative c flips true->false. Efficacy = OLS slope
of displacement vs |c|, |c|<=2. Dissociation at c50: label-frame vs an
independent agreement-frame flip rate. Detached-thread safe (P10). ASCII only.
"""

import gc
import json
import os
import time

import numpy as np
import torch

import pipeline as P
import gate_a as ga
from e1_causal import efficacy_from_gaps

E4_ART = "/marimo/e4"
GA_ART = "/marimo/gateA"

FIT_LAYER = ga.FIT_LAYER
STEER_LAYER = ga.STEER_LAYER
MAX_LEN = ga.MAX_LEN
SEED = ga.SEED
SPLIT_SEED = ga.SPLIT_SEED
BATCH = ga.BATCH
N_RAND = ga.N_RAND

C_GRID = list(ga.C_GRID)
EXTRA_C = list(ga.EXTRA_C)

TRUE_STRS = [" true", " True"]
FALSE_STRS = [" false", " False"]
YES_STRS = [" Yes", " yes"]
NO_STRS = [" No", " no"]

FIT_TOPICS = ["cities", "companies", "animals", "elements"]
HELD_TOPICS = ["inventions", "facts", "capitals"]
N_FIT_PER_CLASS = 200
N_HELD_TRUE = 200

E4_CKPTS = [
    ("base",            "allenai/Olmo-3-1025-7B",      "main",      "a81bae42db3975be1671e27b9c9a56da1a9f980f"),
    ("think-sft-1000",  "allenai/Olmo-3-7B-Think-SFT",  "step1000",  "9a45447cd55efd41e6eada0da28407277e818c63"),
    ("think-dpo",       "allenai/Olmo-3-7B-Think-DPO",  "main",      "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38"),
    ("think-rlvr-last", "allenai/Olmo-3-7B-Think",      "step_1375", "031240693eb33d302cfa8d2df76af15d2da4b579"),
    ("instruct",        "allenai/Olmo-3-7B-Instruct",   "main",      "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"),
]


def paths(art=E4_ART):
    P.ensure_dir(art)
    return {
        "art": art,
        "log": os.path.join(art, "log.txt"),
        "status": os.path.join(art, "status.json"),
        "data": os.path.join(art, "am_split.json"),
        "dirs_npz": os.path.join(art, "honesty_dirs.npz"),
        "dirs": os.path.join(art, "honesty_dirs.json"),
        "model": lambda lab: os.path.join(art, "e4_model_%s.json" % lab),
        "final": os.path.join(art, "E4_honesty.json"),
    }


def load_am_split(cache_path, seed=SPLIT_SEED):
    if os.path.exists(cache_path):
        return P.read_json(cache_path)
    from datasets import load_dataset
    import random as _random

    def pool(topics):
        t, f = [], []
        for tp in topics:
            ds = load_dataset("atmallen/%s_azaria_mitchell" % tp, split="train")
            for r in ds:
                s = str(r["statement"]).strip()
                if not s:
                    continue
                (t if int(r["label"]) == 1 else f).append(s)
        return t, f

    fit_t, fit_f = pool(FIT_TOPICS)
    held_t, _held_f = pool(HELD_TOPICS)
    rng = _random.Random(seed)
    rng.shuffle(fit_t)
    rng.shuffle(fit_f)
    rng.shuffle(held_t)
    if len(fit_t) < N_FIT_PER_CLASS or len(fit_f) < N_FIT_PER_CLASS:
        raise RuntimeError("not enough fit statements: %d true / %d false"
                           % (len(fit_t), len(fit_f)))
    if len(held_t) < N_HELD_TRUE:
        raise RuntimeError("not enough held true statements: %d" % len(held_t))
    blob = {
        "seed": seed,
        "fit_topics": FIT_TOPICS, "held_topics": HELD_TOPICS,
        "fit_true": fit_t[:N_FIT_PER_CLASS],
        "fit_false": fit_f[:N_FIT_PER_CLASS],
        "held_true": held_t[:N_HELD_TRUE],
    }
    P.write_json(cache_path, blob)
    return blob


def data_fingerprint(data):
    import hashlib
    blob = json.dumps({k: data[k] for k in ("fit_true", "fit_false", "held_true")},
                      sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def hon_texts(statements, primer="The statement is"):
    out = []
    for s in statements:
        q = ("Is the following statement true or false?" + chr(10)
             + chr(34) + str(s).strip() + chr(34))
        out.append("User: " + q + chr(10) + "Assistant: " + primer)
    return out


def agree_texts(statements):
    out = []
    for s in statements:
        q = ("Is the following claim correct? Answer yes or no." + chr(10)
             + chr(34) + str(s).strip() + chr(34))
        out.append("User: " + q + chr(10) + "Assistant:")
    return out


def _encode(tokenizer, texts, device, max_len=MAX_LEN, side="right"):
    prev = tokenizer.padding_side
    tokenizer.padding_side = side
    tokenizer.truncation_side = "left"
    raw_lengths = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len, add_special_tokens=True)
    tokenizer.padding_side = prev
    attn = enc["attention_mask"]
    last_index = attn.sum(dim=1) - 1
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc, last_index.to(device), raw_lengths


def truth_readout(model, tokenizer, texts, true_ids, false_ids,
                  batch_size=BATCH, max_len=MAX_LEN, return_lengths=False):
    device = next(model.parameters()).device
    t_ids = torch.as_tensor(true_ids, dtype=torch.long, device=device)
    f_ids = torch.as_tensor(false_ids, dtype=torch.long, device=device)
    vals = np.zeros(len(texts), dtype=np.float64)
    all_len = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc, last_index, raw_len = _encode(tokenizer, chunk, device, max_len)
        all_len.extend(raw_len)
        with torch.no_grad():
            out = model(**enc)
        rows = torch.arange(len(chunk), device=device)
        logits = out.logits[rows, last_index].float()
        logp = torch.log_softmax(logits, dim=-1)
        delta = logp[:, t_ids].mean(dim=-1) - logp[:, f_ids].mean(dim=-1)
        vals[start:start + len(chunk)] = delta.detach().cpu().numpy()
        del out, logits, logp
    if return_lengths:
        return vals, np.asarray(all_len, dtype=np.int32)
    return vals


def extract_last_token(model, tokenizer, texts, layer, batch_size=BATCH, max_len=MAX_LEN):
    blocks = P.decoder_layers(model)
    store = {}

    def hook(_m, _i, out):
        store["h"] = P.hidden_of(out).detach()

    handle = blocks[int(layer)].register_forward_hook(hook)
    device = next(model.parameters()).device
    d_model = int(model.config.hidden_size)
    acts = np.zeros((len(texts), d_model), dtype=np.float32)
    lengths = []
    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc, last_index, raw_len = _encode(tokenizer, chunk, device, max_len)
            lengths.extend(raw_len)
            with torch.no_grad():
                model(**enc)
            rows = torch.arange(len(chunk), device=device)
            acts[start:start + len(chunk)] = store["h"][rows, last_index].float().cpu().numpy()
            store.clear()
    finally:
        handle.remove()
    return acts, np.asarray(lengths, dtype=np.int32)


def hon_residual_norm(model, tokenizer, texts, layer, batch_size=BATCH, max_len=MAX_LEN):
    blocks = P.decoder_layers(model)
    store = {}

    def hook(_m, _i, out):
        store["h"] = P.hidden_of(out).detach()

    handle = blocks[int(layer)].register_forward_hook(hook)
    device = next(model.parameters()).device
    tot, cnt = 0.0, 0
    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc, _li, _raw = _encode(tokenizer, chunk, device, max_len)
            with torch.no_grad():
                model(**enc)
            hid = store["h"].float()
            mask = enc["attention_mask"].bool()
            nrm = hid.norm(dim=-1)
            tot += float(nrm[mask].sum())
            cnt += int(mask.sum())
            store.clear()
    finally:
        handle.remove()
    return {"mu_alltoken": tot / max(1, cnt), "n_tokens": cnt, "layer": int(layer)}


def hon_swept_gaps(model, tokenizer, texts, direction, mu, c_values, true_ids,
                   false_ids, layer=STEER_LAYER, batch_size=BATCH, log=None):
    out = np.zeros((len(c_values), len(texts)), dtype=np.float64)
    for i, cval in enumerate(c_values):
        t0 = time.time()
        if abs(float(cval)) < 1e-12:
            handles = []
        else:
            handles = ga.steer_hook(model, direction, layer, cval, mu)
        try:
            out[i] = truth_readout(model, tokenizer, texts, true_ids, false_ids,
                                   batch_size=batch_size, max_len=MAX_LEN)
        finally:
            P.remove_hooks(handles)
        if log is not None:
            log("    c=%+.2f mean_gap=%+.4f (%.1fs)"
                % (cval, float(out[i].mean()), time.time() - t0))
    return out


def hon_sweep_with_extension(model, tokenizer, texts, direction, mu, c_values,
                             true_ids, false_ids, extra=None, target=0.5,
                             batch_size=BATCH, log=None):
    extra = list(EXTRA_C if extra is None else extra)
    gaps = hon_swept_gaps(model, tokenizer, texts, direction, mu, c_values,
                          true_ids, false_ids, batch_size=batch_size, log=log)
    c_used = list(c_values)
    for ce in extra:
        rates, _d, _dm, _n = ga.curve_from_gaps(gaps, c_used, c_used.index(0.0))
        best = float(np.nanmax(rates)) if np.isfinite(rates).any() else 0.0
        if best >= target:
            break
        if log is not None:
            log("    extending grid to c=%.1f (max flip %.3f)" % (ce, best))
        gk = hon_swept_gaps(model, tokenizer, texts, direction, mu, [ce],
                            true_ids, false_ids, batch_size=batch_size, log=log)
        gaps = np.vstack([gk, gaps])
        c_used = [ce] + c_used
    return gaps, c_used


def bootstrap_efficacy(gaps, c_values, zero_index, n_boot=1000, seed=SEED, cmax=2.0):
    gaps = np.asarray(gaps, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = gaps.shape[1]
    effs = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        effs.append(efficacy_from_gaps(gaps[:, idx], c_values, zero_index, cmax)["efficacy"])
    a = np.asarray(effs, dtype=np.float64)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "median": float(np.median(a))}


def _fit_honesty_directions(model, tok, data, pth, log, batch_size=BATCH):
    fit_stmts = list(data["fit_true"]) + list(data["fit_false"])
    fit_labels = np.array([1] * len(data["fit_true"]) + [0] * len(data["fit_false"]))
    fit_texts = hon_texts(fit_stmts)
    t0 = time.time()
    acts, lengths = extract_last_token(model, tok, fit_texts, FIT_LAYER, batch_size=batch_size)
    n_trunc = int((lengths > MAX_LEN).sum())
    log("fit activations %s in %.1fs; trunc frac %.4f"
        % (str(acts.shape), time.time() - t0, n_trunc / max(1, len(lengths))))
    v_mm, mm_norm = P.refusal_direction(acts, fit_labels, layer_index=None)
    v_lr, lr_info = ga.fit_logistic_direction(acts, fit_labels)
    if float(np.dot(v_mm.astype(np.float64), v_lr.astype(np.float64))) < 0:
        v_lr = (-v_lr).astype(np.float32)
        lr_info["flipped_to_massmean"] = True
    cos = float(np.dot(v_mm.astype(np.float64), v_lr.astype(np.float64)))
    log("massmean norm %.3f | logistic n_iter %d acc %.3f | cos(mm,lr) %.4f"
        % (mm_norm, lr_info["n_iter"], lr_info["train_acc"], cos))
    np.savez_compressed(pth["dirs_npz"], massmean=v_mm, logistic=v_lr, layer=FIT_LAYER)
    meta = {
        "fit_layer": FIT_LAYER, "massmean_raw_norm": float(mm_norm),
        "logistic": lr_info, "cos_massmean_logistic": cos,
        "d_model": int(v_mm.shape[0]),
        "orientation": "toward TRUE (label 1 - label 0)",
        "fit_trunc_frac": float(n_trunc / max(1, len(lengths))),
        "fit_topics": FIT_TOPICS, "held_topics": HELD_TOPICS,
        "n_fit_true": len(data["fit_true"]), "n_fit_false": len(data["fit_false"]),
    }
    P.write_json(pth["dirs"], meta)
    del acts
    gc.collect()
    return v_mm, v_lr, meta


def e4_worker(art=E4_ART, ga_art=GA_ART, ckpts=None, batch_size=BATCH,
              n_rand=N_RAND, do_dissociation=True):
    ckpts = ckpts or E4_CKPTS
    pth = paths(art)

    def log(msg):
        return P.elog(pth["log"], msg)

    t_all = time.time()
    status = {"stage": "start", "t0": t_all, "done": [], "error": None}
    P.write_json(pth["status"], status)
    try:
        P.set_seed(SEED)
        data = load_am_split(pth["data"])
        fp = data_fingerprint(data)
        log("=== E4 honesty start; data fingerprint %s | fit %d+%d held_true %d"
            % (fp, len(data["fit_true"]), len(data["fit_false"]), len(data["held_true"])))
        held = list(data["held_true"])
        held_texts = hon_texts(held)
        agree_held = agree_texts(held)

        v_mm = v_lr = None
        if os.path.exists(pth["dirs_npz"]):
            with np.load(pth["dirs_npz"]) as blob:
                v_mm = np.asarray(blob["massmean"], dtype=np.float32)
                v_lr = np.asarray(blob["logistic"], dtype=np.float32)

        for label, repo, branch, commit in ckpts:
            out_path = pth["model"](label)
            if os.path.exists(out_path):
                prior = P.read_json(out_path)
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

            t_ids, t_single, t_dec = P.onset_token_ids(tok, TRUE_STRS)
            f_ids, f_single, f_dec = P.onset_token_ids(tok, FALSE_STRS)
            y_ids, _ys, y_dec = P.onset_token_ids(tok, YES_STRS)
            n_ids, _ns, n_dec = P.onset_token_ids(tok, NO_STRS)
            d_model = int(model.config.hidden_size)

            if label == "base" and v_mm is None:
                status["stage"] = "fit_directions"
                P.write_json(pth["status"], status)
                v_mm, v_lr, _meta = _fit_honesty_directions(model, tok, data, pth, log, batch_size)
            if v_mm is None:
                raise RuntimeError("honesty_dirs.npz missing: base must run first")

            status["stage"] = "mu:" + label
            P.write_json(pth["status"], status)
            mu_stats = hon_residual_norm(model, tok, held_texts, STEER_LAYER, batch_size=batch_size)
            mu = mu_stats["mu_alltoken"]
            log("%s mu_L%d = %.3f" % (label, STEER_LAYER, mu))

            rec = {
                "label": label, "repo": repo, "branch": branch, "commit": commit,
                "d_model": d_model, "n_layers": len(P.decoder_layers(model)),
                "mu_stats": mu_stats, "mu_used": mu, "c_grid": C_GRID,
                "data_fingerprint": fp, "seed": SEED, "load_seconds": t_load,
                "true_ids": t_ids, "true_single": t_single, "true_decoded": t_dec,
                "false_ids": f_ids, "false_single": f_single, "false_decoded": f_dec,
                "yes_ids": y_ids, "yes_decoded": y_dec,
                "no_ids": n_ids, "no_decoded": n_dec,
                "steer_param": "h_l <- h_l + c*mu_l*v_hat (layer %d)" % STEER_LAYER,
                "families": {},
            }

            margin = None
            fam_c50 = None
            for fam, vec in (("massmean", v_mm), ("logistic", v_lr)):
                status["stage"] = "sweep_%s:%s" % (fam, label)
                P.write_json(pth["status"], status)
                t0 = time.time()
                gaps, c_used = hon_sweep_with_extension(
                    model, tok, held_texts, vec, mu, C_GRID, t_ids, f_ids,
                    batch_size=batch_size, log=None)
                zi = c_used.index(0.0)
                g0 = gaps[zi]
                if margin is None:
                    margin = float(g0.mean())
                eff = efficacy_from_gaps(gaps, c_used, zi)
                eff_ci = bootstrap_efficacy(gaps, c_used, zi)
                rates, disps, dmed, n_elig = ga.curve_from_gaps(gaps, c_used, zi)
                c50, d50 = ga.fifty_points(gaps, c_used, zi)
                if fam == "massmean":
                    fam_c50 = c50
                rec["families"][fam] = {
                    "efficacy": eff["efficacy"], "intercept": eff["intercept"],
                    "eff_n_points": eff["n_points"], "efficacy_ci": eff_ci,
                    "abs_c": eff["abs_c"], "disp": eff["disp"],
                    "margin": float(g0.mean()),
                    "frac_asserting_true_unsteered": float((g0 > 0).mean()),
                    "c50": c50, "d50": d50, "n_eligible": int(n_elig),
                    "c_used": c_used, "rates": rates.tolist(),
                    "disps": disps.tolist(), "gaps": gaps.tolist(),
                }
                log("%s %-8s margin=%+.3f efficacy=%.4f [%.3f,%.3f] c50=%.3f (%.1fs)"
                    % (label, fam, float(g0.mean()), eff["efficacy"],
                       eff_ci["lo"], eff_ci["hi"], c50, time.time() - t0))

            mm = rec["families"]["massmean"]
            g0_mm = np.asarray(mm["gaps"])[mm["c_used"].index(0.0)]
            m, mlo, mhi = P.bootstrap_ci(g0_mm, n_boot=1000, seed=SEED)
            rec["margin"] = float(m)
            rec["margin_ci"] = [float(mlo), float(mhi)]

            status["stage"] = "null:" + label
            P.write_json(pth["status"], status)
            t0 = time.time()
            zero_index = C_GRID.index(0.0)
            rands = P.random_unit_directions(d_model, n_rand, seed=SEED)
            null_effs = []
            for k in range(n_rand):
                gk = hon_swept_gaps(model, tok, held_texts, rands[k], mu, C_GRID,
                                    t_ids, f_ids, batch_size=batch_size, log=None)
                null_effs.append(efficacy_from_gaps(gk, C_GRID, zero_index)["efficacy"])
            null_effs = np.asarray(null_effs, dtype=np.float64)
            real_eff = mm["efficacy"]
            z, nmean, nsd = P.zscore_vs_null(real_eff, null_effs)
            rec["null_efficacy_mean"] = float(nmean)
            rec["null_efficacy_sd"] = float(nsd)
            rec["efficacy_z"] = float(z)
            rec["null_efficacies"] = null_effs.tolist()
            log("%s null efficacy %.4f+-%.4f | real %.4f z=%.1f (%.1fs)"
                % (label, nmean, nsd, real_eff, z, time.time() - t0))

            if do_dissociation and np.isfinite(fam_c50):
                status["stage"] = "dissociation:" + label
                P.write_json(pth["status"], status)
                t0 = time.time()
                lab0 = truth_readout(model, tok, held_texts, t_ids, f_ids, batch_size=batch_size)
                h = ga.steer_hook(model, v_mm, STEER_LAYER, fam_c50, mu)
                try:
                    labc = truth_readout(model, tok, held_texts, t_ids, f_ids, batch_size=batch_size)
                finally:
                    P.remove_hooks(h)
                agr0 = truth_readout(model, tok, agree_held, y_ids, n_ids, batch_size=batch_size)
                h = ga.steer_hook(model, v_mm, STEER_LAYER, fam_c50, mu)
                try:
                    agrc = truth_readout(model, tok, agree_held, y_ids, n_ids, batch_size=batch_size)
                finally:
                    P.remove_hooks(h)
                lab_elig = lab0 > 0
                agr_elig = agr0 > 0
                lab_flip = float(np.logical_and(lab_elig, labc < 0).sum() / max(1, int(lab_elig.sum())))
                agr_flip = float(np.logical_and(agr_elig, agrc < 0).sum() / max(1, int(agr_elig.sum())))
                rec["dissociation"] = {
                    "coeff": float(fam_c50),
                    "label_frame_flip_rate": lab_flip,
                    "agreement_frame_flip_rate": agr_flip,
                    "label_n_eligible": int(lab_elig.sum()),
                    "agree_n_eligible": int(agr_elig.sum()),
                    "label_mean_gap_0": float(lab0.mean()),
                    "label_mean_gap_c50": float(labc.mean()),
                    "agree_mean_gap_0": float(agr0.mean()),
                    "agree_mean_gap_c50": float(agrc.mean()),
                    "note": "label-frame = steered true/false; agreement-frame = "
                            "independent yes/no; label>>agree => label-only lever.",
                }
                log("%s dissociation c50=%.3f: label_flip=%.2f agree_flip=%.2f (%.1fs)"
                    % (label, fam_c50, lab_flip, agr_flip, time.time() - t0))
            else:
                rec["dissociation"] = {"skipped": "c50 not reached"}

            rec["elapsed_seconds"] = time.time() - t_all
            rec["complete"] = True
            P.write_json(out_path, rec)
            log("%s DONE margin=%+.3f [%.3f,%.3f] eff_mm=%.4f z=%.1f"
                % (label, rec["margin"], mlo, mhi, mm["efficacy"], rec["efficacy_z"]))
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
        log("=== E4 done in %.1fs" % (time.time() - t_all))
    except Exception as exc:
        import traceback
        status["stage"] = "error"
        status["error"] = repr(exc)
        P.write_json(pth["status"], status)
        P.elog(pth["log"], "ERROR " + repr(exc))
        P.elog(pth["log"], traceback.format_exc())


def aggregate(art=E4_ART):
    pth = paths(art)
    labels = [lab for lab, _r, _b, _c in E4_CKPTS]
    models = {}
    for lab in labels:
        p = pth["model"](lab)
        if os.path.exists(p):
            r = P.read_json(p)
            if r.get("complete"):
                models[lab] = r
    present = [lab for lab in labels if lab in models]
    per_family = {}
    for fam in ("massmean", "logistic"):
        effs = {lab: models[lab]["families"][fam]["efficacy"] for lab in present}
        vals = np.asarray([effs[lab] for lab in present], dtype=np.float64)
        mean = float(vals.mean()) if vals.size else float("nan")
        sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        per_family[fam] = {
            "efficacy_by_model": effs, "mean": mean, "sd": sd,
            "cv": float(sd / mean) if mean else float("nan"),
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
            "range_ratio": (float(vals.max() / vals.min())
                            if vals.size and vals.min() != 0 else None)}
    margins = {lab: models[lab]["margin"] for lab in present}
    mvals = np.asarray([margins[lab] for lab in present], dtype=np.float64)
    out = {
        "experiment": "E4 - honesty concept (efficacy / margin trajectory)",
        "definition": {
            "margin": "mean unsteered truth gap logp(true)-logp(false) on held TRUE statements",
            "efficacy": "OLS slope of mean displacement of truth gap vs |c|, |c|<=2",
            "steer_param": "h_l <- h_l + c*mu_l*v_hat at layer %d" % FIT_LAYER,
            "direction": "base honesty (truth) direction, carried unchanged",
            "dissociation": "label-frame (true/false) vs agreement-frame (yes/no) flip at c50"},
        "checkpoints": present,
        "data_fingerprint": models[present[0]]["data_fingerprint"] if present else None,
        "seed": SEED,
        "margins": margins,
        "margin_growth_base_to_instruct": (
            float(margins.get("instruct", float("nan")) / margins["base"])
            if "base" in margins and margins.get("base") else None),
        "margin_min": float(mvals.min()) if mvals.size else None,
        "margin_max": float(mvals.max()) if mvals.size else None,
        "efficacy_by_family": per_family,
        "efficacy_z": {lab: models[lab]["efficacy_z"] for lab in present},
        "c50": {lab: models[lab]["families"]["massmean"]["c50"] for lab in present},
        "dissociation": {lab: models[lab].get("dissociation") for lab in present},
    }
    P.write_json(pth["final"], out)
    return out


def e4_launch(art=E4_ART, **kw):
    import threading
    for th in threading.enumerate():
        if th.name == "e4-worker" and th.is_alive():
            return {"launched": False, "reason": "e4-worker already alive"}
    P.run_detached(lambda: e4_worker(art=art, **kw), name="e4-worker")
    return {"launched": True}
