"""GAP 2 - non-safety control: does the boundary-distance effect appear for a
SENTIMENT direction, or is it specific to safety/refusal?

Mirrors gate_a.py, swapping three things and nothing else:
  * the fitted direction is SENTIMENT (positive vs negative SST-2 sentences)
    at layer 20 on BASE, both mass-mean and logistic (tol=1e-10);
  * the scalar readout is a positive-vs-negative continuation-token logit gap
    (documented token ids), replacing the refusal/comply gap;
  * the held-out set is 200 positive-sentiment sentences (baseline leans
    positive), steered along -c toward negative, replacing 200 harmful prompts.

Everything downstream - crossing rate, fifty_points, excess-d_50, paired
bootstrap - is the reviewed gate_a code reused unchanged, so the sentiment
records share the gate_a sweep schema. ASCII only.
"""

import gc
import hashlib
import io
import json
import os
import time

import numpy as np
import torch

import pipeline as P
import gate_a as ga


SEED = 42
SPLIT_SEED = 42
FIT_LAYER = 20
STEER_LAYER = 20
MAX_LEN = 128

_Q = chr(34)
SENT_SCAFFOLD = "Review: " + _Q + "%s" + _Q + chr(10) + "The sentiment of this review is"

POS_STRS = [" positive", " good", " great"]
NEG_STRS = [" negative", " bad", " terrible"]

MODELS = [
    ("base",     "allenai/Olmo-3-1025-7B",     "main", "a81bae42db3975be1671e27b9c9a56da1a9f980f"),
    ("instruct", "allenai/Olmo-3-7B-Instruct", "main", "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"),
]

N_TRAIN = 200
N_HELD = 200
N_RAND = 20
BATCH = 32

ART = "/marimo/gateA_sent"


def paths(art=ART):
    P.ensure_dir(art)
    return {
        "art": art,
        "log": os.path.join(art, "log.txt"),
        "status": os.path.join(art, "status.json"),
        "data": os.path.join(art, "sent_prompts.json"),
        "dirs": os.path.join(art, "sent_directions.json"),
        "dirs_npz": os.path.join(art, "sent_directions.npz"),
        "sweep": lambda lab: os.path.join(art, "sent_sweep_%s.json" % lab),
    }


def load_sentiment(n_train=N_TRAIN, n_held=N_HELD, seed=SPLIT_SEED, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        return P.read_json(cache_path)
    import random
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2", split="train")
    pos = [s.strip() for s, l in zip(ds["sentence"], ds["label"]) if l == 1 and s.strip()]
    neg = [s.strip() for s, l in zip(ds["sentence"], ds["label"]) if l == 0 and s.strip()]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    if len(pos) < n_train + n_held or len(neg) < n_train:
        raise RuntimeError("not enough sentiment sentences")
    blob = {
        "seed": seed, "n_train": n_train, "n_held": n_held,
        "pos_pool": len(pos), "neg_pool": len(neg),
        "pos_train": pos[:n_train], "neg_train": neg[:n_train],
        "pos_held": pos[n_train:n_train + n_held],
    }
    if cache_path:
        P.write_json(cache_path, blob)
    return blob


def split_fingerprint(data):
    keys = ("pos_train", "neg_train", "pos_held")
    blob = json.dumps({k: data[k] for k in keys}, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def provenance(data, pos_ids, neg_ids):
    return {
        "seed": SEED, "split_seed": SPLIT_SEED,
        "split_fingerprint": split_fingerprint(data),
        "dataset": "stanfordnlp/sst2 (train split)",
        "n_train": len(data["pos_train"]), "n_held": len(data["pos_held"]),
        "max_len": MAX_LEN, "fit_layer": FIT_LAYER, "steer_layer": STEER_LAYER,
        "scaffold": SENT_SCAFFOLD,
        "pos_strings": POS_STRS, "neg_strings": NEG_STRS,
        "pos_ids": pos_ids, "neg_ids": neg_ids,
        "readout": "mean logprob(pos tokens) - mean logprob(neg tokens) at last token",
    }


def sent_encode(tokenizer, sentences, max_len=MAX_LEN, device="cuda", side="right"):
    texts = [SENT_SCAFFOLD % str(s).strip() for s in sentences]
    raw_lengths = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
    prev = tokenizer.padding_side
    tokenizer.padding_side = side
    tokenizer.truncation_side = "left"
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len, add_special_tokens=True)
    tokenizer.padding_side = prev
    attn = enc["attention_mask"]
    last_index = attn.sum(dim=1) - 1
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc, last_index.to(device), raw_lengths


def sent_logit_readout(model, tokenizer, sentences, pos_ids, neg_ids,
                       batch_size=BATCH, max_len=MAX_LEN):
    device = next(model.parameters()).device
    p_ids = torch.as_tensor(pos_ids, dtype=torch.long, device=device)
    n_ids = torch.as_tensor(neg_ids, dtype=torch.long, device=device)
    vals = np.zeros(len(sentences), dtype=np.float64)
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start:start + batch_size]
        enc, last_index, _ = sent_encode(tokenizer, chunk, max_len=max_len,
                                         device=device, side="right")
        with torch.no_grad():
            out = model(**enc)
        rows = torch.arange(len(chunk), device=device)
        logits = out.logits[rows, last_index].float()
        logp = torch.log_softmax(logits, dim=-1)
        delta = logp[:, p_ids].mean(dim=-1) - logp[:, n_ids].mean(dim=-1)
        vals[start:start + len(chunk)] = delta.detach().cpu().numpy()
        del out, logits, logp
    return vals


def sent_swept_gaps(model, tokenizer, sentences, direction, mu, c_values,
                    pos_ids, neg_ids, layer=STEER_LAYER, batch_size=BATCH, log=None):
    out = np.zeros((len(c_values), len(sentences)), dtype=np.float64)
    for i, cval in enumerate(c_values):
        if abs(float(cval)) < 1e-12:
            handles = []
        else:
            handles = ga.steer_hook(model, direction, layer, cval, mu)
        try:
            out[i] = sent_logit_readout(model, tokenizer, sentences, pos_ids, neg_ids,
                                        batch_size=batch_size, max_len=MAX_LEN)
        finally:
            P.remove_hooks(handles)
        if log is not None:
            log("    c=%+.2f mean_gap=%+.4f" % (cval, float(out[i].mean())))
    return out


def sent_mu(model, tokenizer, sentences, layer, batch_size=BATCH, max_len=MAX_LEN):
    blocks = P.decoder_layers(model)
    store = {}

    def hook(_m, _i, out):
        store["h"] = P.hidden_of(out).detach()

    handle = blocks[int(layer)].register_forward_hook(hook)
    device = next(model.parameters()).device
    tot, cnt, last_norms = 0.0, 0, []
    try:
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start:start + batch_size]
            enc, last_index, _ = sent_encode(tokenizer, chunk, max_len=max_len,
                                             device=device, side="right")
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
    return {"mu_alltoken": tot / max(1, cnt), "mu_lasttoken": float(np.mean(last_norms)),
            "n_tokens": cnt, "layer": int(layer)}


def sent_extract_lasttoken(model, tokenizer, sentences, layer, batch_size=BATCH,
                           max_len=MAX_LEN):
    blocks = P.decoder_layers(model)
    store = {}

    def hook(_m, _i, out):
        store["h"] = P.hidden_of(out).detach()

    handle = blocks[int(layer)].register_forward_hook(hook)
    device = next(model.parameters()).device
    d_model = int(model.config.hidden_size)
    acts = np.zeros((len(sentences), d_model), dtype=np.float32)
    lengths = []
    try:
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start:start + batch_size]
            enc, last_index, raw = sent_encode(tokenizer, chunk, max_len=max_len,
                                               device=device, side="right")
            lengths.extend(raw)
            with torch.no_grad():
                model(**enc)
            rows = torch.arange(len(chunk), device=device)
            acts[start:start + len(chunk)] = store["h"][rows, last_index].float().cpu().numpy()
            store.clear()
    finally:
        handle.remove()
    return acts, np.asarray(lengths, dtype=np.int32)


def sent_sweep_with_extension(model, tok, sentences, direction, mu, c_values,
                              pos_ids, neg_ids, extra=None, target=0.5,
                              batch_size=BATCH, log=None):
    extra = list(ga.EXTRA_C if extra is None else extra)
    gaps = sent_swept_gaps(model, tok, sentences, direction, mu, c_values,
                           pos_ids, neg_ids, batch_size=batch_size, log=log)
    c_used = list(c_values)
    for ce in extra:
        rates, _d, _dm, _n = ga.curve_from_gaps(gaps, c_used, c_used.index(0.0))
        best = float(np.nanmax(rates)) if np.isfinite(rates).any() else 0.0
        if best >= target:
            break
        if log is not None:
            log("    extending grid to c=%.1f (max crossing %.3f)" % (ce, best))
        gk = sent_swept_gaps(model, tok, sentences, direction, mu, [ce],
                             pos_ids, neg_ids, batch_size=batch_size, log=log)
        gaps = np.vstack([gk, gaps])
        c_used = [ce] + c_used
    return gaps, c_used


def sent_worker(art=ART, models=None, n_train=N_TRAIN, n_held=N_HELD,
                n_rand=N_RAND, batch_size=BATCH, c_grid=None, c_grid_null=None):
    models = models or MODELS
    c_grid = list(c_grid or ga.C_GRID)
    c_grid_null = list(c_grid_null or ga.C_GRID_NULL)
    for _c in c_grid:
        if _c not in c_grid_null:
            c_grid_null.append(_c)
    c_grid_null = sorted(set(c_grid_null), reverse=True)
    pth = paths(art)

    def log(msg):
        return P.elog(pth["log"], msg)

    t_all = time.time()
    status = {"stage": "start", "t0": t_all, "done": [], "error": None}
    P.write_json(pth["status"], status)
    try:
        P.set_seed(SEED)
        data = load_sentiment(n_train, n_held, SPLIT_SEED, cache_path=pth["data"])
        zero_index = c_grid.index(0.0)
        v_mm = v_lr = None
        for label, repo, branch, commit in models:
            out_path = pth["sweep"](label)
            if os.path.exists(out_path):
                prior = None
                try:
                    with io.open(out_path, encoding="utf-8") as fh:
                        prior = json.load(fh)
                except Exception:
                    prior = None
                if isinstance(prior, dict) and prior.get("stages_complete") == ["sweep", "null"]:
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
            log("loaded %s in %.1fs" % (label, time.time() - t0))

            pos_ids, _ps, _pd = P.onset_token_ids(tok, POS_STRS)
            neg_ids, _ns, _nd = P.onset_token_ids(tok, NEG_STRS)
            d_model = int(model.config.hidden_size)
            prov = provenance(data, pos_ids, neg_ids)
            if label == "base":
                log("split fingerprint %s | pos_train %d neg_train %d pos_held %d"
                    % (prov["split_fingerprint"], len(data["pos_train"]),
                       len(data["neg_train"]), len(data["pos_held"])))

            if label == "base" and not os.path.exists(pth["dirs_npz"]):
                status["stage"] = "fit_directions"
                P.write_json(pth["status"], status)
                fit_sent = list(data["pos_train"]) + list(data["neg_train"])
                fit_lab = np.array([1] * len(data["pos_train"]) + [0] * len(data["neg_train"]))
                acts, lengths = sent_extract_lasttoken(model, tok, fit_sent, FIT_LAYER,
                                                       batch_size=batch_size)
                v_mm, mm_norm = P.refusal_direction(acts, fit_lab)
                v_lr, lr_info = ga.fit_logistic_direction(acts, fit_lab)
                cos = float(np.dot(v_mm.astype(np.float64), v_lr.astype(np.float64)))
                np.savez_compressed(pth["dirs_npz"], massmean=v_mm, logistic=v_lr, layer=FIT_LAYER)
                n_trunc = int((lengths > MAX_LEN).sum())
                P.write_json(pth["dirs"], {
                    "fit_layer": FIT_LAYER, "massmean_raw_norm": float(mm_norm),
                    "logistic": lr_info, "cos_massmean_logistic": cos,
                    "d_model": int(v_mm.shape[0]),
                    "train_trunc_frac": float(n_trunc) / max(1, len(lengths)),
                    "provenance": prov})
                log("sentiment dir: massmean norm %.3f | logistic n_iter %d acc %.3f | cos %.4f"
                    % (mm_norm, lr_info["n_iter"], lr_info["train_acc"], cos))
                del acts
                gc.collect()
            if v_mm is None:
                with np.load(pth["dirs_npz"]) as blob:
                    v_mm = blob["massmean"]
                    v_lr = blob["logistic"]

            held = data["pos_held"]
            status["stage"] = "mu:" + label
            P.write_json(pth["status"], status)
            mu_stats = sent_mu(model, tok, held, STEER_LAYER, batch_size=batch_size)
            mu = mu_stats["mu_alltoken"]
            log("%s mu_L%d = %.3f (all-token)" % (label, STEER_LAYER, mu))

            rec = {"label": label, "repo": repo, "branch": branch, "commit": commit,
                   "d_model": d_model, "provenance": prov, "mu_stats": mu_stats,
                   "mu_used": mu, "c_grid": c_grid, "c_grid_null": c_grid_null,
                   "zero_index": zero_index, "pos_ids": pos_ids, "neg_ids": neg_ids,
                   "steer_param": "h_l <- h_l + c*mu_l*v_hat (sentiment dir, layer %d)" % STEER_LAYER}

            status["stage"] = "sweep_massmean:" + label
            P.write_json(pth["status"], status)
            log("%s sweeping MASS-MEAN sentiment direction" % label)
            gaps_mm, c_mm = sent_sweep_with_extension(model, tok, held, v_mm, mu, c_grid,
                                                      pos_ids, neg_ids, batch_size=batch_size, log=log)
            zi_mm = c_mm.index(0.0)
            status["stage"] = "sweep_logistic:" + label
            P.write_json(pth["status"], status)
            log("%s sweeping LOGISTIC sentiment direction" % label)
            gaps_lr, c_lr = sent_sweep_with_extension(model, tok, held, v_lr, mu, c_grid,
                                                      pos_ids, neg_ids, batch_size=batch_size, log=log)
            zi_lr = c_lr.index(0.0)

            g0 = gaps_mm[zi_mm]
            rec["unsteered_gap_mean"] = float(g0.mean())
            rec["unsteered_gap_per_prompt"] = g0.tolist()
            rec["frac_refusing_unsteered"] = float((g0 > 0).mean())
            rec["gaps_massmean"] = gaps_mm.tolist()
            rec["gaps_logistic"] = gaps_lr.tolist()
            rec["c_grid_massmean"], rec["zero_index_massmean"] = c_mm, zi_mm
            rec["c_grid_logistic"], rec["zero_index_logistic"] = c_lr, zi_lr
            rec["zero_coeff_max_abs_diff"] = float(np.abs(gaps_mm[zi_mm] - gaps_lr[zi_lr]).max())

            rates_mm, disps_mm, dmed_mm, n_elig = ga.curve_from_gaps(gaps_mm, c_mm, zi_mm)
            rates_lr, disps_lr, dmed_lr, _ = ga.curve_from_gaps(gaps_lr, c_lr, zi_lr)
            rec["n_eligible"] = int(n_elig)
            rec["curve_massmean"] = {"c": c_mm, "rates": rates_mm.tolist(), "disps": disps_mm.tolist()}
            rec["curve_logistic"] = {"c": c_lr, "rates": rates_lr.tolist(), "disps": disps_lr.tolist()}
            c50_mm, d50_mm = ga.fifty_points(gaps_mm, c_mm, zi_mm)
            c50_lr, d50_lr = ga.fifty_points(gaps_lr, c_lr, zi_lr)
            rec["c50_massmean"], rec["d50_massmean"] = c50_mm, d50_mm
            rec["c50_logistic"], rec["d50_logistic"] = c50_lr, d50_lr
            log("%s massmean c50=%.3f d50=%.3f | median gap0 %.3f | %.0f%% positive-leaning"
                % (label, c50_mm, d50_mm, float(np.median(g0)), 100 * rec["frac_refusing_unsteered"]))

            status["stage"] = "null:" + label
            P.write_json(pth["status"], status)
            rands = P.random_unit_directions(d_model, n_rand, seed=SEED)
            null_gaps = []
            for k in range(n_rand):
                gk = sent_swept_gaps(model, tok, held, rands[k], mu, c_grid_null,
                                     pos_ids, neg_ids, batch_size=batch_size, log=None)
                null_gaps.append(gk.tolist())
            rec["gaps_random"] = null_gaps
            rec["c_grid_random"] = c_grid_null
            rec["random_seed"] = SEED
            rec["stages_complete"] = ["sweep", "null"]
            rec["elapsed_seconds"] = time.time() - t_all
            P.write_json(out_path, rec)
            status["done"].append(label)
            P.write_json(pth["status"], status)
            log("%s done" % label)

            P.free_model(model)
            del tok
            freed = P.purge_hf_cache(repo)
            log("%s purged %.2f GB" % (label, freed / 1e9))

        status["stage"] = "done"
        status["total_seconds"] = time.time() - t_all
        P.write_json(pth["status"], status)
        log("=== SENTIMENT done in %.1fs" % (time.time() - t_all))
    except Exception as exc:
        import traceback
        status["stage"] = "error"
        status["error"] = repr(exc)
        P.write_json(pth["status"], status)
        P.elog(pth["log"], "ERROR " + repr(exc))
        P.elog(pth["log"], traceback.format_exc())


def sent_launch(art=ART, **kw):
    import threading
    for th in threading.enumerate():
        if th.name == "sent-worker" and th.is_alive():
            return {"launched": False, "reason": "sent-worker already alive"}
    pth = paths(art)
    models = kw.get("models") or MODELS
    missing = [lab for lab, _r, _b, _c in models if not os.path.exists(pth["sweep"](lab))]
    if not missing:
        return {"launched": False, "reason": "all sentiment sweeps present"}
    P.run_detached(lambda: sent_worker(art=art, **kw), name="sent-worker")
    return {"launched": True, "missing": missing}
