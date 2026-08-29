"""E5 - second model family (Box B, attack iv: lineage/family).

Does "margin grows, efficacy constant" hold OFF OLMo? Fit the base refusal
direction on Qwen3-8B-Base (AdvBench harmful vs Alpaca benign) at the layer
proportional to OLMo's L20 (round(20/32 * n_layers) = L22 for Qwen3-8B's 36
layers), carry it UNCHANGED to Qwen3-8B (the chat/instruct endpoint). Compute
margin + efficacy (both diff-in-means + logistic) on base and instruct, the
refusal-onset behavioural crossing at each model's own c50, and a HarmBench-cls
genuine-harm check on both (reusing e2_behavioural).

FORMAT (protocol P6 + task): the format-matched comparison is base and instruct
BOTH under the neutral scaffold. The instruct model is ALSO measured under its
own chat template (enable_thinking=False) as a deployment-realistic arm. Format
and model are not separable in the chat arm -> reported as a caveat.

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
from e1_causal import efficacy_from_gaps
import e2_behavioural as e2

E5_ART = "/marimo/e5"
GA_ART = "/marimo/gateA"

MAX_LEN = ga.MAX_LEN
SEED = ga.SEED
SPLIT_SEED = ga.SPLIT_SEED
BATCH = ga.BATCH
N_RAND = ga.N_RAND

C_GRID = list(ga.C_GRID)
EXTRA_C = list(ga.EXTRA_C)

REF_NUM = 20
REF_DEN = 32

N_TRAIN = 200
N_HELD = 200
N_GEN = 80
N_JUDGE = 40
GEN_BATCH = 16
GEN_NEW = 256

# arms: (label, repo, fmt). base has no chat template; instruct measured both ways.
E5_ARMS = [
    ("base-neutral",     "Qwen/Qwen3-8B-Base", "neutral"),
    ("instruct-neutral", "Qwen/Qwen3-8B",      "neutral"),
    ("instruct-chat",    "Qwen/Qwen3-8B",      "chat"),
]
FIT_REPO = "Qwen/Qwen3-8B-Base"


def prop_layer(n_layers):
    return int(round(REF_NUM / REF_DEN * n_layers))


def resolve_commit(repo):
    from huggingface_hub import HfApi
    return HfApi().model_info(repo).sha


def paths(art=E5_ART):
    P.ensure_dir(art)
    return {
        "art": art,
        "log": os.path.join(art, "log.txt"),
        "status": os.path.join(art, "status.json"),
        "prompts": os.path.join(art, "prompts.json"),
        "dirs_npz": os.path.join(art, "qwen_dirs.npz"),
        "dirs": os.path.join(art, "qwen_dirs.json"),
        "arm": lambda lab: os.path.join(art, "e5_arm_%s.json" % lab),
        "final": os.path.join(art, "E5_second_family.json"),
    }


def chat_encode_batch(tokenizer, raw_prompts, max_len=MAX_LEN, device="cuda", side="right"):
    """Qwen chat template, direct-answer mode (enable_thinking=False so the first
    generated token is the refusal/comply onset, not a <think> tag)."""
    texts = []
    for p in raw_prompts:
        msgs = [{"role": "user", "content": str(p).strip()}]
        try:
            t = tokenizer.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
        except TypeError:
            t = tokenizer.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        texts.append(t)
    raw_lengths = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    prev = tokenizer.padding_side
    tokenizer.padding_side = side
    tokenizer.truncation_side = "left"
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len, add_special_tokens=False)
    tokenizer.padding_side = prev
    attn = enc["attention_mask"]
    if side == "right":
        last_index = attn.sum(dim=1) - 1
    else:
        last_index = torch.full((attn.shape[0],), attn.shape[1] - 1, dtype=torch.long)
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc, last_index.to(device), raw_lengths


def encoder_for(fmt):
    return chat_encode_batch if fmt == "chat" else P.encode_batch


def readout(model, tokenizer, prompts, r_ids, c_ids, fmt, batch_size=BATCH, max_len=MAX_LEN):
    enc_fn = encoder_for(fmt)
    device = next(model.parameters()).device
    r = torch.as_tensor(r_ids, dtype=torch.long, device=device)
    c = torch.as_tensor(c_ids, dtype=torch.long, device=device)
    vals = np.zeros(len(prompts), dtype=np.float64)
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc, last_index, _raw = enc_fn(tokenizer, chunk, max_len=max_len, device=device, side="right")
        with torch.no_grad():
            out = model(**enc)
        rows = torch.arange(len(chunk), device=device)
        logits = out.logits[rows, last_index].float()
        logp = torch.log_softmax(logits, dim=-1)
        vals[start:start + len(chunk)] = (logp[:, r].mean(dim=-1) - logp[:, c].mean(dim=-1)).cpu().numpy()
        del out, logits, logp
    return vals


def mu_stats(model, tokenizer, prompts, layer, fmt, batch_size=BATCH, max_len=MAX_LEN):
    enc_fn = encoder_for(fmt)
    blocks = P.decoder_layers(model)
    store = {}

    def hook(_m, _i, out):
        store["h"] = P.hidden_of(out).detach()

    handle = blocks[int(layer)].register_forward_hook(hook)
    device = next(model.parameters()).device
    tot, cnt = 0.0, 0
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            enc, _li, _raw = enc_fn(tokenizer, chunk, max_len=max_len, device=device, side="right")
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


def swept_gaps(model, tokenizer, prompts, direction, mu, c_values, r_ids, c_ids,
               layer, fmt, batch_size=BATCH, log=None):
    out = np.zeros((len(c_values), len(prompts)), dtype=np.float64)
    for i, cval in enumerate(c_values):
        t0 = time.time()
        if abs(float(cval)) < 1e-12:
            handles = []
        else:
            handles = ga.steer_hook(model, direction, layer, cval, mu)
        try:
            out[i] = readout(model, tokenizer, prompts, r_ids, c_ids, fmt, batch_size=batch_size)
        finally:
            P.remove_hooks(handles)
        if log is not None:
            log("    c=%+.2f mean_gap=%+.4f (%.1fs)" % (cval, float(out[i].mean()), time.time() - t0))
    return out


def sweep_with_extension(model, tokenizer, prompts, direction, mu, c_values, r_ids,
                         c_ids, layer, fmt, extra=None, target=0.5, batch_size=BATCH, log=None):
    extra = list(EXTRA_C if extra is None else extra)
    gaps = swept_gaps(model, tokenizer, prompts, direction, mu, c_values, r_ids, c_ids,
                      layer, fmt, batch_size=batch_size, log=log)
    c_used = list(c_values)
    for ce in extra:
        rates, _d, _dm, _n = ga.curve_from_gaps(gaps, c_used, c_used.index(0.0))
        best = float(np.nanmax(rates)) if np.isfinite(rates).any() else 0.0
        if best >= target:
            break
        if log is not None:
            log("    extending grid to c=%.1f (max crossing %.3f)" % (ce, best))
        gk = swept_gaps(model, tokenizer, prompts, direction, mu, [ce], r_ids, c_ids,
                        layer, fmt, batch_size=batch_size, log=log)
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


def generate(model, tokenizer, prompts, fmt, direction, mu, coeff, layer,
             max_new=GEN_NEW, batch_size=GEN_BATCH):
    enc_fn = encoder_for(fmt)
    device = next(model.parameters()).device
    if abs(float(coeff)) < 1e-12:
        handles = []
    else:
        handles = ga.steer_hook(model, direction, layer, coeff, mu)
    outs = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            enc, _last, _raw = enc_fn(tokenizer, chunk, max_len=MAX_LEN, device=device, side="left")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tokenizer.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
    finally:
        P.remove_hooks(handles)
    return outs


def _fit_base_direction(model, tok, data, layer, pth, log, batch_size=BATCH):
    fit_prompts = list(data["harm_train"]) + list(data["ben_train"])
    fit_labels = np.array([1] * len(data["harm_train"]) + [0] * len(data["ben_train"]))
    t0 = time.time()
    pack = P.extract_activations(model, tok, fit_prompts, [layer], batch_size=batch_size)
    log("fit activations %s in %.1fs; trunc frac %.4f"
        % (str(pack["acts"].shape), time.time() - t0, pack["frac_trunc"]))
    v_mm, mm_norm = P.refusal_direction(pack["acts"], fit_labels, layer_index=0)
    v_lr, lr_info = ga.fit_logistic_direction(pack["acts"][0], fit_labels)
    if float(np.dot(v_mm.astype(np.float64), v_lr.astype(np.float64))) < 0:
        v_lr = (-v_lr).astype(np.float32)
        lr_info["flipped_to_massmean"] = True
    cos = float(np.dot(v_mm.astype(np.float64), v_lr.astype(np.float64)))
    log("massmean norm %.3f | logistic n_iter %d acc %.3f | cos(mm,lr) %.4f"
        % (mm_norm, lr_info["n_iter"], lr_info["train_acc"], cos))
    np.savez_compressed(pth["dirs_npz"], massmean=v_mm, logistic=v_lr, layer=layer)
    meta = {
        "fit_layer": int(layer), "massmean_raw_norm": float(mm_norm),
        "logistic": lr_info, "cos_massmean_logistic": cos,
        "d_model": int(v_mm.shape[0]), "fit_repo": FIT_REPO,
        "fit_trunc_frac": float(pack["frac_trunc"]),
        "n_train": len(data["harm_train"]),
    }
    P.write_json(pth["dirs"], meta)
    del pack
    gc.collect()
    return v_mm, v_lr, meta


def e5_worker(art=E5_ART, ga_art=GA_ART, batch_size=BATCH, n_rand=N_RAND,
              n_gen=N_GEN, do_generation=True):
    pth = paths(art)
    ga_pth = ga.paths(ga_art)

    def log(msg):
        return P.elog(pth["log"], msg)

    groups = [("base", "Qwen/Qwen3-8B-Base", ["neutral"]),
              ("instruct", "Qwen/Qwen3-8B", ["neutral", "chat"])]
    t_all = time.time()
    status = {"stage": "start", "t0": t_all, "done": [], "error": None}
    P.write_json(pth["status"], status)
    try:
        P.set_seed(SEED)
        data = P.load_prompts(n_train=N_TRAIN, n_held=N_HELD, seed=SPLIT_SEED,
                              cache_path=pth["prompts"])
        fp = ga.split_fingerprint(data)
        log("=== E5 second family start; refusal split fingerprint %s" % fp)
        harm_held = list(data["harm_held"])
        gen_prompts = harm_held[:n_gen]
        zero_index = C_GRID.index(0.0)

        v_mm = v_lr = None
        fit_layer = None
        if os.path.exists(pth["dirs_npz"]):
            with np.load(pth["dirs_npz"]) as blob:
                v_mm = np.asarray(blob["massmean"], dtype=np.float32)
                v_lr = np.asarray(blob["logistic"], dtype=np.float32)
                fit_layer = int(blob["layer"])

        for mlabel, repo, fmts in groups:
            need = [f for f in fmts if not (os.path.exists(pth["arm"]("%s-%s" % (mlabel, f)))
                    and P.read_json(pth["arm"]("%s-%s" % (mlabel, f))).get("complete"))]
            if not need:
                log("skip %s (all formats complete)" % mlabel)
                continue

            status["stage"] = "load:" + mlabel
            P.write_json(pth["status"], status)
            commit = resolve_commit(repo)
            t0 = time.time()
            log("loading %s @ %s" % (repo, commit[:10]))
            tok = P.load_tokenizer(repo, revision=commit)
            model = P.load_model(repo, revision=commit)
            n_layers = len(P.decoder_layers(model))
            layer = prop_layer(n_layers)
            d_model = int(model.config.hidden_size)
            log("loaded %s in %.1fs; n_layers=%d prop_layer=%d d_model=%d"
                % (mlabel, time.time() - t0, n_layers, layer, d_model))

            r_ids, r_single, r_dec = P.onset_token_ids(tok, P.REFUSAL_STRS)
            c_ids, c_single, c_dec = P.onset_token_ids(tok, P.COMPLY_STRS)

            if repo == FIT_REPO and v_mm is None:
                status["stage"] = "fit_direction"
                P.write_json(pth["status"], status)
                v_mm, v_lr, _meta = _fit_base_direction(model, tok, data, layer, pth, log, batch_size)
                fit_layer = layer
            if v_mm is None:
                raise RuntimeError("qwen_dirs.npz missing: base (fit repo) must run first")
            if layer != fit_layer:
                log("WARN steer layer %d != fit layer %d" % (layer, fit_layer))

            for fmt in need:
                arm = "%s-%s" % (mlabel, fmt)
                status["stage"] = "mu:" + arm
                P.write_json(pth["status"], status)
                mus = mu_stats(model, tok, harm_held, layer, fmt, batch_size=batch_size)
                mu = mus["mu_alltoken"]
                log("%s mu_L%d(%s) = %.3f" % (arm, layer, fmt, mu))

                rec = {
                    "arm": arm, "model": mlabel, "repo": repo, "commit": commit,
                    "fmt": fmt, "fit_layer": int(fit_layer), "steer_layer": int(layer),
                    "n_layers": n_layers, "d_model": d_model, "mu_stats": mus, "mu_used": mu,
                    "c_grid": C_GRID, "split_fingerprint": fp, "seed": SEED,
                    "prop_layer_rule": "round(20/32 * n_layers)",
                    "refusal_ids": r_ids, "refusal_decoded": r_dec,
                    "comply_ids": c_ids, "comply_decoded": c_dec,
                    "families": {},
                }
                margin = None
                fam_c50 = None
                for fam, vec in (("massmean", v_mm), ("logistic", v_lr)):
                    status["stage"] = "sweep_%s:%s" % (fam, arm)
                    P.write_json(pth["status"], status)
                    t0 = time.time()
                    gaps, c_used = sweep_with_extension(model, tok, harm_held, vec, mu,
                                                        C_GRID, r_ids, c_ids, layer, fmt,
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
                        "efficacy_ci": eff_ci, "abs_c": eff["abs_c"], "disp": eff["disp"],
                        "margin": float(g0.mean()),
                        "frac_refusing_unsteered": float((g0 > 0).mean()),
                        "c50": c50, "d50": d50, "n_eligible": int(n_elig),
                        "c_used": c_used, "rates": rates.tolist(), "gaps": gaps.tolist(),
                    }
                    log("%s %-8s margin=%+.3f efficacy=%.4f [%.3f,%.3f] c50=%.3f (%.1fs)"
                        % (arm, fam, float(g0.mean()), eff["efficacy"], eff_ci["lo"],
                           eff_ci["hi"], c50, time.time() - t0))

                mm = rec["families"]["massmean"]
                g0_mm = np.asarray(mm["gaps"])[mm["c_used"].index(0.0)]
                m, mlo, mhi = P.bootstrap_ci(g0_mm, n_boot=1000, seed=SEED)
                rec["margin"] = float(m)
                rec["margin_ci"] = [float(mlo), float(mhi)]

                status["stage"] = "null:" + arm
                P.write_json(pth["status"], status)
                t0 = time.time()
                rands = P.random_unit_directions(d_model, n_rand, seed=SEED)
                null_effs = []
                for k in range(n_rand):
                    gk = swept_gaps(model, tok, harm_held, rands[k], mu, C_GRID, r_ids,
                                    c_ids, layer, fmt, batch_size=batch_size, log=None)
                    null_effs.append(efficacy_from_gaps(gk, C_GRID, zero_index)["efficacy"])
                null_effs = np.asarray(null_effs, dtype=np.float64)
                z, nmean, nsd = P.zscore_vs_null(mm["efficacy"], null_effs)
                rec["efficacy_z"] = float(z)
                rec["null_efficacy_mean"] = float(nmean)
                rec["null_efficacy_sd"] = float(nsd)
                rec["null_efficacies"] = null_effs.tolist()
                log("%s null eff %.4f+-%.4f real %.4f z=%.1f (%.1fs)"
                    % (arm, nmean, nsd, mm["efficacy"], z, time.time() - t0))

                if do_generation and np.isfinite(fam_c50):
                    status["stage"] = "generate:" + arm
                    P.write_json(pth["status"], status)
                    t0 = time.time()
                    txt0 = generate(model, tok, gen_prompts, fmt, v_mm, mu, 0.0, layer,
                                    max_new=GEN_NEW, batch_size=GEN_BATCH)
                    txtc = generate(model, tok, gen_prompts, fmt, v_mm, mu, fam_c50, layer,
                                    max_new=GEN_NEW, batch_size=GEN_BATCH)
                    ref0 = [bool(P.is_refusal(t)) for t in txt0]
                    refc = [bool(P.is_refusal(t)) for t in txtc]
                    rec["behavioural"] = {
                        "coeff": float(fam_c50), "n": len(gen_prompts),
                        "max_new_tokens": GEN_NEW,
                        "unsteered_refusal_rate": float(np.mean(ref0)),
                        "steered_refusal_rate": float(np.mean(refc)),
                        "unsteered_refusal": ref0, "steered_refusal": refc,
                        "unsteered_think": ga.unclosed_think_fraction(txt0),
                        "steered_think": ga.unclosed_think_fraction(txtc),
                    }
                    rec["gen_prompts"] = gen_prompts
                    rec["gen_unsteered_text"] = txt0
                    rec["gen_steered_text"] = txtc
                    log("%s gen refusal %.2f -> %.2f at c=%.3f (%.1fs)"
                        % (arm, float(np.mean(ref0)), float(np.mean(refc)), fam_c50,
                           time.time() - t0))
                else:
                    rec["behavioural"] = {"skipped": "c50 not reached"}

                rec["elapsed_seconds"] = time.time() - t_all
                rec["complete"] = True
                P.write_json(pth["arm"](arm), rec)
                log("%s DONE margin=%+.3f [%.3f,%.3f] eff_mm=%.4f z=%.1f"
                    % (arm, rec["margin"], mlo, mhi, mm["efficacy"], rec["efficacy_z"]))
                status["done"].append(arm)
                P.write_json(pth["status"], status)

            P.free_model(model)
            del tok
            freed = P.purge_hf_cache(repo)
            log("%s freed %.2f GB cache" % (mlabel, freed / 1e9))

        status["stage"] = "phase1_done"
        status["phase1_seconds"] = time.time() - t_all
        P.write_json(pth["status"], status)
        log("=== E5 phase1 done in %.1fs" % (time.time() - t_all))
    except Exception as exc:
        import traceback
        status["stage"] = "error"
        status["error"] = repr(exc)
        P.write_json(pth["status"], status)
        P.elog(pth["log"], "ERROR " + repr(exc))
        P.elog(pth["log"], traceback.format_exc())


def e5_judge(art=E5_ART, repo=e2.HARMBENCH_REPO, n_judge=N_JUDGE, batch_size=8):
    """Phase 2: HarmBench-cls genuine-harm check on unsteered(c=0) and steered
    (c50) generations for every arm. Reuses e2_behavioural's judge harness."""
    pth = paths(art)

    def log(msg):
        return P.elog(pth["log"], msg)

    status = P.read_json(pth["status"]) if os.path.exists(pth["status"]) else {}
    status["stage"] = "judge:load"
    P.write_json(pth["status"], status)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        import transformers.utils.import_utils as _iu
        _iu.is_sentencepiece_available.cache_clear()
    except Exception:
        pass
    t0 = time.time()
    log("loading HarmBench cls %s" % repo)
    cls_tok = AutoTokenizer.from_pretrained(repo, use_fast=False)
    if cls_tok.pad_token is None:
        cls_tok.pad_token = cls_tok.eos_token
    cls_model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to("cuda")
    cls_model.eval()
    log("loaded HarmBench cls in %.1fs" % (time.time() - t0))
    try:
        for f in sorted(os.listdir(art)):
            if not (f.startswith("e5_arm_") and f.endswith(".json")):
                continue
            p = os.path.join(art, f)
            rec = P.read_json(p)
            if rec.get("judge_complete") or "gen_prompts" not in rec:
                continue
            arm = rec["arm"]
            status["stage"] = "judge:" + arm
            P.write_json(pth["status"], status)
            beh = rec["gen_prompts"][:n_judge]
            out = {}
            for key, txt in (("unsteered", rec["gen_unsteered_text"]),
                             ("steered", rec["gen_steered_text"])):
                verdicts = e2._harmbench_judge_batch(cls_model, cls_tok, beh,
                                                     txt[:n_judge], batch_size=batch_size)
                out[key + "_harmful"] = verdicts
                out[key + "_harmful_rate"] = float(np.mean(verdicts)) if verdicts else float("nan")
            rec["harmbench"] = out
            rec["harmbench_repo"] = repo
            rec["judge_complete"] = True
            P.write_json(p, rec)
            log("%s HarmBench harmful: unsteered %.2f -> steered %.2f"
                % (arm, out["unsteered_harmful_rate"], out["steered_harmful_rate"]))
        status["stage"] = "judge_done"
        P.write_json(pth["status"], status)
        aggregate(art)
        log("=== E5 judge done")
    finally:
        P.free_model(cls_model)
        del cls_tok
        P.purge_hf_cache(repo)


def aggregate(art=E5_ART):
    pth = paths(art)
    arms = {}
    for f in sorted(os.listdir(art)):
        if f.startswith("e5_arm_") and f.endswith(".json"):
            r = P.read_json(os.path.join(art, f))
            if r.get("complete"):
                arms[r["arm"]] = r
    order = ["base-neutral", "instruct-neutral", "instruct-chat"]
    present = [a for a in order if a in arms]

    def eff(a, fam):
        return arms[a]["families"][fam]["efficacy"]

    margins = {a: arms[a]["margin"] for a in present}
    out = {
        "experiment": "E5 - second model family (Qwen3-8B base vs instruct)",
        "definition": {
            "margin": "mean unsteered refusal logit gap on held-out harmful prompts",
            "efficacy": "OLS slope of mean displacement of refusal gap vs |c|, |c|<=2",
            "steer_param": "h_l <- h_l + c*mu_l*v_hat at the proportional layer",
            "direction": "Qwen3-8B-Base diff-in-means (+logistic) refusal direction, carried unchanged",
            "format": "base+instruct neutral scaffold is format-matched; instruct-chat "
                      "uses the model's own template (enable_thinking=False) as a caveat"},
        "arms": present,
        "split_fingerprint": arms[present[0]]["split_fingerprint"] if present else None,
        "fit_layer": arms[present[0]]["fit_layer"] if present else None,
        "n_layers": arms[present[0]]["n_layers"] if present else None,
        "seed": SEED,
        "margins": margins,
        "efficacy_massmean": {a: eff(a, "massmean") for a in present},
        "efficacy_logistic": {a: eff(a, "logistic") for a in present},
        "efficacy_ci_massmean": {a: arms[a]["families"]["massmean"]["efficacy_ci"] for a in present},
        "efficacy_z": {a: arms[a]["efficacy_z"] for a in present},
        "c50_massmean": {a: arms[a]["families"]["massmean"]["c50"] for a in present},
        "behavioural": {a: arms[a].get("behavioural") for a in present},
        "harmbench": {a: arms[a].get("harmbench") for a in present},
    }
    # format-matched replication check: base-neutral vs instruct-neutral
    if "base-neutral" in margins and "instruct-neutral" in margins:
        b, i = margins["base-neutral"], margins["instruct-neutral"]
        out["margin_growth_neutral"] = float(i / b) if b else None
        eb = eff("base-neutral", "massmean")
        ei = eff("instruct-neutral", "massmean")
        out["efficacy_ratio_neutral"] = float(ei / eb) if eb else None
    P.write_json(pth["final"], out)
    return out


def e5_launch(art=E5_ART, **kw):
    import threading
    for th in threading.enumerate():
        if th.name == "e5-worker" and th.is_alive():
            return {"launched": False, "reason": "e5-worker already alive"}
    P.run_detached(lambda: e5_worker(art=art, **kw), name="e5-worker")
    return {"launched": True}


def e5_judge_launch(art=E5_ART, **kw):
    import threading
    for th in threading.enumerate():
        if th.name == "e5-judge" and th.is_alive():
            return {"launched": False, "reason": "e5-judge already alive"}
    P.run_detached(lambda: e5_judge(art=art, **kw), name="e5-judge")
    return {"launched": True}
