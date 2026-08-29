"""E5-extra: extend second-family generalization to Llama-3.1-8B and gemma-2-9b.

Uses the unsloth FULL-PRECISION mirrors because the official Meta/Google repos
are license-gated (user-approved). INTEGRITY CAVEAT (recorded in every result
file): these are third-party mirrors, not the official repos; the exact mirror
commit is pinned per model. Before fitting, verify_fp() asserts the model loaded
in bf16/fp16 with the expected parameter count and NO quantization config -- a
4-bit load would corrupt margin/efficacy, so a failed check STOPS that family.

Same protocol as the Qwen E5 run (reuses e5_family + e2_behavioural exactly):
fit refusal direction on base (diff-in-means + logistic tol=1e-10) at the layer
proportional to OLMo L20 (round(20/32*n_layers): Llama 32->20, Gemma 42->26),
carry unchanged to instruct; margin + per-dose efficacy (|c|<=2 slope) with
bootstrap CIs + 20-direction random null (report z); refusal-onset behavioural
crossing at each model's c50; HarmBench-cls genuine-harm at c50 on base and
instruct (llama-tokenizer fallback); base neutral, instruct neutral AND its own
chat template. Detached-thread safe (P10). ASCII only.
"""

import gc
import json
import os
import time

import numpy as np
import torch

import pipeline as P
import gate_a as ga
import e5_family as e5
import e2_behavioural as e2
from e1_causal import efficacy_from_gaps

GA_ART = "/marimo/gateA"
MAX_LEN = ga.MAX_LEN
SEED = ga.SEED
SPLIT_SEED = ga.SPLIT_SEED
BATCH = ga.BATCH
N_RAND = ga.N_RAND
C_GRID = list(ga.C_GRID)

N_TRAIN = 200
N_HELD = 200
N_GEN = 80
N_JUDGE = 40
GEN_BATCH = 16
GEN_NEW = 256

FAMILIES = {
    "llama": {"base": "unsloth/Meta-Llama-3.1-8B",
              "instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
              "art": "/marimo/e5_llama", "attn": None,
              "pmin": 7.5e9, "pmax": 8.6e9},
    "gemma": {"base": "unsloth/gemma-2-9b",
              "instruct": "unsloth/gemma-2-9b-it",
              "art": "/marimo/e5_gemma", "attn": "eager",
              "pmin": 8.7e9, "pmax": 9.7e9},
}


def paths(art):
    P.ensure_dir(art)
    return {
        "art": art,
        "log": os.path.join(art, "log.txt"),
        "status": os.path.join(art, "status.json"),
        "dirs_npz": os.path.join(art, "dirs.npz"),
        "dirs": os.path.join(art, "dirs.json"),
        "arm": lambda lab: os.path.join(art, "arm_%s.json" % lab),
        "final": lambda name: os.path.join(art, "E5_%s.json" % name),
    }


def load_fp(repo, commit, attn=None):
    from transformers import AutoModelForCausalLM
    kw = dict(revision=commit, dtype=torch.bfloat16)
    if attn:
        kw["attn_implementation"] = attn
    m = AutoModelForCausalLM.from_pretrained(repo, **kw).to("cuda")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def verify_fp(model, pmin, pmax):
    """Assert full precision: no quantization config, no 4/8-bit flag, no
    uint8/int8 params, expected parameter count."""
    qc = getattr(model.config, "quantization_config", None)
    is4 = bool(getattr(model, "is_loaded_in_4bit", False))
    is8 = bool(getattr(model, "is_loaded_in_8bit", False))
    dts = {}
    for p in model.parameters():
        dts[str(p.dtype)] = dts.get(str(p.dtype), 0) + 1
    n = int(sum(p.numel() for p in model.parameters()))
    bad_dt = any(dt in ("torch.uint8", "torch.int8") for dt in dts)
    ok = (qc is None) and (not is4) and (not is8) and (not bad_dt) and (pmin <= n <= pmax)
    return bool(ok), {"quantization_config": str(qc), "is_4bit": is4, "is_8bit": is8,
                      "param_count": n, "param_billions": n / 1e9, "dtypes": dts,
                      "pmin": pmin, "pmax": pmax, "ok": bool(ok)}


def measure_arm(model, tok, mlabel, fmt, repo, commit, vinfo, v_mm, v_lr, layer,
                fit_repo, fit_commit, r_ids, r_dec, c_ids, c_dec, d_model, n_layers,
                harm, gen_prompts, fam, fp, pth, log, setstage, do_generation=True):
    arm = "%s-%s" % (mlabel, fmt)
    setstage("mu:" + arm)
    mus = e5.mu_stats(model, tok, harm, layer, fmt, batch_size=BATCH)
    mu = mus["mu_alltoken"]
    log("%s mu_L%d(%s)=%.3f" % (arm, layer, fmt, mu))
    rec = {"arm": arm, "model": mlabel, "family": fam, "repo": repo, "commit": commit,
           "mirror": True, "mirror_note": "third-party unsloth full-precision mirror "
           "(official Meta/Google repos license-gated); pinned commit above",
           "fit_repo": fit_repo, "fit_commit": fit_commit, "fmt": fmt,
           "fit_layer": int(layer), "steer_layer": int(layer), "n_layers": n_layers,
           "d_model": d_model, "mu_stats": mus, "mu_used": mu, "c_grid": C_GRID,
           "split_fingerprint": fp, "seed": SEED, "prop_layer_rule": "round(20/32*n_layers)",
           "verify_fp": vinfo, "refusal_ids": r_ids, "refusal_decoded": r_dec,
           "comply_ids": c_ids, "comply_decoded": c_dec, "families": {}}
    margin = None
    fam_c50 = None
    for famd, vec in (("massmean", v_mm), ("logistic", v_lr)):
        setstage("sweep_%s:%s" % (famd, arm))
        t0 = time.time()
        gaps, cu = e5.sweep_with_extension(model, tok, harm, vec, mu, C_GRID, r_ids,
                                           c_ids, layer, fmt, batch_size=BATCH, log=None)
        zi = cu.index(0.0)
        g0 = gaps[zi]
        if margin is None:
            margin = float(g0.mean())
        eff = efficacy_from_gaps(gaps, cu, zi)
        effci = e5.bootstrap_efficacy(gaps, cu, zi)
        rates, disps, dm2, ne = ga.curve_from_gaps(gaps, cu, zi)
        c50, d50 = ga.fifty_points(gaps, cu, zi)
        if famd == "massmean":
            fam_c50 = c50
        rec["families"][famd] = {"efficacy": eff["efficacy"], "efficacy_ci": effci,
            "margin": float(g0.mean()), "frac_refusing_unsteered": float((g0 > 0).mean()),
            "c50": c50, "d50": d50, "n_eligible": int(ne), "c_used": cu,
            "rates": rates.tolist(), "gaps": gaps.tolist()}
        log("%s %-8s margin=%+.3f eff=%.4f [%.3f,%.3f] c50=%.3f (%.1fs)"
            % (arm, famd, float(g0.mean()), eff["efficacy"], effci["lo"], effci["hi"],
               c50, time.time() - t0))
    mm = rec["families"]["massmean"]
    g0mm = np.asarray(mm["gaps"])[mm["c_used"].index(0.0)]
    m, mlo, mhi = P.bootstrap_ci(g0mm, n_boot=1000, seed=SEED)
    rec["margin"] = float(m)
    rec["margin_ci"] = [float(mlo), float(mhi)]
    setstage("null:" + arm)
    t0 = time.time()
    rands = P.random_unit_directions(d_model, N_RAND, seed=SEED)
    ne_ = []
    for k in range(N_RAND):
        gk = e5.swept_gaps(model, tok, harm, rands[k], mu, C_GRID, r_ids, c_ids,
                           layer, fmt, batch_size=BATCH, log=None)
        ne_.append(efficacy_from_gaps(gk, C_GRID, C_GRID.index(0.0))["efficacy"])
    ne_ = np.asarray(ne_, dtype=np.float64)
    z, nm, nsd = P.zscore_vs_null(mm["efficacy"], ne_)
    rec["efficacy_z"] = float(z)
    rec["null_efficacy_mean"] = float(nm)
    rec["null_efficacy_sd"] = float(nsd)
    rec["null_efficacies"] = ne_.tolist()
    log("%s null eff %.4f+-%.4f real %.4f z=%.1f (%.1fs)"
        % (arm, nm, nsd, mm["efficacy"], z, time.time() - t0))
    if do_generation and np.isfinite(fam_c50):
        setstage("gen:" + arm)
        t0 = time.time()
        txt0 = e5.generate(model, tok, gen_prompts, fmt, v_mm, mu, 0.0, layer,
                           max_new=GEN_NEW, batch_size=GEN_BATCH)
        txtc = e5.generate(model, tok, gen_prompts, fmt, v_mm, mu, fam_c50, layer,
                           max_new=GEN_NEW, batch_size=GEN_BATCH)
        ref0 = [bool(P.is_refusal(t)) for t in txt0]
        refc = [bool(P.is_refusal(t)) for t in txtc]
        rec["behavioural"] = {"coeff": float(fam_c50), "n": len(gen_prompts),
            "max_new_tokens": GEN_NEW, "unsteered_refusal_rate": float(np.mean(ref0)),
            "steered_refusal_rate": float(np.mean(refc)), "unsteered_refusal": ref0,
            "steered_refusal": refc, "unsteered_think": ga.unclosed_think_fraction(txt0),
            "steered_think": ga.unclosed_think_fraction(txtc)}
        rec["gen_prompts"] = gen_prompts
        rec["gen_unsteered_text"] = txt0
        rec["gen_steered_text"] = txtc
        log("%s gen refusal %.2f->%.2f at c=%.3f (%.1fs)"
            % (arm, float(np.mean(ref0)), float(np.mean(refc)), fam_c50, time.time() - t0))
    else:
        rec["behavioural"] = {"skipped": "c50 not reached"}
    rec["complete"] = True
    P.write_json(pth["arm"](arm), rec)
    log("%s DONE margin=%+.3f eff_mm=%.4f z=%.1f"
        % (arm, rec["margin"], mm["efficacy"], rec["efficacy_z"]))


def run_family(fam, do_generation=True):
    cfg = FAMILIES[fam]
    pth = paths(cfg["art"])

    def log(m):
        return P.elog(pth["log"], m)

    st = {"stage": "start", "done": [], "error": None, "family": fam}

    def setstage(s):
        st["stage"] = s
        P.write_json(pth["status"], st)

    setstage("start")
    t_all = time.time()
    try:
        P.set_seed(SEED)
        data = P.load_prompts(n_train=N_TRAIN, n_held=N_HELD, seed=SPLIT_SEED,
                              cache_path=os.path.join(GA_ART, "prompts.json"))
        fp = ga.split_fingerprint(data)
        harm = list(data["harm_held"])
        gen_prompts = harm[:N_GEN]
        log("=== E5-%s start; split %s" % (fam, fp))
        v_mm = v_lr = None
        fit_layer = None
        fit_commit = None
        if os.path.exists(pth["dirs_npz"]):
            with np.load(pth["dirs_npz"]) as b:
                v_mm = np.asarray(b["massmean"], dtype=np.float32)
                v_lr = np.asarray(b["logistic"], dtype=np.float32)
                fit_layer = int(b["layer"])
            if os.path.exists(pth["dirs"]):
                fit_commit = P.read_json(pth["dirs"]).get("fit_commit")
        from huggingface_hub import HfApi
        groups = [("base", cfg["base"], ["neutral"]),
                  ("instruct", cfg["instruct"], ["neutral", "chat"])]
        for mlabel, repo, fmts in groups:
            need = [f for f in fmts if not (os.path.exists(pth["arm"]("%s-%s" % (mlabel, f)))
                    and P.read_json(pth["arm"]("%s-%s" % (mlabel, f))).get("complete"))]
            if not need:
                log("skip %s (all formats complete)" % mlabel)
                continue
            commit = HfApi().model_info(repo).sha
            setstage("load:" + mlabel)
            t0 = time.time()
            log("loading MIRROR %s @ %s" % (repo, commit[:10]))
            tok = P.load_tokenizer(repo, revision=commit)
            model = load_fp(repo, commit, cfg["attn"])
            ok, vinfo = verify_fp(model, cfg["pmin"], cfg["pmax"])
            log("%s verify_fp ok=%s params=%.3fB dtypes=%s qc=%s"
                % (mlabel, ok, vinfo["param_billions"], vinfo["dtypes"], vinfo["quantization_config"]))
            if not ok:
                setstage("error")
                st["error"] = "FULL-PRECISION CHECK FAILED for %s: %s" % (repo, json.dumps(vinfo))
                P.write_json(pth["status"], st)
                log("STOP family %s: verify_fp FAILED (quantized/precision) %s" % (fam, vinfo))
                P.free_model(model)
                return
            n_layers = len(P.decoder_layers(model))
            layer = e5.prop_layer(n_layers)
            d_model = int(model.config.hidden_size)
            log("%s loaded %.1fs n_layers=%d prop_layer=%d d_model=%d"
                % (mlabel, time.time() - t0, n_layers, layer, d_model))
            r_ids, _rs, r_dec = P.onset_token_ids(tok, P.REFUSAL_STRS)
            c_ids, _cs, c_dec = P.onset_token_ids(tok, P.COMPLY_STRS)
            if repo == cfg["base"] and v_mm is None:
                setstage("fit")
                pack = P.extract_activations(model, tok, list(data["harm_train"]) + list(data["ben_train"]),
                                             [layer], batch_size=BATCH)
                labels = np.array([1] * len(data["harm_train"]) + [0] * len(data["ben_train"]))
                v_mm, mmn = P.refusal_direction(pack["acts"], labels, layer_index=0)
                v_lr, lri = ga.fit_logistic_direction(pack["acts"][0], labels)
                if float(np.dot(v_mm.astype(np.float64), v_lr.astype(np.float64))) < 0:
                    v_lr = (-v_lr).astype(np.float32)
                cos = float(np.dot(v_mm.astype(np.float64), v_lr.astype(np.float64)))
                np.savez_compressed(pth["dirs_npz"], massmean=v_mm, logistic=v_lr, layer=layer)
                fit_layer = layer
                fit_commit = commit
                P.write_json(pth["dirs"], {"fit_layer": layer, "massmean_raw_norm": float(mmn),
                    "logistic": lri, "cos_massmean_logistic": cos, "d_model": d_model,
                    "fit_repo": repo, "fit_commit": commit, "mirror": True,
                    "verify_fp": vinfo, "fit_trunc_frac": float(pack["frac_trunc"])})
                log("fit %s dir L%d massmean_norm=%.3f logistic_iter=%d cos=%.4f"
                    % (fam, layer, mmn, lri["n_iter"], cos))
                del pack
                gc.collect()
            if v_mm is None:
                raise RuntimeError("dirs.npz missing: base must run first")
            for fmt in need:
                measure_arm(model, tok, mlabel, fmt, repo, commit, vinfo, v_mm, v_lr,
                            layer, cfg["base"], fit_commit, r_ids, r_dec, c_ids, c_dec,
                            d_model, n_layers, harm, gen_prompts, fam, fp, pth, log,
                            setstage, do_generation)
                st["done"].append("%s-%s" % (mlabel, fmt))
                P.write_json(pth["status"], st)
            P.free_model(model)
            del tok
            freed = P.purge_hf_cache(repo)
            log("%s freed %.2f GB" % (mlabel, freed / 1e9))
        setstage("phase1_done")
        st["phase1_seconds"] = time.time() - t_all
        P.write_json(pth["status"], st)
        log("=== E5-%s phase1 done in %.1fs" % (fam, time.time() - t_all))
    except Exception as exc:
        import traceback
        st["stage"] = "error"
        st["error"] = repr(exc)
        P.write_json(pth["status"], st)
        P.elog(pth["log"], "ERROR " + repr(exc))
        P.elog(pth["log"], traceback.format_exc())


def judge_family(fam, cls_tok_repo="hf-internal-testing/llama-tokenizer",
                 cls_repo=None, n_judge=N_JUDGE, batch_size=8):
    cls_repo = cls_repo or e2.HARMBENCH_REPO
    pth = paths(FAMILIES[fam]["art"])

    def log(m):
        return P.elog(pth["log"], m)

    js = os.path.join(pth["art"], "judge_status.json")
    P.write_json(js, {"stage": "load", "err": None})
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        log("judge: loading fallback tok %s + weights %s" % (cls_tok_repo, cls_repo))
        tok = AutoTokenizer.from_pretrained(cls_tok_repo)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(cls_repo, dtype=torch.bfloat16).to("cuda")
        mdl.eval()
        log("judge: loaded cls in %.1fs" % (time.time() - t0))
        for f in sorted(os.listdir(pth["art"])):
            if not (f.startswith("arm_") and f.endswith(".json")):
                continue
            p = os.path.join(pth["art"], f)
            rec = P.read_json(p)
            if rec.get("judge_complete") or "gen_prompts" not in rec:
                continue
            arm = rec["arm"]
            P.write_json(js, {"stage": "judge:" + arm})
            beh = rec["gen_prompts"][:n_judge]
            out = {}
            for key, txt in (("unsteered", rec["gen_unsteered_text"]),
                             ("steered", rec["gen_steered_text"])):
                v = e2._harmbench_judge_batch(mdl, tok, beh, txt[:n_judge], batch_size=batch_size)
                out[key + "_harmful"] = v
                out[key + "_harmful_rate"] = float(np.mean(v)) if v else float("nan")
            rec["harmbench"] = out
            rec["harmbench_repo"] = cls_repo
            rec["harmbench_tokenizer"] = cls_tok_repo
            rec["judge_complete"] = True
            P.write_json(p, rec)
            log("judge %s HarmBench harmful: unsteered %.2f -> steered %.2f"
                % (arm, out["unsteered_harmful_rate"], out["steered_harmful_rate"]))
        P.write_json(js, {"stage": "judge_done"})
        aggregate(fam)
        log("=== E5-%s judge done" % fam)
        P.free_model(mdl)
        del tok
        P.purge_hf_cache(cls_repo)
    except Exception as exc:
        import traceback
        P.write_json(js, {"stage": "error", "err": repr(exc)})
        P.elog(pth["log"], "JUDGE ERROR " + repr(exc))
        P.elog(pth["log"], traceback.format_exc())


def aggregate(fam):
    pth = paths(FAMILIES[fam]["art"])
    arms = {}
    for f in sorted(os.listdir(pth["art"])):
        if f.startswith("arm_") and f.endswith(".json"):
            r = P.read_json(os.path.join(pth["art"], f))
            if r.get("complete"):
                arms[r["arm"]] = r
    order = ["base-neutral", "instruct-neutral", "instruct-chat"]
    present = [a for a in order if a in arms]

    def eff(a, f):
        return arms[a]["families"][f]["efficacy"]

    margins = {a: arms[a]["margin"] for a in present}
    out = {
        "experiment": "E5 - second family (%s, unsloth full-precision mirror)" % fam,
        "family": fam,
        "mirror_caveat": "third-party unsloth mirrors used because official "
                         "Meta/Google repos are license-gated; commits pinned per arm; "
                         "full-precision verified (verify_fp in each arm record). "
                         "Caveat belongs in the paper limitations.",
        "definition": {"margin": "mean unsteered refusal logit gap on held-out harmful",
            "efficacy": "OLS slope of displacement vs |c|, |c|<=2",
            "steer_param": "h_l <- h_l + c*mu_l*v_hat at proportional layer",
            "direction": "base diff-in-means (+logistic) refusal direction, carried unchanged",
            "format": "base+instruct neutral is format-matched; instruct-chat uses own template"},
        "arms": present,
        "base_repo": FAMILIES[fam]["base"], "instruct_repo": FAMILIES[fam]["instruct"],
        "commits": {a: arms[a]["commit"] for a in present},
        "fit_layer": arms[present[0]]["fit_layer"] if present else None,
        "n_layers": arms[present[0]]["n_layers"] if present else None,
        "d_model": arms[present[0]]["d_model"] if present else None,
        "split_fingerprint": arms[present[0]]["split_fingerprint"] if present else None,
        "verify_fp": {a: arms[a]["verify_fp"] for a in present},
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
    if "base-neutral" in margins and "instruct-neutral" in margins:
        b, i = margins["base-neutral"], margins["instruct-neutral"]
        out["margin_growth_neutral"] = float(i / b) if b else None
        eb, ei = eff("base-neutral", "massmean"), eff("instruct-neutral", "massmean")
        out["efficacy_ratio_neutral"] = float(ei / eb) if eb else None
    if "instruct-chat" in margins and "base-neutral" in margins:
        out["margin_growth_chat"] = float(margins["instruct-chat"] / margins["base-neutral"])
    P.write_json(pth["final"](fam), out)
    return out


def run_family_launch(fam, **kw):
    import threading
    nm = "e5x-%s" % fam
    for th in threading.enumerate():
        if th.name == nm and th.is_alive():
            return {"launched": False, "reason": nm + " alive"}
    P.run_detached(lambda: run_family(fam, **kw), name=nm)
    return {"launched": True, "family": fam}


def judge_launch(fam, **kw):
    import threading
    nm = "e5x-judge-%s" % fam
    for th in threading.enumerate():
        if th.name == nm and th.is_alive():
            return {"launched": False, "reason": nm + " alive"}
    P.run_detached(lambda: judge_family(fam, **kw), name=nm)
    return {"launched": True, "family": fam}
