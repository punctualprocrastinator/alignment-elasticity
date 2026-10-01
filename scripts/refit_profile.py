# -*- coding: utf-8 -*-
"""Is the per-prompt reshuffle real, or an artifact of a stale probe?

`scripts/per_prompt_lever.py` showed that the FROZEN base refusal direction's
per-prompt action is completely reorganised by SFT (Spearman vs base: Instruct
-0.004) while RL-Zero preserves it (0.97-1.00), even though the MEAN lever is
conserved (CV 0.078).

The obvious objection: at Instruct the base direction may simply no longer BE
the refusal feature, so what we measured is a stale probe decaying rather than
the model's per-example structure changing. This script settles it. At each
checkpoint we REFIT the difference-in-means direction on the train split, sweep
doses with that refit direction, and recompute the per-prompt profile on the
held-out split.

  Thesis (reorganisation is real): the refit profile ALSO decorrelates from
  base's, so the model's per-example response to its own refusal direction has
  genuinely been reorganised. The finding is probe-independent.

  Negation (stale probe): the refit profile recovers correlation with base's,
  meaning the direction moved but the per-example structure did not. That is a
  weaker and different paper, and it needs knowing BEFORE submission.

Controls:
  * `cos_refit_base` per checkpoint - how far the direction itself moved.
  * A SHUFFLED-LABEL refit arm as a floor: refit on permuted harmful/benign
    labels gives a direction with no refusal content, so its profile correlation
    is the null level any "recovery" must beat.
  * Fit on train, evaluate on held - no in-sample inflation.

Parallel by design: pass a checkpoint slice per box.
    python scripts/refit_profile.py --ckpts base,think-sft-1000,think-sft-15000
Each box writes results/refit_profile_<label>.json independently; merge offline.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P   # noqa: E402
import gate_a as ga    # noqa: E402

GA_ART = "results/gateA_run"
FIT_LAYER = STEER_LAYER = 20
# Dense near zero (where the lever is fit) and out to the aligned crossing doses.
C_GRID = [0.0, -0.125, -0.25, -0.375, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0]

REPOS = {
    "base":              ("allenai/Olmo-3-1025-7B",        "main"),
    "think-sft-1000":    ("allenai/Olmo-3-7B-Think-SFT",   "step1000"),
    "think-sft-15000":   ("allenai/Olmo-3-7B-Think-SFT",   "step15000"),
    "think-sft-43000":   ("allenai/Olmo-3-7B-Think-SFT",   "step43000"),
    "think-dpo":         ("allenai/Olmo-3-7B-Think-DPO",   "main"),
    "think-rlvr-first":  ("allenai/Olmo-3-7B-Think",       "step_100"),
    "think-rlvr-last":   ("allenai/Olmo-3-7B-Think",       "step_1375"),
    "instruct":          ("allenai/Olmo-3-7B-Instruct",    "main"),
    "rlz-math":          ("allenai/Olmo-3-7B-RLZero-Math", "main"),
    "rlz-code":          ("allenai/Olmo-3-7B-RLZero-Code", "main"),
}


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def per_prompt_lever(gaps, c_grid, window=1.0):
    """Through-origin displacement slope for EACH prompt (matches lever.py)."""
    g = np.asarray(gaps, float)
    c = np.asarray(c_grid, float)
    zi = int(np.argmin(np.abs(c)))
    m = (c < 0) & (np.abs(c) <= window)
    x = np.abs(c[m])
    D = (g[zi][None, :] - g)[m]
    return (x @ D) / (x @ x)


def run(labels):
    import torch
    data = P.load_prompts(n_train=200, n_held=200, seed=ga.SPLIT_SEED,
                          cache_path=os.path.join(GA_ART, "prompts.json"))
    fp = ga.split_fingerprint(data)
    if fp != ga.EXPECTED_SPLIT_FP:
        raise RuntimeError("split fingerprint mismatch: %s (STOP)" % fp)

    harm_tr, ben_tr = list(data["harm_train"]), list(data["ben_train"])
    harm_held = list(data["harm_held"])
    fit_prompts = harm_tr + ben_tr
    fit_labels = np.array([1] * len(harm_tr) + [0] * len(ben_tr))
    rng = np.random.default_rng(ga.SEED)
    shuf_labels = rng.permutation(fit_labels)          # null-floor arm

    z = np.load(os.path.join(GA_ART, "directions.npz"))
    v_base = np.asarray(z["direction"], float).ravel()
    v_base /= np.linalg.norm(v_base) + 1e-12
    log("split %s | fit %d | held %d | %d checkpoints"
        % (fp[:8], len(fit_prompts), len(harm_held), len(labels)))

    for label in labels:
        out_path = "results/refit_profile_%s.json" % label
        if os.path.exists(out_path):
            log("skip %s (done)" % label)
            continue
        repo, rev = REPOS[label]
        log("loading %s @ %s" % (repo, rev))
        t0 = time.time()
        model = P.load_model(repo, revision=rev)
        tok = P.load_tokenizer(repo, revision=rev)
        r_ids = P.onset_token_ids(tok, P.REFUSAL_STRS)
        c_ids = P.onset_token_ids(tok, P.COMPLY_STRS)
        mu = ga.residual_norm_stats(model, tok, harm_held, STEER_LAYER)["mu_alltoken"]

        pack = P.extract_activations(model, tok, fit_prompts, [FIT_LAYER],
                                     cache_path=None)
        li = list(pack["layers"]).index(FIT_LAYER)
        v_refit, _ = P.refusal_direction(pack["acts"], fit_labels, layer_index=li)
        v_shuf, _ = P.refusal_direction(pack["acts"], shuf_labels, layer_index=li)
        cos_rb = float(np.dot(v_refit / np.linalg.norm(v_refit), v_base))
        log("%s mu=%.2f cos(refit,base)=%.3f" % (label, mu, cos_rb))

        rec = {"label": label, "repo": repo, "revision": rev,
               "split_fingerprint": fp, "mu_used": float(mu),
               "c_grid": C_GRID, "fit_layer": FIT_LAYER,
               "cos_refit_base": cos_rb,
               "cos_shuffled_base": float(np.dot(v_shuf / np.linalg.norm(v_shuf), v_base)),
               "steer_param": "h += c*mu*v_hat (layer %d)" % STEER_LAYER}

        for arm, vec in (("refit", v_refit), ("shuffled", v_shuf)):
            g = ga.swept_gaps(model, tok, harm_held, vec, mu, C_GRID,
                              r_ids, c_ids, layer=STEER_LAYER)
            rec["gaps_" + arm] = np.asarray(g).tolist()
            rec["per_prompt_lever_" + arm] = per_prompt_lever(g, C_GRID).tolist()
            log("  %s arm swept" % arm)

        rec["elapsed_seconds"] = time.time() - t0
        json.dump(rec, open(out_path, "w"))
        log("%s DONE in %.0fs -> %s" % (label, rec["elapsed_seconds"], out_path))
        P.free_model(model)
        del tok
        torch.cuda.empty_cache()
    log("all assigned checkpoints complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True,
                    help="comma-separated labels, e.g. base,think-dpo,instruct")
    a = ap.parse_args()
    run([s.strip() for s in a.ckpts.split(",") if s.strip()])
