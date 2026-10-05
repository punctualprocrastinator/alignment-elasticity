# -*- coding: utf-8 -*-
"""Does the per-example reorganisation happen for a SECOND concept?

One direction in one family is an anecdote. This asks whether the per-prompt
profile reorganisation we measured for REFUSAL also happens for HONESTY, using
the committed E4 sweeps (per-model `families.massmean.gaps`, 23 doses x 200
held-out true/false statements, base honesty direction carried unchanged).

Two outcomes, both informative and they say different things:

  BOTH reorganise   -> supervised fine-tuning reorganises per-example structure
                       generally; a property of SFT rather than of refusal.
  ONLY refusal      -> SFT reorganises the per-example structure of the concept
                       it OPTIMISES, while leaving an untargeted concept's
                       structure intact. That is sharper and more surprising: it
                       predicts which features should reorganise under a given
                       fine-tune.

Honesty also has a much flatter mean lever than refusal (CV 0.021 vs 0.078), so
if its profile reorganises too, the gap between "aggregate says preserved" and
"per-example says destroyed" is even wider there than in the headline result.

Note honesty covers 5 checkpoints (base, SFT-1k, DPO, RLVR-last, Instruct) and
has NO RL-Zero arm, so the SFT-vs-RL control is not available for it.

Zero GPU.
"""
import json, os
import numpy as np

E4 = {"base": "results/e4_model_base.json",
      "SFT 1k": "results/e4_model_think-sft-1000.json",
      "DPO": "results/e4_model_think-dpo.json",
      "RLVR last": "results/e4_model_think-rlvr-last.json",
      "Instruct": "results/e4_model_instruct.json"}
REFUSAL = [("base", "base"), ("SFT 1k", "think-sft-1000"), ("DPO", "think-dpo"),
           ("RLVR last", "think-rlvr-last"), ("Instruct", "instruct")]
WINDOW = 1.0


def _rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den > 0 else float("nan")


def profile(gaps, c_grid, window=WINDOW):
    """Through-origin per-prompt displacement slope (same estimator as lever.py)."""
    g = np.asarray(gaps, float)
    c = np.asarray(c_grid, float)
    zi = int(np.argmin(np.abs(c)))
    m = (c < 0) & (np.abs(c) <= window)
    if m.sum() < 2:
        raise RuntimeError("window too narrow: %d doses" % m.sum())
    x = np.abs(c[m])
    D = (g[zi][None, :] - g)[m]
    return (x @ D) / (x @ x)


def honesty_profiles():
    out = {}
    for name, path in E4.items():
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        fam = d["families"]["massmean"]
        out[name] = profile(fam["gaps"], fam["c_used"])
    return out


def refusal_profiles():
    out = {}
    for name, key in REFUSAL:
        for p in ("results/gateA_traj/gateA_sweep_%s.json" % key,
                  "results/gateA_sweep_%s.json" % key):
            if os.path.exists(p):
                d = json.load(open(p))
                out[name] = profile(d["gaps_massmean"], d["c_grid_massmean"])
                break
    return out


def main():
    hon, ref = honesty_profiles(), refusal_profiles()
    print("Per-example profile correlation against base, by concept")
    print("(mean lever CV across the flow: refusal 0.078, honesty 0.021)\n")
    print("%-12s %14s %14s" % ("checkpoint", "REFUSAL rho", "HONESTY rho"))
    rows = {}
    for name in E4:
        r = spearman(ref["base"], ref[name]) if name in ref else float("nan")
        h = spearman(hon["base"], hon[name]) if name in hon else float("nan")
        rows[name] = dict(refusal_rho=r, honesty_rho=h)
        print("%-12s %14.3f %14.3f" % (name, r, h))

    hv = np.array([rows[n]["honesty_rho"] for n in E4 if n != "base"])
    rv = np.array([rows[n]["refusal_rho"] for n in E4 if n != "base"])
    print("\nmean rho over post-base checkpoints: refusal %.3f | honesty %.3f"
          % (rv.mean(), hv.mean()))
    print("at the aligned endpoint:            refusal %.3f | honesty %.3f"
          % (rows["Instruct"]["refusal_rho"], rows["Instruct"]["honesty_rho"]))

    # within-checkpoint dispersion, for the between-vs-within comparison
    print("\nwithin-checkpoint per-prompt CV")
    print("%-12s %14s %14s" % ("checkpoint", "refusal", "honesty"))
    for name in E4:
        rc = ref[name].std() / ref[name].mean() if name in ref else float("nan")
        hc = hon[name].std() / hon[name].mean() if name in hon else float("nan")
        rows[name]["refusal_within_cv"] = float(rc)
        rows[name]["honesty_within_cv"] = float(hc)
        print("%-12s %14.3f %14.3f" % (name, rc, hc))

    json.dump({"experiment": "per-example profile, refusal vs honesty",
               "window": WINDOW, "checkpoints": rows},
              open("results/concept_profiles.json", "w"), indent=1)
    print("\nwrote results/concept_profiles.json")


if __name__ == "__main__":
    main()
