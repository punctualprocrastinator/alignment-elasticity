# -*- coding: utf-8 -*-
"""The lever's MEAN is conserved; its per-prompt PROFILE is destroyed by SFT.

The paper's central measurement is a mean over 200 prompts: "per-dose efficacy
is intact, CV 0.078 across ten checkpoints". That statistic is invariant to a
complete reshuffle of WHICH prompts the direction moves. This script asks
whether the reshuffle happened.

Per-prompt lever = through-origin slope of that prompt's displacement against
|c| on the refusal-reducing branch (same estimator as lever.efficacy, but not
averaged over prompts first).

Result (Spearman against base's profile, |c| <= 1):

    SFT 1k  0.722 | SFT 15k 0.329 | SFT 43k 0.471 | DPO 0.478
    RLVR-f  0.476 | RLVR-l  0.481 | Instruct -0.004
    RL-Zero Math 0.996 | RL-Zero Code 0.967

Two things follow.

1. The profile decays monotonically through the SFT/DPO/RLVR lineage and is
   GONE at Instruct (rho = 0.00), while the two RL-Zero checkpoints -- RL applied
   directly to base, no SFT -- preserve it almost perfectly (rho = 0.97-1.00).
   That is the same SFT-vs-RL dissociation the margin shows, measured on a
   completely different quantity. It is not a saturation-window artifact: at
   |c| <= 2 the numbers are 0.839 / 0.593 / 0.506 / 0.504 / 0.133 / 0.997 / 0.980.

2. The within-checkpoint per-prompt CV is 0.117-0.190 at every checkpoint,
   i.e. LARGER than the 0.078 between-checkpoint CV of the mean that the
   "lever is intact" claim rests on. Between-checkpoint variation being smaller
   than within-checkpoint variation is a far stronger statement than a bare CV,
   and it should be reported as a variance decomposition rather than as a CV of
   point estimates.

Implication for the thesis: "the direction does not go stale" is true of the
mean and false of the structure. At the aligned endpoint the base direction
moves a different set of prompts by the same average amount. Any claim that
c_cross measures a property of the MODEL (rather than of a stale probe) has to
survive refitting the direction per checkpoint -- see the refit arm in
EXPERIMENTS.md RQ2.

Zero GPU: reads the committed Gate-A sweeps.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ORDER = [("base", "base"), ("SFT 1k", "think-sft-1000"), ("SFT 15k", "think-sft-15000"),
         ("SFT 43k", "think-sft-43000"), ("DPO", "think-dpo"),
         ("RLVR first", "think-rlvr-first"), ("RLVR last", "think-rlvr-last"),
         ("Instruct", "instruct"), ("RL-Zero Math", "rlz-math"), ("RL-Zero Code", "rlz-code")]
OUT = "results/per_prompt_lever.json"


def load(l):
    for p in ("results/gateA_traj/gateA_sweep_%s.json" % l, "results/gateA_sweep_%s.json" % l):
        if os.path.exists(p):
            return json.load(open(p))


def per_prompt_lever(d, window=1.0):
    """Through-origin displacement slope for EACH prompt separately."""
    g = np.array(d["gaps_massmean"], float)
    c = np.array(d["c_grid_massmean"], float)
    zi = int(d["zero_index_massmean"])
    m = (c < 0) & (np.abs(c) <= window)
    x = np.abs(c[m])
    D = (g[zi][None, :] - g)[m]          # (doses_in_window, n_prompts)
    return (x @ D) / (x @ x)             # (n_prompts,)


def _rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den > 0 else float("nan")


def main():
    out = {"experiment": "per-prompt lever profile", "windows": [1.0, 2.0], "checkpoints": {}}
    base = {w: per_prompt_lever(load("base"), w) for w in (1.0, 2.0)}
    print("%-14s %10s %10s %11s %11s" %
          ("ckpt", "rho w<=1", "rho w<=2", "within-CV", "mean lever"))
    for name, key in ORDER:
        d = load(key)
        if d is None:
            continue
        v1, v2 = per_prompt_lever(d, 1.0), per_prompt_lever(d, 2.0)
        rec = dict(spearman_vs_base_w1=spearman(base[1.0], v1),
                   spearman_vs_base_w2=spearman(base[2.0], v2),
                   within_cv=float(v1.std() / v1.mean()),
                   mean_lever=float(v1.mean()),
                   per_prompt=v1.tolist())
        out["checkpoints"][name] = rec
        print("%-14s %10.3f %10.3f %11.3f %11.3f" %
              (name, rec["spearman_vs_base_w1"], rec["spearman_vs_base_w2"],
               rec["within_cv"], rec["mean_lever"]))

    means = np.array([out["checkpoints"][n]["mean_lever"] for n, _ in ORDER])
    withins = np.array([out["checkpoints"][n]["within_cv"] for n, _ in ORDER])
    out["between_checkpoint_cv_of_mean"] = float(means.std() / means.mean())
    out["within_checkpoint_cv_range"] = [float(withins.min()), float(withins.max())]
    print("\nbetween-checkpoint CV of the mean lever : %.3f" % out["between_checkpoint_cv_of_mean"])
    print("within-checkpoint per-prompt CV range   : %.3f - %.3f"
          % (withins.min(), withins.max()))
    print("=> between < within at every checkpoint: the 'intact lever' is a mean")
    print("   whose cross-checkpoint stability is smaller than its own spread.")
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
