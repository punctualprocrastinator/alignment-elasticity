# -*- coding: utf-8 -*-
"""Merge the per-box refit runs and decide: reorganisation, or stale probe?

Reads results/refit_profile_<label>.json (written independently by each box) and
compares three per-prompt lever profiles against base's:

  frozen   the committed Gate-A sweeps, steering with the FROZEN base direction
  refit    this run, steering with the direction REFIT at that checkpoint
  shuffled this run, refit on permuted harmful/benign labels -- the null floor

Reading:
  * frozen decorrelates AND refit decorrelates  -> the model's per-example
    response has genuinely been reorganised; the finding is probe-independent.
  * frozen decorrelates BUT refit recovers      -> the base direction went stale;
    the per-example structure is intact and the claim must be rewritten.
  * refit ~ shuffled                            -> no refusal signal survives at
    all at that checkpoint; neither story holds and the readout is the problem.

The shuffled arm matters because "recovery" is only meaningful above the level a
direction with no refusal content already achieves.
"""
import glob, json, os
import numpy as np

FROZEN_ORDER = [("base", "base"), ("SFT 1k", "think-sft-1000"),
                ("SFT 15k", "think-sft-15000"), ("SFT 43k", "think-sft-43000"),
                ("DPO", "think-dpo"), ("RLVR first", "think-rlvr-first"),
                ("RLVR last", "think-rlvr-last"), ("Instruct", "instruct"),
                ("RL-Zero Math", "rlz-math"), ("RL-Zero Code", "rlz-code")]
PRETTY = {k: n for n, k in FROZEN_ORDER}


def _rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den > 0 else float("nan")


def frozen_profile(label, window=1.0):
    """Per-prompt lever under the FROZEN base direction, from committed sweeps."""
    for p in ("results/gateA_traj/gateA_sweep_%s.json" % label,
              "results/gateA_sweep_%s.json" % label):
        if os.path.exists(p):
            d = json.load(open(p))
            g = np.array(d["gaps_massmean"], float)
            c = np.array(d["c_grid_massmean"], float)
            zi = int(d["zero_index_massmean"])
            m = (c < 0) & (np.abs(c) <= window)
            x = np.abs(c[m])
            return (x @ (g[zi][None, :] - g)[m]) / (x @ x)
    return None


def main():
    files = sorted(glob.glob("results/refit_profile_*.json"))
    if not files:
        print("no refit_profile_*.json yet")
        return
    recs = {}
    for f in files:
        d = json.load(open(f))
        recs[d["label"]] = d
    print("merged %d checkpoints: %s\n" % (len(recs), ", ".join(sorted(recs))))

    base_frozen = frozen_profile("base")
    base_refit = np.array(recs["base"]["per_prompt_lever_refit"]) if "base" in recs else None

    print("%-14s %8s %9s %9s %9s %9s" %
          ("ckpt", "cos(r,b)", "frozen_rho", "refit_rho", "shuf_rho", "verdict"))
    out = {}
    for name, key in FROZEN_ORDER:
        if key not in recs:
            continue
        r = recs[key]
        fz = frozen_profile(key)
        rf = np.array(r["per_prompt_lever_refit"], float)
        sh = np.array(r["per_prompt_lever_shuffled"], float)
        rho_f = spearman(base_frozen, fz) if fz is not None else float("nan")
        # refit profiles compare against base's own REFIT profile (base refit ~ base frozen)
        ref = base_refit if base_refit is not None else base_frozen
        rho_r = spearman(ref, rf)
        rho_s = spearman(ref, sh)
        if key == "base":
            verdict = "reference"
        elif rho_r - rho_s < 0.15:
            verdict = "no signal"
        elif rho_r > rho_f + 0.25:
            verdict = "STALE PROBE"
        else:
            verdict = "reorganised"
        out[name] = dict(cos_refit_base=r["cos_refit_base"], frozen_rho=rho_f,
                         refit_rho=rho_r, shuffled_rho=rho_s, verdict=verdict)
        print("%-14s %8.3f %9.3f %9.3f %9.3f %9s" %
              (name, r["cos_refit_base"], rho_f, rho_r, rho_s, verdict))

    print("\nverdict key:")
    print("  reorganised  refit profile decorrelates too -> probe-independent finding")
    print("  STALE PROBE  refit recovers >0.25 over frozen -> rewrite the claim")
    print("  no signal    refit indistinguishable from shuffled-label floor")
    json.dump(out, open("results/refit_analysis.json", "w"), indent=1)
    print("\nwrote results/refit_analysis.json")


if __name__ == "__main__":
    main()
