# -*- coding: utf-8 -*-
"""Track B, zero-GPU half. REWRITTEN after audit; read the retractions first.

WHAT THIS FILE USED TO CLAIM, AND WHY IT WAS WRONG
--------------------------------------------------
R1. "corr(margin, fixed-dose crossing) = -0.96, so margin growth predicts the
    audit artifact while the lever does not."  CIRCULAR. Crossing at a fixed
    dose is P(g0 - D < 0 | g0 > 0) and the margin is mean(g0): the same
    distribution, offset by a displacement D that is near-constant across
    checkpoints (CV 0.12). A surrogate that hands EVERY checkpoint the identical
    displacement -- deleting all checkpoint-specific steering information --
    reproduces the crossing rates with R^2 = 0.993 and yields corr = -0.97, i.e.
    a STRONGER association than the real data. The correlation measures the
    monotone map from a mean to a tail probability, not a finding.
    The second half was also backwards: corr(lever, crossing) is ~0 only
    marginally, because the lever's spread is 7x smaller than the margin's.
    Controlling for margin, partial corr(lever, crossing | margin) = +0.67
    (p ~ 0.05) with the correct sign -- the lever explains the residual the
    margin cannot mechanically force. See mechanical_null() below.

R2. "The random-direction floor rises 27x across the flow." WRONG ESTIMATOR.
    That ratio comes from efficacy_published (sign-folded, |c|<=2), which
    lever.py deprecates. Under the estimator actually used the base floor is
    NEGATIVE (-0.03), so the ratio is undefined. Signed floors are non-monotone
    and sign-changing along the flow (Spearman with flow position = +0.14
    excluding Instruct). Nine of ten checkpoints have a floor whose CI contains
    zero. In magnitude (RMS per-direction) terms the rise is ~1.6-1.7x.

R3. "floor/margin CV = 6.37 shows floor and margin are two phenomena." A CV on
    a sign-changing quantity is unbounded and meaningless (it moves 2.6 -> 7.4
    on single-point deletion). The conclusion survives, the warrant does not:
    use the correlation, which is indistinguishable from zero at this n.

R4. The curvature test used the wrong statistic. With
    disp_j(c) = -c(grad.v_j) - (c^2/2)(v_j' H v_j), the SYMMETRIC average
    [disp(-c)+disp(+c)]/2 isolates curvature (the gradient cancels exactly) and
    the antisymmetric half isolates the gradient. Requiring "both branches
    positive" instead tests |curvature| > |gradient|, which is confounded and
    low-powered. Done properly, ONE checkpoint (Instruct) shows significant
    negative curvature; the old criterion also silently discarded SFT 15k, which
    is significant in the OPPOSITE direction.

STATISTICAL NOTES
  * n = 10 checkpoints but n_eff ~ 4: the six Think-lineage points are near
    clones (DPO vs RLVR-first differ by 5.5e-03 in margin, 0.0 in crossing).
    Correlations are reported with a CLUSTER bootstrap over lineages.
  * Fisher-z, not the percentile bootstrap, for correlation CIs: at n=10 the
    percentile interval delivers 90-91% coverage against a nominal 95%, with
    the misses almost all on the far-from-zero side.
  * gate_a draws the 20 random directions with a FIXED seed, so every checkpoint
    is evaluated against the SAME ensemble. Floor comparisons are therefore
    PAIRED, which is both correct and much more powerful (Instruct vs base:
    paired t = 7.8 against unpaired t = 4.4).
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lever as L  # noqa: E402

R = "results"
ORDER = [("base", "base"), ("SFT 1k", "think-sft-1000"), ("SFT 15k", "think-sft-15000"),
         ("SFT 43k", "think-sft-43000"), ("DPO", "think-dpo"), ("RLVR first", "think-rlvr-first"),
         ("RLVR last", "think-rlvr-last"), ("Instruct", "instruct"),
         ("RL-Zero Math", "rlz-math"), ("RL-Zero Code", "rlz-code")]
# lineage clusters, for the cluster bootstrap (n_eff ~ 4, not 10)
CLUSTER = {"base": 0, "SFT 1k": 1, "SFT 15k": 1, "SFT 43k": 1, "DPO": 1,
           "RLVR first": 1, "RLVR last": 1, "Instruct": 2,
           "RL-Zero Math": 3, "RL-Zero Code": 3}
FIXED_C = -0.5
PROBE = 0.5          # matched dose present on BOTH branches of the null grid


def load(l):
    for p in ("%s/gateA_traj/gateA_sweep_%s.json" % (R, l), "%s/gateA_sweep_%s.json" % (R, l)):
        if os.path.exists(p):
            return json.load(open(p))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    d = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / d) if d > 0 else np.nan


def fisher_ci(r, n, alpha=0.05):
    """Fisher-z interval. Correct default at n=10; the percentile bootstrap is
    anticonservative here (90-91% actual coverage at nominal 95%)."""
    if not np.isfinite(r) or abs(r) >= 1 or n < 4:
        return (np.nan, np.nan)
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    lo, hi = z - 1.959964 * se, z + 1.959964 * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def cluster_boot_ci(x, y, clusters, n_boot=4000, seed=0):
    """Resample LINEAGES, not checkpoints. The six Think points are near-clones,
    so a plain bootstrap over 10 points reports roughly half the honest width."""
    x, y, cl = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    groups = [np.where(cl == g)[0] for g in np.unique(cl)]
    rng, out = np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if np.std(x[idx]) > 0 and np.std(y[idx]) > 0:
            out.append(pearson(x[idx], y[idx]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan,) * 2


def partial_corr(a, b, ctrl):
    """corr(a, b | ctrl) by residualising both on ctrl."""
    a, b, c = (np.asarray(v, float) for v in (a, b, ctrl))
    A = np.vstack([c, np.ones_like(c)]).T
    ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    return pearson(ra, rb)


def mechanical_null(rows, names):
    """R1: how much of the margin/crossing correlation is forced by construction?

    Give every checkpoint the SAME displacement (the grand mean) and predict
    crossing as the normal tail probability implied by its own gap mean and sd.
    No checkpoint-specific steering information survives this, so whatever
    correlation remains is mechanical.
    """
    from math import erf, sqrt
    Dbar = float(np.mean([rows[n]["disp_at_fixed"] for n in names]))
    pred = []
    for n in names:
        mu, sd = rows[n]["margin"], rows[n]["sd"]
        z = (Dbar - mu) / sd
        pred.append(0.5 * (1.0 + erf(z / sqrt(2.0))))
    obs = [rows[n]["crossing_fixed"] for n in names]
    marg = [rows[n]["margin"] for n in names]
    ss = 1.0 - np.sum((np.array(obs) - np.array(pred)) ** 2) / \
        np.sum((np.array(obs) - np.mean(obs)) ** 2)
    return dict(Dbar=Dbar, corr_margin_pred=pearson(marg, pred),
                corr_obs_pred=pearson(obs, pred), r2_pred_explains_obs=float(ss))


def main():
    rows = {}
    for name, key in ORDER:
        d = load(key)
        if d is None:
            continue
        g = np.array(d["gaps_massmean"], float)
        c = np.array(d["c_grid_massmean"], float)
        zi = int(d["zero_index_massmean"])
        gr = np.array(d["gaps_random"], float)
        cr = np.array(d["c_grid_random"], float)
        g0, disp = g[zi], L.displacement(g, zi)

        per_dir = np.array([L.efficacy(cr, (g0[None, :] - gr[j]).mean(axis=1))
                            for j in range(gr.shape[0])])
        ci_ = int(np.argmin(np.abs(c - FIXED_C)))

        # R4: symmetric/antisymmetric split at a dose present on BOTH branches
        ineg = int(np.argmin(np.abs(cr + PROBE)))
        ipos = int(np.argmin(np.abs(cr - PROBE)))
        dneg = np.array([(g0 - gr[j, ineg]).mean() for j in range(gr.shape[0])])
        dpos = np.array([(g0 - gr[j, ipos]).mean() for j in range(gr.shape[0])])
        curv, grad = (dneg + dpos) / 2.0, (dneg - dpos) / 2.0

        rows[name] = dict(
            margin=L.margin(g, zi), sd=float(g0.std()), dprime=L.dprime(g, zi),
            lever=L.efficacy(c, disp),
            floor=float(per_dir.mean()),
            floor_sem=float(per_dir.std(ddof=1) / np.sqrt(len(per_dir))),
            floor_rms=float(np.sqrt((per_dir ** 2).mean())),
            floor_per_dir=per_dir.tolist(),
            curv=float(curv.mean()),
            curv_t=float(curv.mean() / (curv.std(ddof=1) / np.sqrt(len(curv)))),
            grad=float(grad.mean()),
            crossing_fixed=L.crossing_rate(g, ci_, zi),
            disp_at_fixed=float(disp[ci_]),
        )

    names = list(rows)
    cl = [CLUSTER[n] for n in names]
    marg = [rows[n]["margin"] for n in names]
    lev = [rows[n]["lever"] for n in names]
    cross = [rows[n]["crossing_fixed"] for n in names]

    print("=" * 90)
    print("R1  Is margin -> fixed-dose crossing a finding, or arithmetic?")
    print("=" * 90)
    mn = mechanical_null(rows, names)
    r_mc = pearson(marg, cross)
    print("  observed   corr(margin, crossing)          = %+.4f" % r_mc)
    print("  MECHANICAL corr(margin, surrogate crossing) = %+.4f   <- no steering info at all"
          % mn["corr_margin_pred"])
    print("  surrogate explains R^2 = %.4f of the real crossing rates" % mn["r2_pred_explains_obs"])
    print("  => the observed value is NOT stronger than the constructed one. Circular.")
    print("  cluster-bootstrap CI (lineages, n_eff~4): [%+.3f, %+.3f]   Fisher: [%+.3f, %+.3f]"
          % (*cluster_boot_ci(marg, cross, cl), *fisher_ci(r_mc, len(names))))
    print("\n  the lever half, done properly:")
    print("    marginal corr(lever, crossing)          = %+.4f  (lever spread is 7x smaller)"
          % pearson(lev, cross))
    print("    PARTIAL  corr(lever, crossing | margin) = %+.4f  <- correct sign, non-trivial"
          % partial_corr(lev, cross, marg))

    print("\n" + "=" * 90)
    print("R2/R3  The random-direction floor")
    print("=" * 90)
    print("%-14s %8s %8s %9s %9s" % ("checkpoint", "floor", "SEM", "t", "|floor| RMS"))
    base_pd = np.array(rows["base"]["floor_per_dir"])
    for n in names:
        r = rows[n]
        t = r["floor"] / r["floor_sem"]
        print("%-14s %8.3f %8.3f %9.2f %9.3f" % (n, r["floor"], r["floor_sem"], t, r["floor_rms"]))
    sig = [n for n in names if abs(rows[n]["floor"] / rows[n]["floor_sem"]) > 2.093]
    print("  floors distinguishable from zero (|t|>2.09): %s" % (sig or "none"))
    dif = np.array(rows["Instruct"]["floor_per_dir"]) - base_pd      # PAIRED: same 20 directions
    print("  Instruct vs base, PAIRED over the shared 20 directions: mean %+.3f, t = %.2f"
          % (dif.mean(), dif.mean() / (dif.std(ddof=1) / np.sqrt(len(dif)))))
    print("  magnitude (RMS) rise base -> Instruct: %.2fx  (NOT the 27x previously claimed,"
          % (rows["Instruct"]["floor_rms"] / rows["base"]["floor_rms"]))
    print("   which used the deprecated sign-folded estimator over a near-zero denominator)")
    r_fm = pearson([rows[n]["floor"] for n in names], marg)
    print("  corr(floor, margin) = %+.3f  Fisher [%+.3f, %+.3f]  -> not distinguishable from 0"
          % (r_fm, *fisher_ci(r_fm, len(names))))

    print("\n" + "=" * 90)
    print("R4  Curvature, isolated properly ([d(-c)+d(+c)]/2 cancels the gradient)")
    print("=" * 90)
    print("%-14s %9s %9s %9s" % ("checkpoint", "curvature", "t", "gradient"))
    for n in names:
        r = rows[n]
        star = " *" if abs(r["curv_t"]) > 2.093 else ""
        print("%-14s %9.3f %9.2f %9.3f%s" % (n, r["curv"], r["curv_t"], r["grad"], star))
    pos = [n for n in names if rows[n]["curv_t"] > 2.093]
    neg = [n for n in names if rows[n]["curv_t"] < -2.093]
    print("  significant NEGATIVE curvature (local max of refusal-ness): %s" % (pos or "none"))
    print("  significant POSITIVE curvature (local min):                 %s" % (neg or "none"))

    json.dump({"experiment": "mechanism analysis (rewritten post-audit)",
               "fixed_c": FIXED_C, "probe_dose": PROBE, "clusters": CLUSTER,
               "checkpoints": rows, "mechanical_null": mn,
               "corr": {"margin_crossing": r_mc,
                        "margin_crossing_cluster_ci": cluster_boot_ci(marg, cross, cl),
                        "lever_crossing_marginal": pearson(lev, cross),
                        "lever_crossing_partial_given_margin": partial_corr(lev, cross, marg),
                        "floor_margin": r_fm}},
              open("%s/mechanism_analysis.json" % R, "w"), indent=1)
    print("\nwrote %s/mechanism_analysis.json" % R)


if __name__ == "__main__":
    main()
