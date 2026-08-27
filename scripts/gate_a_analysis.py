"""Gate A analysis: the three comparisons that decide the paper.

1. crossing rate vs c                    - input-norm dosing, reproduces the collapse
2. crossing rate vs achieved displacement - the boundary-relative view
3. c_50 / d_50 per model with bootstrap CIs and paired difference intervals

Everything here is pure numpy over the per-prompt gap matrices already on disk,
so re-analysis never touches a GPU.
"""

import json
import os

import numpy as np

import gate_a as ga


def load_all(art=ga.ART, labels=None):
    """Sweep records keyed by label, in checkpoint order."""
    labels = labels or [lab for lab, _r, _b, _c in ga.CKPTS]
    pth = ga.paths(art)
    out = {}
    for lab in labels:
        p = pth["sweep"](lab)
        if os.path.exists(p):
            out[lab] = json.load(open(p))
    return out


def load_gen(art=ga.ART, labels=None):
    labels = labels or [lab for lab, _r, _b, _c in ga.CKPTS]
    pth = ga.paths(art)
    out = {}
    for lab in labels:
        p = pth["gen"](lab)
        if os.path.exists(p):
            out[lab] = json.load(open(p))
    return out


def arrays(rec, which="massmean"):
    """(gaps [n_c, n_prompts], c_values, zero_index) for one direction family."""
    gaps = np.asarray(rec["gaps_" + which], dtype=np.float64)
    c = list(rec["c_grid_" + which])
    return gaps, c, int(rec["zero_index_" + which])


def fingerprint_check(recs):
    """Aggregation refuses any result file whose split fingerprint disagrees."""
    fps = {lab: r["provenance"]["split_fingerprint"] for lab, r in recs.items()}
    seeds = {lab: r["provenance"]["seed"] for lab, r in recs.items()}
    return {
        "fingerprints": fps,
        "seeds": seeds,
        "all_agree": len(set(fps.values())) <= 1 and len(set(seeds.values())) <= 1,
    }


# ------------------------------------------------------------------ curves


def curve(rec, which="massmean"):
    """Crossing rate against three x axes.

    `disps` is the achieved displacement of the refusal logit gap. `disps_rel`
    subtracts the model's own MEDIAN unsteered gap, which is the distance the
    median prompt has to travel to reach its boundary: a uniform, perfectly
    efficient push crosses the median prompt at disps_rel == 0 regardless of how
    deep the model sits. That is the axis on which "no residual effect" predicts
    the curves to coincide; raw `disps` is mechanically tied to baseline depth.
    """
    gaps, c, zi = arrays(rec, which)
    rates, disps, disps_med, n_elig = ga.curve_from_gaps(gaps, c, zi)
    med0 = float(np.median(gaps[zi]))
    # disps_rel MUST be built from the median displacement: it is compared
    # against a median baseline gap, and mixing a mean displacement with a
    # median gap makes d50_excess non-zero even under no residual effect, by a
    # model-dependent amount. `disps` (mean) is kept for continuity only.
    return {"c": c, "rates": rates, "disps": disps, "disps_med": disps_med,
            "disps_rel": disps_med - med0, "median_gap0": med0,
            "n_eligible": n_elig, "unsteered_gap": float(gaps[zi].mean())}


def null_curves(rec):
    """One (disps, rates) curve per random direction, referenced to the same
    unsteered gaps. Matched norm, matched coefficient grid."""
    g0 = np.asarray(rec["unsteered_gap_per_prompt"], dtype=np.float64)
    c_null = list(rec["c_grid_random"])
    out = []
    for gk in rec["gaps_random"]:
        gk = np.asarray(gk, dtype=np.float64)
        rates, disps = [], []
        for i in range(gk.shape[0]):
            r, _n = ga.crossing_rate(gk[i], g0)
            rates.append(r)
            disps.append(float((g0 - gk[i]).mean()))
        out.append({"c": c_null, "rates": np.asarray(rates),
                    "disps": np.asarray(disps)})
    return out


def null_band(rec, q_lo=2.5, q_hi=97.5):
    """Percentile band of the random-direction crossing rate at each matched c."""
    nc = null_curves(rec)
    c = np.asarray(nc[0]["c"], dtype=np.float64)
    R = np.vstack([x["rates"] for x in nc])
    D = np.vstack([x["disps"] for x in nc])
    return {
        "c": c.tolist(),
        "rate_lo": np.nanpercentile(R, q_lo, axis=0).tolist(),
        "rate_hi": np.nanpercentile(R, q_hi, axis=0).tolist(),
        "rate_mean": np.nanmean(R, axis=0).tolist(),
        "rate_sd": np.nanstd(R, axis=0, ddof=1).tolist(),
        "disp_lo": np.nanpercentile(D, q_lo, axis=0).tolist(),
        "disp_hi": np.nanpercentile(D, q_hi, axis=0).tolist(),
        "disp_mean": np.nanmean(D, axis=0).tolist(),
        "disp_sd": np.nanstd(D, axis=0, ddof=1).tolist(),
        "n_directions": int(R.shape[0]),
    }


def z_vs_null(rec, which="massmean"):
    """Effect of the base direction as a z-score against the random-direction
    null at MATCHED coefficient (protocol P3)."""
    cur = curve(rec, which)
    nb = null_band(rec)
    c_null = np.asarray(nb["c"], dtype=np.float64)
    rows = []
    for i, cv in enumerate(cur["c"]):
        j = np.where(np.isclose(c_null, cv))[0]
        if j.size == 0:
            continue
        j = int(j[0])
        sd_r = nb["rate_sd"][j]
        sd_d = nb["disp_sd"][j]
        rows.append({
            "c": float(cv),
            "rate": float(cur["rates"][i]),
            "null_rate_mean": float(nb["rate_mean"][j]),
            "rate_z": (float((cur["rates"][i] - nb["rate_mean"][j]) / sd_r)
                       if sd_r and sd_r > 0 else None),
            "disp": float(cur["disps"][i]),
            "null_disp_mean": float(nb["disp_mean"][j]),
            "disp_z": (float((cur["disps"][i] - nb["disp_mean"][j]) / sd_d)
                       if sd_d and sd_d > 0 else None),
        })
    return rows


# ------------------------------------------------------------------ bootstrap


def paired_bootstrap(recs, which="massmean", n_boot=1000, seed=ga.SEED,
                     target=0.5):
    """PAIRED bootstrap of c_50 and d_50 across models.

    Every model is scored on the identical held-out prompt set in the identical
    order, so one resampled index vector is applied to all of them. That pairing
    is what makes the d_50 *difference* intervals powerful enough to be worth
    reporting (protocol P4).
    """
    labels = list(recs.keys())
    packs = {lab: arrays(recs[lab], which) for lab in labels}
    ns = {lab: packs[lab][0].shape[1] for lab in labels}
    if len(set(ns.values())) != 1:
        raise ValueError("paired bootstrap needs identical prompt counts: %r" % ns)
    n = ns[labels[0]]

    rng = np.random.default_rng(seed)
    c50 = {lab: [] for lab in labels}
    d50 = {lab: [] for lab in labels}
    d50x = {lab: [] for lab in labels}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        for lab in labels:
            gaps, c, zi = packs[lab]
            a, b = ga.fifty_points(gaps[:, idx], c, zi, target)
            c50[lab].append(a)
            d50[lab].append(b)
            d50x[lab].append(b - float(np.median(gaps[zi][idx])))

    point = {}
    for lab in labels:
        gaps, c, zi = packs[lab]
        a, b = ga.fifty_points(gaps, c, zi, target)
        med0 = float(np.median(gaps[zi]))
        point[lab] = {"c50": a, "d50": b, "d50_excess": b - med0,
                      "median_gap0": med0}

    table = {}
    for lab in labels:
        table[lab] = {
            "c50": point[lab]["c50"],
            "c50_ci": ga.pct_ci(c50[lab]),
            "d50": point[lab]["d50"],
            "d50_ci": ga.pct_ci(d50[lab]),
            "median_gap0": point[lab]["median_gap0"],
            "d50_excess": point[lab]["d50_excess"],
            "d50_excess_ci": ga.pct_ci(d50x[lab]),
            "unsteered_gap": float(recs[lab]["unsteered_gap_mean"]),
            "frac_refusing_unsteered": float(recs[lab]["frac_refusing_unsteered"]),
            "n_eligible": int(recs[lab]["n_eligible"]),
            "mu": float(recs[lab]["mu_used"]),
        }

    diffs = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            da = np.asarray(d50[a], dtype=np.float64)
            db = np.asarray(d50[b], dtype=np.float64)
            ok = np.isfinite(da) & np.isfinite(db)
            ci = ga.pct_ci(da[ok] - db[ok])
            pt = point[a]["d50"] - point[b]["d50"]
            ci["point"] = float(pt) if np.isfinite(pt) else None
            ci["excludes_zero"] = bool(
                ci["lo"] is not None and (ci["lo"] > 0 or ci["hi"] < 0))
            diffs["d50:%s-%s" % (a, b)] = ci

            xa = np.asarray(d50x[a], dtype=np.float64)
            xb = np.asarray(d50x[b], dtype=np.float64)
            okx = np.isfinite(xa) & np.isfinite(xb)
            cix = ga.pct_ci(xa[okx] - xb[okx])
            ptx = point[a]["d50_excess"] - point[b]["d50_excess"]
            cix["point"] = float(ptx) if np.isfinite(ptx) else None
            cix["excludes_zero"] = bool(
                cix["lo"] is not None and (cix["lo"] > 0 or cix["hi"] < 0))
            diffs["d50excess:%s-%s" % (a, b)] = cix

            ca = np.asarray(c50[a], dtype=np.float64)
            cb = np.asarray(c50[b], dtype=np.float64)
            ok2 = np.isfinite(ca) & np.isfinite(cb)
            ci2 = ga.pct_ci(ca[ok2] - cb[ok2])
            pt2 = point[a]["c50"] - point[b]["c50"]
            ci2["point"] = float(pt2) if np.isfinite(pt2) else None
            ci2["excludes_zero"] = bool(
                ci2["lo"] is not None and (ci2["lo"] > 0 or ci2["hi"] < 0))
            diffs["c50:%s-%s" % (a, b)] = ci2

    return {"which": which, "n_boot": int(n_boot), "seed": int(seed),
            "target": target, "table": table, "diffs": diffs}


# ------------------------------------------------------------------ collapse


def _interp_rate(x_vals, rates, grid):
    """Crossing rate on a common x axis. Duplicate x values are averaged, and
    the curve is evaluated only inside each model's own swept range."""
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(rates, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return np.full(np.shape(grid), np.nan, dtype=np.float64)
    # Displacement axes saturate and reverse, so the same x can occur at a low
    # dose and again past the peak. Averaging those two produces a rate no dose
    # actually produced. Keep the rising prefix, where x is a function of dose.
    peak = int(np.nanargmax(x))
    x, y = x[:peak + 1], y[:peak + 1]
    if x.size < 2:
        return np.full(np.shape(grid), np.nan, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux, inv = np.unique(x, return_inverse=True)
    uy = np.zeros_like(ux)
    for i in range(ux.size):
        uy[i] = y[inv == i].mean()
    if ux.size < 2:
        return np.full(np.shape(grid), np.nan, dtype=np.float64)
    out = np.interp(grid, ux, uy, left=np.nan, right=np.nan)
    out[(grid < ux.min()) | (grid > ux.max())] = np.nan
    return out


def collapse_spread(recs, which="massmean", n_grid=25):
    """Spread of the crossing-rate curves across models under the two dosings.

    This is analysis 1 vs analysis 2 reduced to a single number each: if
    displacement-matching collapses the curves, `mean_spread` falls sharply from
    the c-matched value to the displacement-matched value.
    """
    labels = list(recs.keys())
    curves = {lab: curve(recs[lab], which) for lab in labels}

    c_common = sorted(set.intersection(*[set(curves[l]["c"]) for l in labels]))
    c_grid = np.asarray(c_common, dtype=np.float64)
    Rc = np.vstack([_interp_rate(curves[l]["c"], curves[l]["rates"], c_grid)
                    for l in labels])

    d_lo = max(float(np.nanmin(curves[l]["disps"])) for l in labels)
    d_hi = min(float(np.nanmax(curves[l]["disps"])) for l in labels)
    # A shrinking comparison region shrinks the spread for reasons unrelated to
    # collapse. Base's maximum achievable displacement can sit BELOW a deeper
    # model's d_50 (Gate A finding 2), leaving a sliver or nothing at all.
    d_widths = [float(np.nanmax(curves[l]["disps"]) - np.nanmin(curves[l]["disps"]))
                for l in labels]
    d_overlap = {"lo": d_lo, "hi": d_hi, "width": d_hi - d_lo,
                 "frac_of_narrowest": ((d_hi - d_lo) / min(d_widths)
                                       if min(d_widths) > 0 else 0.0),
                 "degenerate": bool(d_hi - d_lo <= 0)}
    d_grid = np.linspace(d_lo, d_hi, n_grid)
    Rd = np.vstack([_interp_rate(curves[l]["disps"], curves[l]["rates"], d_grid)
                    for l in labels])

    r_lo = max(float(np.nanmin(curves[l]["disps_rel"])) for l in labels)
    r_hi = min(float(np.nanmax(curves[l]["disps_rel"])) for l in labels)
    r_widths = [float(np.nanmax(curves[l]["disps_rel"]) - np.nanmin(curves[l]["disps_rel"]))
                for l in labels]
    r_overlap = {"lo": r_lo, "hi": r_hi, "width": r_hi - r_lo,
                 "frac_of_narrowest": ((r_hi - r_lo) / min(r_widths)
                                       if min(r_widths) > 0 else 0.0),
                 "degenerate": bool(r_hi - r_lo <= 0)}
    r_grid = np.linspace(r_lo, r_hi, n_grid)
    Rr = np.vstack([_interp_rate(curves[l]["disps_rel"], curves[l]["rates"], r_grid)
                    for l in labels])

    def _spread(M):
        # A grid column with fewer than two models present is not a spread.
        M = np.asarray(M, dtype=np.float64)
        n_present = np.sum(np.isfinite(M), axis=0)
        s = np.nanmax(M, axis=0) - np.nanmin(M, axis=0)
        s[n_present < 2] = np.nan
        return s

    def _coverage(M):
        M = np.asarray(M, dtype=np.float64)
        return float(np.mean(np.sum(np.isfinite(M), axis=0) >= 2))

    sc, sd, sr = _spread(Rc), _spread(Rd), _spread(Rr)
    cov_c, cov_d, cov_r = _coverage(Rc), _coverage(Rd), _coverage(Rr)
    return {
        "which": which,
        "labels": labels,
        "rel_grid": r_grid.tolist(),
        "rates_by_disp_rel": Rr.tolist(),
        "mean_spread_by_disp_rel": float(np.nanmean(sr)),
        "max_spread_by_disp_rel": float(np.nanmax(sr)),
        "spread_reduction_rel": (float(1.0 - np.nanmean(sr) / np.nanmean(sc))
                                 if np.nanmean(sc) > 0 else None),
        "median_gap0": {l: curves[l]["median_gap0"] for l in labels},
        "c_grid": c_grid.tolist(),
        "rates_by_c": Rc.tolist(),
        "d_grid": d_grid.tolist(),
        "rates_by_disp": Rd.tolist(),
        "mean_spread_by_c": float(np.nanmean(sc)),
        "max_spread_by_c": float(np.nanmax(sc)),
        "mean_spread_by_disp": float(np.nanmean(sd)),
        "max_spread_by_disp": float(np.nanmax(sd)),
        "spread_reduction": (float(1.0 - np.nanmean(sd) / np.nanmean(sc))
                             if np.nanmean(sc) > 0 else None),
        "displacement_overlap": [float(d_lo), float(d_hi)],
        # Diagnostics: a spread computed over a sliver of overlap, or over grid
        # columns where only one model is present, is not evidence of collapse.
        "overlap_disp": d_overlap,
        "overlap_disp_rel": r_overlap,
        "coverage_by_c": cov_c,
        "coverage_by_disp": cov_d,
        "coverage_by_disp_rel": cov_r,
        "spread_trustworthy": bool(
            (not d_overlap["degenerate"]) and (not r_overlap["degenerate"])
            and d_overlap["frac_of_narrowest"] >= 0.25
            and r_overlap["frac_of_narrowest"] >= 0.25
            and cov_d >= 0.5 and cov_r >= 0.5),
    }


# ------------------------------------------------------------------ verdict


def verdict(summary, spread_threshold=0.5):
    """Which of the two pre-registered outcomes obtained.

    The decisive statistic is d_50_EXCESS = d_50 - median unsteered gap, not raw
    d_50. Raw d_50 is mechanically pinned to how deep a model sits (to cross the
    median prompt you must displace it by roughly its own baseline gap), so
    "equal raw d_50" is not the no-residual-effect null - it is close to
    impossible. Excess displacement is what a model costs you OVER AND ABOVE its
    baseline distance, and that is the boundary-relative effect size.

    Outcome A - the collapse is fully explained by baseline distance: every
    pairwise d_50_excess interval covers zero AND matching on boundary-relative
    displacement substantially shrinks the between-model spread.
    Outcome B - a residual effect survives: some pair's d_50_excess interval
    excludes zero, i.e. a model still crosses less at equal distance past its
    own boundary.
    """
    diffs = summary["bootstrap_massmean"]["diffs"]
    ex_diffs = {k: v for k, v in diffs.items() if k.startswith("d50excess:")}
    raw_diffs = {k: v for k, v in diffs.items() if k.startswith("d50:")}
    sig = {k: v for k, v in ex_diffs.items() if v.get("excludes_zero")}
    sig_raw = {k: v for k, v in raw_diffs.items() if v.get("excludes_zero")}

    coll = summary["collapse_massmean"]
    red = coll.get("spread_reduction_rel")
    red_raw = coll.get("spread_reduction")
    collapsed = (red is not None) and (red >= spread_threshold)

    if not sig and collapsed:
        outcome = "A"
        stmt = ("OUTCOME A. The apparent steering collapse is an artifact of "
                "baseline distance to the boundary. Once dose is expressed as "
                "displacement PAST each model's own boundary, the crossing-rate "
                "curves coincide (between-model spread falls by %.0f%%) and every "
                "pairwise d_50-excess interval covers zero. Clean negative "
                "methods result: a stale base direction is not weaker on the "
                "post-trained models, it simply has further to push."
                % (100 * red))
    elif sig:
        outcome = "B"
        stmt = ("OUTCOME B. A residual effect survives boundary-matched dosing: "
                + "; ".join(
                    "%s excess-d_50 diff %.3f [%.3f, %.3f]"
                    % (k[10:], v["point"], v["lo"], v["hi"])
                    for k, v in sig.items())
                + ". These models cross less even at equal displacement past "
                  "their own boundary, so post-training changed more than the "
                  "baseline distance.")
    else:
        outcome = "A-partial"
        stmt = ("MIXED. No pairwise d_50-excess interval excludes zero, but "
                "boundary-relative matching reduced the between-model spread by "
                "only %s, so the negative result is underpowered rather than "
                "established."
                % ("%.0f%%" % (100 * red) if red is not None else "an undefined amount"))
    return {
        "outcome": outcome,
        "statement": stmt,
        "decisive_statistic": "d50_excess = d50 - median(unsteered gap)",
        "significant_d50_excess_diffs": sig,
        "significant_raw_d50_diffs": sig_raw,
        "spread_reduction_boundary_relative": red,
        "spread_reduction_raw_displacement": red_raw,
        "mean_spread_by_c": coll["mean_spread_by_c"],
        "mean_spread_by_disp": coll["mean_spread_by_disp"],
        "mean_spread_by_disp_rel": coll["mean_spread_by_disp_rel"],
    }


def summarise(art=ga.ART, n_boot=1000, seed=ga.SEED):
    """Everything the paper needs, in one JSON-serialisable dict."""
    recs = load_all(art)
    gens = load_gen(art)
    if not recs:
        return {"error": "no sweep artifacts found in " + art}

    dirs_meta = {}
    pth = ga.paths(art)
    if os.path.exists(pth["dirs"]):
        dirs_meta = json.load(open(pth["dirs"]))
    smoke = json.load(open(pth["smoke"])) if os.path.exists(pth["smoke"]) else {}

    models = {}
    for lab, r in recs.items():
        models[lab] = {
            "repo": r["repo"], "branch": r["branch"], "commit": r["commit"],
            "mu_stats": r["mu_stats"], "mu_used": r["mu_used"],
            "unsteered_gap_mean": r["unsteered_gap_mean"],
            "frac_refusing_unsteered": r["frac_refusing_unsteered"],
            "n_eligible": r["n_eligible"],
            "c50_massmean": r["c50_massmean"], "d50_massmean": r["d50_massmean"],
            "c50_logistic": r["c50_logistic"], "d50_logistic": r["d50_logistic"],
            "curve_massmean": r["curve_massmean"],
            "curve_logistic": r["curve_logistic"],
            "zero_coeff_max_abs_diff": r["zero_coeff_max_abs_diff"],
            "behavioural": r.get("behavioural"),
            "load_seconds": r.get("load_seconds"),
        }

    summary = {
        "experiment": "Gate A - boundary-matched dosing of a stale base refusal direction",
        "provenance": list(recs.values())[0]["provenance"],
        "fingerprint_check": fingerprint_check(recs),
        "directions": dirs_meta,
        "smoke": smoke,
        "steer_param": list(recs.values())[0]["steer_param"],
        "models": models,
        "bootstrap_massmean": paired_bootstrap(recs, "massmean", n_boot, seed),
        "bootstrap_logistic": paired_bootstrap(recs, "logistic", n_boot, seed),
        "collapse_massmean": collapse_spread(recs, "massmean"),
        "collapse_logistic": collapse_spread(recs, "logistic"),
        "null_band": {lab: null_band(r) for lab, r in recs.items()},
        "z_vs_null_massmean": {lab: z_vs_null(r, "massmean") for lab, r in recs.items()},
        "behavioural": {
            lab: {k: g[k] for k in (
                "coeff", "n", "max_new_tokens", "unsteered_refusal_rate",
                "steered_refusal_rate", "unsteered_think", "steered_think")}
            for lab, g in gens.items()
        },
    }
    summary["verdict"] = verdict(summary)
    return summary


# ------------------------------------------------------------------ figure


MODEL_COLORS = {
    "base": "#1f77b4",
    "instruct": "#d62728",
    "rlz-math": "#2ca02c",
    "rlz-code": "#9467bd",
}
MODEL_LABELS = {
    "base": "Olmo-3-7B (base)",
    "instruct": "Olmo-3-7B-Instruct",
    "rlz-math": "RL-Zero-Math (step 1900)",
    "rlz-code": "RL-Zero-Code (step 2900)",
}


def make_figure(summary, path, which="massmean", c_min=-6.5):
    """Two panels: the misleading input-norm view and the boundary-relative one.

    Left is what 2605.13329's dosing shows; right is the same data re-expressed
    against the displacement actually achieved in the refusal logit gap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [l for l in ["base", "instruct", "rlz-math", "rlz-code"]
              if l in summary["models"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=200,
                             facecolor="white")

    # ---- random-direction null envelope, pooled over models
    nb = summary["null_band"]

    # panel 1: crossing rate vs c
    ax = axes[0]
    _c_all, _lo, _hi = None, None, None
    for lab in labels:
        b = nb[lab]
        c = np.asarray(b["c"], dtype=np.float64)
        lo = np.asarray(b["rate_lo"], dtype=np.float64)
        hi = np.asarray(b["rate_hi"], dtype=np.float64)
        if _c_all is None:
            _c_all, _lo, _hi = c, lo.copy(), hi.copy()
        else:
            _lo = np.minimum(_lo, lo)
            _hi = np.maximum(_hi, hi)
    if _c_all is not None:
        m = _c_all >= c_min
        ax.fill_between(_c_all[m], _lo[m], _hi[m], color="0.75", alpha=0.55,
                        lw=0, zorder=1,
                        label="random directions, matched norm\n(20 per model, 95% band)")

    for lab in labels:
        cur = summary["models"][lab]["curve_" + which]
        c = np.asarray(cur["c"], dtype=np.float64)
        r = np.asarray(cur["rates"], dtype=np.float64)
        m = c >= c_min
        ax.plot(c[m], r[m], "-o", ms=4.2, lw=2.0, color=MODEL_COLORS[lab],
                label=MODEL_LABELS[lab], zorder=3)
    ax.axhline(0.5, color="0.35", ls="--", lw=1.1, zorder=2)
    ax.set_xlabel("steering coefficient c   (input-norm dosing)", fontsize=12)
    ax.set_ylabel("crossing rate\n(fraction driven refusal -> compliance)",
                  fontsize=12)
    ax.set_title("1.  Dose = c * mu_L20   --   the misleading view", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)

    # panel 2: crossing rate vs achieved displacement
    ax = axes[1]
    dlo, dhi, rlo, rhi = None, None, None, None
    for lab in labels:
        b = nb[lab]
        d = np.asarray(b["disp_mean"], dtype=np.float64)
        r_l = np.asarray(b["rate_lo"], dtype=np.float64)
        r_h = np.asarray(b["rate_hi"], dtype=np.float64)
        o = np.argsort(d)
        ax.fill_between(d[o], r_l[o], r_h[o], color="0.75", alpha=0.45, lw=0,
                        zorder=1,
                        label=("random directions, matched norm\n(20 per model, 95% band)"
                               if lab == labels[0] else None))

    for lab in labels:
        cur = summary["models"][lab]["curve_" + which]
        d = np.asarray(cur["disps"], dtype=np.float64)
        r = np.asarray(cur["rates"], dtype=np.float64)
        o = np.argsort(d)
        ax.plot(d[o], r[o], "-o", ms=4.2, lw=2.0, color=MODEL_COLORS[lab],
                label=MODEL_LABELS[lab], zorder=3)
        d50 = summary["models"][lab]["d50_" + which]
        if d50 is not None and np.isfinite(d50):
            ax.plot([d50], [0.5], marker="*", ms=15,
                    color=MODEL_COLORS[lab], mec="white", mew=0.8, zorder=4)
    ax.axhline(0.5, color="0.35", ls="--", lw=1.1, zorder=2)
    ax.set_xlabel("achieved displacement of refusal logit gap\n"
                  "(unsteered minus steered; positive = toward compliance)",
                  fontsize=12)
    ax.set_ylabel("crossing rate", fontsize=12)
    ax.set_title("2.  Dose = achieved displacement   --   boundary-relative",
                 fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    v = summary["verdict"]
    fig.suptitle(
        "Gate A: base-model layer-20 refusal direction, carried unchanged to every checkpoint"
        "   |   stars = d_50   |   outcome %s" % v["outcome"],
        fontsize=13.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def make_figure_rel(summary, path, which="massmean"):
    """Supplementary: the axis that actually tests the hypothesis.

    x is displacement PAST each model's own median unsteered gap, so x=0 is
    every model's own boundary. Under "no residual effect" the curves coincide.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [l for l in ["base", "instruct", "rlz-math", "rlz-code"]
              if l in summary["models"]]
    coll = summary["collapse_" + which]
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200, facecolor="white")

    for lab in labels:
        cur = summary["models"][lab]["curve_" + which]
        med0 = coll["median_gap0"][lab]
        d = np.asarray(cur["disps"], dtype=np.float64) - med0
        r = np.asarray(cur["rates"], dtype=np.float64)
        o = np.argsort(d)
        ax.plot(d[o], r[o], "-o", ms=4.2, lw=2.0, color=MODEL_COLORS[lab],
                label="%s  (median gap %.2f)" % (MODEL_LABELS[lab], med0))
    ax.axhline(0.5, color="0.35", ls="--", lw=1.1)
    ax.axvline(0.0, color="0.35", ls=":", lw=1.1)
    ax.set_xlabel("displacement PAST the model's own boundary\n"
                  "(achieved displacement minus median unsteered gap)",
                  fontsize=12)
    ax.set_ylabel("crossing rate", fontsize=12)
    ax.set_title("Gate A supplementary: boundary-relative dosing\n"
                 "spread vs c %.3f  ->  vs displacement %.3f  ->  vs "
                 "boundary-relative %.3f"
                 % (coll["mean_spread_by_c"], coll["mean_spread_by_disp"],
                    coll["mean_spread_by_disp_rel"]), fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path
