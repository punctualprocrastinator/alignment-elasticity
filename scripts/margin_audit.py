# -*- coding: utf-8 -*-
"""Track A: the margin-normalized steering audit protocol.

Builds and validates the corrected instrument entirely from existing Gate-A
sweeps (zero GPU). Sweep schema per checkpoint:
    gaps_massmean          (n_doses, n_prompts)  logit gap at each dose
    c_grid_massmean        (n_doses,)            dose grid, ascending
    zero_index_massmean    int                   index of c = 0
    mu_used                float                 dose scale (steering is c*mu*v_hat)

Definitions (paper):
    margin_m    = mean unsteered gap                       "load"
    efficacy_m  = slope of mean displacement vs |c| near 0  "lever"
    D_m(c)      = mean per-prompt displacement at dose c
    crossing    = fraction of prompts with gap < 0

Protocols compared:
    mu_fixed(c)      dose every model at the same c. Because steering is
                     h += c*mu*v_hat this IS the mu-normalized (persona-vector)
                     protocol; and since CV(mu)=0.035 it is numerically close to
                     a fixed raw magnitude too. This is the baseline that fails.
    margin_lin(a)    c = a * margin / efficacy          (linear extrapolation)
    margin_inv(a)    c = D^{-1}(a * margin)             (saturation-honest)

Circularity control (A3): margin/efficacy are fit on prompt split S1 and the
crossing rate is evaluated on the disjoint split S2. If the collapse only
appears in-split, the protocol is circular and the result is void.

Usage:
    python scripts/margin_audit.py                # full report -> results/margin_audit.json
"""
import json, os, itertools
import numpy as np

R = "results"
ORDER = [
    ("base", "base"), ("SFT 1k", "think-sft-1000"), ("SFT 15k", "think-sft-15000"),
    ("SFT 43k", "think-sft-43000"), ("DPO", "think-dpo"),
    ("RLVR first", "think-rlvr-first"), ("RLVR last", "think-rlvr-last"),
    ("Instruct", "instruct"), ("RL-Zero Math", "rlz-math"), ("RL-Zero Code", "rlz-code"),
]
NEAR_WINDOW = 2.0      # |c| window for the near-zero efficacy fit (matches gate_a_analysis)
FIXED_C = -0.5         # baseline fixed dose used in the paper's Table 1
ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5]
SEED = 42


# ---------------------------------------------------------------- loading
def load_sweep(label):
    for p in (f"{R}/gateA_traj/gateA_sweep_{label}.json", f"{R}/gateA_sweep_{label}.json"):
        if os.path.exists(p):
            return json.load(open(p))
    return None


def stats(d, prompt_idx=None):
    """Per-model quantities, optionally restricted to a subset of prompts."""
    g = np.array(d["gaps_massmean"], dtype=float)          # (n_doses, n_prompts)
    c = np.array(d["c_grid_massmean"], dtype=float)
    zi = int(d["zero_index_massmean"])
    if prompt_idx is not None:
        g = g[:, prompt_idx]
    g0 = g[zi]                                             # unsteered gaps
    D = (g0[None, :] - g).mean(axis=1)                     # mean displacement per dose
    return dict(c=c, g=g, zi=zi, g0=g0, D=D,
                margin=float(g0.mean()), std=float(g0.std()),
                dprime=float(g0.mean() / g0.std()),
                efficacy=efficacy_nearzero(c, D),
                mu=float(d.get("mu_used", np.nan)),
                n=g.shape[1])


def efficacy_nearzero(c, D, window=NEAR_WINDOW):
    """Slope of displacement vs |c| in the near-zero regime (with intercept)."""
    absc = np.abs(c)
    m = absc <= window
    A = np.vstack([absc[m], np.ones(m.sum())]).T
    return float(np.linalg.lstsq(A, D[m], rcond=None)[0][0])


# ---------------------------------------------------------------- protocol
def crossing_at(c_target, c_grid, g):
    """Fraction of prompts whose gap < 0 at dose c_target (per-prompt interp)."""
    if not np.isfinite(c_target):
        return np.nan
    c_target = float(np.clip(c_target, c_grid.min(), c_grid.max()))
    vals = np.array([np.interp(c_target, c_grid, g[:, p]) for p in range(g.shape[1])])
    return float((vals < 0).mean())


def invert_displacement(target_D, c_grid, D):
    """Smallest-|c| dose on the negative branch achieving mean displacement target_D.

    Steering toward compliance uses c < 0, where D increases as c decreases.
    Returns the grid minimum if target exceeds the achievable ceiling (saturation).
    """
    neg = c_grid <= 0
    cn, Dn = c_grid[neg], D[neg]
    order = np.argsort(Dn)                  # ascending in D
    Dn_s, cn_s = Dn[order], cn[order]
    if target_D >= Dn_s[-1]:                # beyond the measured ceiling
        return float(cn_s[-1]), True        # saturated
    return float(np.interp(target_D, Dn_s, cn_s)), False


def doses(st, alpha):
    """Both margin-normalized dose variants for one model at level alpha."""
    target = alpha * st["margin"]
    c_lin = -target / st["efficacy"] if st["efficacy"] > 0 else np.nan
    c_inv, sat = invert_displacement(target, st["c"], st["D"])
    return c_lin, c_inv, sat


# ---------------------------------------------------------------- evaluation
def spread(vals):
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    return float(v.max() - v.min()) if v.size else np.nan


def kendall_tau(a, b):
    """Kendall tau-b, no scipy dependency."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    con = dis = ta = tb = 0
    for i, j in itertools.combinations(range(n), 2):
        da, db = a[i] - a[j], b[i] - b[j]
        if da == 0 and db == 0:
            continue
        if da == 0:
            ta += 1
        elif db == 0:
            tb += 1
        elif np.sign(da) == np.sign(db):
            con += 1
        else:
            dis += 1
    den = np.sqrt((con + dis + ta) * (con + dis + tb))
    return float((con - dis) / den) if den > 0 else np.nan


def evaluate(fit_idx=None, eval_idx=None):
    """Run all protocols. fit_idx/eval_idx implement the split-half control."""
    out = {}
    for name, key in ORDER:
        d = load_sweep(key)
        if d is None:
            continue
        st_fit = stats(d, fit_idx)                     # margin/efficacy from fit split
        st_ev = stats(d, eval_idx)                     # crossing measured on eval split
        rec = dict(margin=st_fit["margin"], dprime=st_fit["dprime"],
                   efficacy=st_fit["efficacy"], mu=st_fit["mu"],
                   n_fit=st_fit["n"], n_eval=st_ev["n"])
        rec["mu_fixed"] = crossing_at(FIXED_C, st_ev["c"], st_ev["g"])
        for a in ALPHAS:
            c_lin, c_inv, sat = doses(st_fit, a)
            rec[f"lin_a{a}"] = crossing_at(c_lin, st_ev["c"], st_ev["g"])
            rec[f"inv_a{a}"] = crossing_at(c_inv, st_ev["c"], st_ev["g"])
            rec[f"c_inv_a{a}"] = c_inv
            rec[f"sat_a{a}"] = bool(sat)
        out[name] = rec
    return out


def summarize(res, title):
    names = list(res.keys())
    print(f"\n===== {title} =====")
    hdr = "%-14s %7s %7s %8s | %8s" % ("checkpoint", "margin", "d'", "lever", "mu_fixed")
    hdr += "".join(" %8s" % f"inv a={a}" for a in ALPHAS)
    print(hdr)
    for n in names:
        r = res[n]
        row = "%-14s %7.2f %7.2f %8.2f | %8.2f" % (
            n, r["margin"], r["dprime"], r["efficacy"], r["mu_fixed"])
        row += "".join(" %8.2f" % r[f"inv_a{a}"] for a in ALPHAS)
        print(row)

    print("\n-- crossing-rate SPREAD across checkpoints (lower = fairer instrument) --")
    s_fixed = spread([res[n]["mu_fixed"] for n in names])
    print("  mu-normalized fixed dose (c=%.2f) : %.3f   <- the baseline that fails" % (FIXED_C, s_fixed))
    best = None
    for a in ALPHAS:
        s_lin = spread([res[n][f"lin_a{a}"] for n in names])
        s_inv = spread([res[n][f"inv_a{a}"] for n in names])
        nsat = sum(res[n][f"sat_a{a}"] for n in names)
        print("  margin-normalized a=%-5s linear %.3f | inverted %.3f  (saturated %d/%d)"
              % (a, s_lin, s_inv, nsat, len(names)))
        if best is None or s_inv < best[1]:
            best = (a, s_inv)
    print("  best inverted: alpha=%s spread=%.3f" % best)

    # ranking recovery vs ground truth = per-model efficacy (lever) ordering
    truth = [res[n]["efficacy"] for n in names]
    tau_fixed = kendall_tau([res[n]["mu_fixed"] for n in names], truth)
    tau_inv = kendall_tau([res[n]["inv_a1.0"] for n in names], truth)
    print("\n-- ranking recovery (Kendall tau vs per-dose lever) --")
    print("  mu-normalized fixed dose : tau = %+.3f" % tau_fixed)
    print("  margin-normalized a=1.0  : tau = %+.3f" % tau_inv)
    return dict(spread_fixed=s_fixed, best_alpha=best[0], best_spread=best[1],
                tau_fixed=tau_fixed, tau_inv=tau_inv)


def main():
    print("Margin-normalized steering audit  (Track A, zero-GPU)")
    print("mu CV across checkpoints:", end=" ")
    mus = []
    for _, k in ORDER:
        d = load_sweep(k)
        if d:
            mus.append(d.get("mu_used", np.nan))
    mus = np.array(mus, float)
    print("%.4f  (flat => mu-normalization cannot equalize the boundary)" % (mus.std() / mus.mean()))

    full = evaluate()
    s_full = summarize(full, "A1/A2  in-split (fit and evaluate on all 200 prompts)")

    # ---- A3 circularity control: fit on S1, evaluate on disjoint S2 ----
    d0 = load_sweep(ORDER[0][1])
    n = np.array(d0["gaps_massmean"]).shape[1]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    s1, s2 = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    held = evaluate(fit_idx=s1, eval_idx=s2)
    s_held = summarize(held, "A3  HELD-OUT (margin/lever fit on S1, crossing scored on S2)")

    out = dict(experiment="margin-normalized audit", seed=SEED, fixed_c=FIXED_C,
               alphas=ALPHAS, near_window=NEAR_WINDOW, mu_cv=float(mus.std() / mus.mean()),
               in_split=full, held_out=held,
               summary=dict(in_split=s_full, held_out=s_held))
    os.makedirs(R, exist_ok=True)
    json.dump(out, open(f"{R}/margin_audit.json", "w"), indent=2)
    print("\nwrote %s/margin_audit.json" % R)

    print("\n=== GATE 1 ===")
    print("in-split  spread: %.3f -> %.3f" % (s_full["spread_fixed"], s_full["best_spread"]))
    print("held-out  spread: %.3f -> %.3f" % (s_held["spread_fixed"], s_held["best_spread"]))
    verdict = "PASS" if s_held["best_spread"] < 0.20 else "FAIL"
    print("held-out spread < 0.20 ? %s" % verdict)


if __name__ == "__main__":
    main()
