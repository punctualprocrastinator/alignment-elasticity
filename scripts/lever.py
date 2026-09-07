# -*- coding: utf-8 -*-
"""Canonical lever/load estimators for the Gate-A sweeps.

Single source of truth. These were previously reimplemented in
paper-boundary/{equivalence_test,secant_analysis,make_figures}.py and drifted;
two bugs were shared by all three copies:

  BUG 1 (sign fold). `absc = np.abs(c); near = absc <= 2.0` includes the three
  POSITIVE doses (+0.25, +0.5, +1.0), which steer TOWARD refusal and therefore
  have NEGATIVE displacement. Folding on |c| places opposite-signed y at
  duplicate x, collapsing fit R^2 to ~0.5 and giving every fit a negative
  intercept where a lever through the origin demands ~0. The bias is
  model-dependent (Instruct -7%, others -13..-19%), so it distorts the
  cross-checkpoint comparison the paper rests on.

  BUG 2 (window). |c| <= 2.0 is not "the linear region". base and both RL-Zero
  checkpoints PEAK inside that window and sit at 97-99% of their ceiling at
  |c|=2, so their "slope" is mostly plateau, while Instruct is still rising.
  Saturation onset is set by the margin, so the window fraction is a direct
  function of the very quantity the lever is supposed to be independent of.
  |c| <= 1.0 is the largest window that is pre-saturation for every checkpoint.

Effect of fixing both: CV 0.127 -> 0.078; and see specific_efficacy() below,
which additionally subtracts the random-direction floor (that floor rises 27x
across the flow, from 0.07 at base to 2.02 at Instruct, so NOT subtracting it
inflates the aligned model's apparent lever).
"""
import numpy as np

DEFAULT_WINDOW = 1.0          # pre-saturation for every checkpoint
PUBLISHED_WINDOW = 2.0        # the pre-audit value, kept for reproduction


# ------------------------------------------------------------------ basics
def displacement(gaps, zero_index):
    """Mean per-prompt displacement from the unsteered gap, per dose."""
    g = np.asarray(gaps, dtype=float)
    g0 = g[zero_index]
    return (g0[None, :] - g).mean(axis=1)


def margin(gaps, zero_index):
    """Load: mean unsteered logit gap."""
    return float(np.asarray(gaps, dtype=float)[zero_index].mean())


def dprime(gaps, zero_index):
    """Scale-free margin (mean / sd). Immune to any global logit rescaling."""
    g0 = np.asarray(gaps, dtype=float)[zero_index]
    return float(g0.mean() / g0.std())


# ------------------------------------------------------------------ lever
def efficacy(c_grid, disp, window=DEFAULT_WINDOW, through_origin=True):
    """Per-dose lever: slope of displacement vs |c| on the refusal-REDUCING
    branch only, inside a pre-saturation window.

    through_origin=True because a zero dose must produce zero displacement;
    fitting a free intercept on a folded window was absorbing the sign-fold
    error into a spurious negative offset.
    """
    c = np.asarray(c_grid, dtype=float)
    D = np.asarray(disp, dtype=float)
    m = (c < 0) & (np.abs(c) <= window)
    if m.sum() < 2:
        return float("nan")
    x, y = np.abs(c[m]), D[m]
    if through_origin:
        return float((x @ y) / (x @ x))
    A = np.vstack([x, np.ones(len(x))]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def efficacy_published(c_grid, disp, window=PUBLISHED_WINDOW):
    """The pre-audit estimator (sign-folded, free intercept, |c|<=2).

    Retained ONLY to reproduce published numbers. Do not use for new results.
    """
    c = np.asarray(c_grid, dtype=float)
    D = np.asarray(disp, dtype=float)
    absc = np.abs(c)
    m = absc <= window
    A = np.vstack([absc[m], np.ones(m.sum())]).T
    return float(np.linalg.lstsq(A, D[m], rcond=None)[0][0])


def specific_efficacy(c_grid, disp, gaps_random, gaps, zero_index,
                      c_grid_random=None, window=DEFAULT_WINDOW):
    """Lever with the random-direction floor subtracted.

    The random floor is not constant across the flow: it rises ~27x from base to
    Instruct, because aligned models' gaps move more under ANY perturbation. A
    claim that *this direction specifically* is more effective must clear that
    floor, so report this alongside the raw lever.
    """
    gr = np.asarray(gaps_random, dtype=float)      # (k, n_doses, n_prompts)
    g0 = np.asarray(gaps, dtype=float)[zero_index]
    cr = np.asarray(c_grid_random if c_grid_random is not None else c_grid, dtype=float)
    floor = np.mean([efficacy(cr, (g0[None, :] - gr[j]).mean(axis=1), window=window)
                     for j in range(gr.shape[0])])
    return efficacy(c_grid, disp, window=window) - float(floor), float(floor)


# ------------------------------------------------------------------ crossing
def crossing_rate(gaps, dose_index, zero_index, conditional=True):
    """Fraction of prompts whose gap is < 0 at a dose.

    conditional=True restricts the denominator to prompts that could actually
    cross (unsteered gap > 0), matching gate_a.crossing_rate. The unconditional
    form counts never-refusing prompts as "crossings", which inflates exactly
    the small-margin models (35.5% of base's prompts start below zero; 0% of
    Instruct's), i.e. it biases toward the paper's claim.
    """
    g = np.asarray(gaps, dtype=float)
    elig = g[zero_index] > 0 if conditional else np.ones(g.shape[1], bool)
    if elig.sum() == 0:
        return float("nan")
    return float((g[dose_index][elig] < 0).mean())


def secant_at_crossing(c_grid, disp, crossing_dose):
    """Secant lever from 0 to a model's OWN crossing dose: D(c50)/|c50|.

    The pre-audit code took the secant at a FIXED |c|=0.5 for every model while
    the text described it as "to each model's own crossing dose". Evaluated as
    described, the comparison reverses (base 5.40, Instruct 6.40) relative to
    the fixed-dose version (base 6.42, Instruct 5.99).
    """
    c = np.asarray(c_grid, dtype=float)
    D = np.asarray(disp, dtype=float)
    cd = abs(float(crossing_dose))
    if cd <= 0:
        return float("nan")
    # Negative branch ONLY. Sorting the full grid by |c| interleaves the
    # positive doses, whose displacement is negative, and interpolating across
    # them reproduces the sign-fold bug this module exists to remove.
    neg = c <= 0
    x, y = np.abs(c[neg]), D[neg]
    order = np.argsort(x)
    return float(np.interp(cd, x[order], y[order]) / cd)


def ceiling(disp):
    """Maximum achievable mean displacement (saturation ceiling)."""
    return float(np.asarray(disp, dtype=float).max())
