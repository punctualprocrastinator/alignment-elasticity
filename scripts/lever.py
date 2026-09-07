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
  checkpoints sit at 96-99% of their ceiling by |c|=2, so their "slope" is mostly
  plateau, while Instruct is still rising. Saturation onset is set by the margin,
  so the window fraction is a direct function of the very quantity the lever is
  supposed to be independent of.

  DEFAULT_WINDOW=1.0 NARROWS this problem but does not remove it: at |c|=1 base
  is still at 0.90 of its ceiling, rl-zero at 0.91-0.92, against Instruct's 0.62,
  so the window fraction still ranges 0.58-0.92 across checkpoints. There is no
  window on this grid that is genuinely pre-saturation for every checkpoint, and
  the module should not be read as claiming one. (Note also that ceiling() is
  max displacement over the SAMPLED grid, and for 6 of 10 checkpoints the argmax
  sits at the grid edge c=-12, so those "fractions of ceiling" are upper bounds.)

Effect of fixing both: CV 0.127 -> 0.078. But 1.0 sits at the MINIMUM of the
CV-vs-window curve (0.116 / 0.078 / 0.082 / 0.106 at windows 0.5/1.0/1.5/2.0),
and the Instruct/base ratio crosses 1.0 within that same range (0.94 / 1.14 /
1.27 / 1.39). So: the direction of the improvement over efficacy_published is
robust across every window, its magnitude is not, and no claim should rest on
Instruct's lever being LARGER than base's. Use window_sensitivity() and
efficacy_secant_mean() as the robustness companions, and report them.
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
                      c_grid_random, window=DEFAULT_WINDOW, return_sem=False):
    """Lever with the random-direction floor subtracted.

    `c_grid_random` is REQUIRED and must be the sweep's own `c_grid_random`.
    It is stored DESCENDING (+1 ... -12) while `c_grid_massmean` is ASCENDING
    (-12 ... +1) -- they are exact reverses (gate_a.py builds the null grid with
    reverse=True). Defaulting it to `c_grid` silently pairs each random-direction
    row with a reversed abscissa and produces garbage (base floor -0.03 -> +1.38,
    Instruct +0.99 -> +8.99, CV 0.058 -> 1.55), so there is no default.

    On the floor itself: it is estimated from only 20 directions with
    per-direction sd ~0.9-1.5 (SEM ~0.2-0.3), so individual checkpoint floors are
    noisy and their differences are near the noise scale. Subtraction is
    nonetheless the right correction -- the floor is an additive displacement in
    the same units against the same g0; a ratio is undefined because the floor
    changes sign across checkpoints, and a z-score would answer "is this
    direction unusual among random ones" rather than "how much of the lever is
    non-specific". Use return_sem=True and report the SEM with any quantitative
    statement about how the floor varies.
    """
    gr = np.asarray(gaps_random, dtype=float)      # (k, n_doses, n_prompts)
    g0 = np.asarray(gaps, dtype=float)[zero_index]
    cr = np.asarray(c_grid_random, dtype=float)
    if cr.shape[0] != gr.shape[1]:
        raise ValueError("c_grid_random has %d doses but gaps_random has %d"
                         % (cr.shape[0], gr.shape[1]))
    per_dir = np.array([efficacy(cr, (g0[None, :] - gr[j]).mean(axis=1), window=window)
                        for j in range(gr.shape[0])])
    floor = float(np.mean(per_dir))
    spec = efficacy(c_grid, disp, window=window) - floor
    if return_sem:
        return spec, floor, float(per_dir.std(ddof=1) / np.sqrt(len(per_dir)))
    return spec, floor


def efficacy_secant_mean(c_grid, disp, window=DEFAULT_WINDOW):
    """Robustness companion to efficacy(): the UNWEIGHTED mean of the per-dose
    secants D(c)/|c| inside the window.

    efficacy() minimises squared error through the origin, i.e. Sum(xy)/Sum(x^2),
    which weights each dose by x^2. Over the four in-window doses that is
    53/30/13/3 percent for |c| = 1.00/0.75/0.50/0.25 -- so the estimator whose
    stated purpose is to avoid saturation gives the LARGEST dose sixteen times
    the weight of the smallest. This variant weights them equally.
    """
    c = np.asarray(c_grid, dtype=float)
    D = np.asarray(disp, dtype=float)
    m = (c < 0) & (np.abs(c) <= window)
    if m.sum() < 1:
        return float("nan")
    return float(np.mean(D[m] / np.abs(c[m])))


def window_sensitivity(c_grid, disp_list, windows=(0.5, 1.0, 1.5, 2.0)):
    """CV of the lever across checkpoints, as a function of the fit window.

    DEFAULT_WINDOW=1.0 happens to sit at the minimum of this curve, so quoting
    the CV there alone would be reporting the argmin of a tuning knob. Callers
    should report the whole curve. The sign of the improvement over
    efficacy_published is robust across every window; the magnitude is not.
    """
    out = {}
    for w in windows:
        e = np.array([efficacy(c_grid, d, window=w) for d in disp_list])
        out[w] = float(e.std() / e.mean())
    return out


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
    # A NaN in the unsteered gaps would silently SHRINK the denominator here
    # (nan > 0 is False, so the prompt vanishes from elig with no error and the
    # function still returns a plausible rate). Fail loudly instead.
    if not np.isfinite(g[zero_index]).all():
        raise ValueError("non-finite unsteered gaps: crossing rate denominator "
                         "would be silently reduced")
    elig = g[zero_index] > 0 if conditional else np.ones(g.shape[1], bool)
    if elig.sum() == 0:
        return float("nan")
    return float((g[dose_index][elig] < 0).mean())


def secant_at_crossing(c_grid, disp, crossing_dose):
    """Secant lever from 0 to a model's OWN crossing dose: D(c50)/|c50|.

    The pre-audit code took the secant at a FIXED |c|=0.5 for every model while
    the text described it as "to each model's own crossing dose". Evaluated as
    described, the comparison reverses: base 5.56, Instruct 6.44, against the
    fixed-dose version's base 6.42, Instruct 5.99.

    np.interp CLAMPS outside the grid rather than extrapolating, so a crossing
    dose beyond |c|=12 would silently return a meaningless value; all ten real
    c50 are within [0.26, 1.23].
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
