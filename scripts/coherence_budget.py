# -*- coding: utf-8 -*-
"""Coherence-budgeted controllability: the objection both InterpScience
reviewers raised, and the one the paper never answered.

The paper claims fixed-dose audits MISRANK aligned models: the lever is intact,
only the margin grew, so the aligned model is not really less controllable.
Both reviewers pushed back on the same point:

  MaBV: "A direction can keep moving a refusal score even when actually changing
         the response takes an intervention strong enough to break coherent
         generation ... Such tests may be capturing a real difference under a
         limited intervention budget."
  JMJy: "the resulting incoherence in outputs from aligned models suggests that
         this method does not produce a genuine behavioral shift."

If crossing an aligned model's margin requires a dose that destroys fluency,
then under any realistic intervention budget the aligned model IS less
controllable, and "misrank" is backwards rather than merely over-framed.

THE TEST. For each checkpoint, define a coherence budget: the largest |c| whose
generations stay as fluent as that model's own UNSTEERED output (repetition
within the unsteered 95th percentile, distinct-3 above its 5th percentile), for
at least BUDGET_FRAC of prompts. Then ask what the direction achieves AT that
budget, not at whatever dose happens to flip the onset.

  H_paper     within-budget crossing is comparable across checkpoints
              -> the fixed-dose misranking claim survives
  H_reviewers within-budget crossing collapses for aligned models
              -> aligned models are genuinely less controllable under a
                 fluency constraint; the normative claim must be withdrawn and
                 the contribution rescoped to the lever/load decomposition

Zero GPU: reads the stored E2 generations (5 checkpoints x 7 doses x 80 prompts).
"""
import json, os, re, sys
from collections import Counter
import numpy as np

CKPTS = ["base", "sft-1000", "dpo", "rlvr-last", "instruct"]
BUDGET_FRAC = 0.80          # a dose is "affordable" if >=80% of outputs stay fluent
OUT = "results/coherence_budget.json"


def strip_think(t):
    return t.split("</think>", 1)[1] if "</think>" in t else t


def rep(text):
    w = re.sub(r"\s+", " ", strip_think(text)).strip().split()
    if len(w) < 10:
        return dict(maxtok=np.nan, d3=np.nan)
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return dict(maxtok=Counter(w).most_common(1)[0][1] / len(w),
                d3=len(set(tri)) / max(1, len(tri)))


def main():
    out = {"experiment": "coherence-budgeted controllability",
           "budget_frac": BUDGET_FRAC, "checkpoints": {}}
    print("%-11s %7s %9s %9s %9s %9s" %
          ("ckpt", "c", "coher_fr", "refus_rate", "affordable", "gap_cross"))
    for lab in CKPTS:
        d = json.load(open("results/E2_dose_%s.json" % lab))
        pcs = sorted(d["per_coeff"], key=lambda e: abs(e["coeff"]))
        uns = [e for e in pcs if e["coeff"] == 0.0][0]
        ur = [rep(t) for t in uns["texts"]]
        gate = dict(hi=float(np.nanpercentile([r["maxtok"] for r in ur], 95)),
                    lo=float(np.nanpercentile([r["d3"] for r in ur], 5)))
        rows, budget = [], None
        for e in pcs:
            sr = [rep(t) for t in e["texts"]]
            coh = np.array([np.isfinite(r["maxtok"]) and r["maxtok"] <= gate["hi"]
                            and r["d3"] >= gate["lo"] for r in sr])
            gaps = np.array(e["per_prompt_gap"], float)
            rr = float(np.mean(e["refusal"]))
            aff = bool(coh.mean() >= BUDGET_FRAC)
            rows.append(dict(coeff=e["coeff"], coherent_frac=float(coh.mean()),
                             refusal_rate=rr, affordable=aff,
                             gap_cross=float((gaps < 0).mean())))
            if aff:
                budget = rows[-1]
            print("%-11s %7.3f %9.2f %9.2f %9s %9.2f" %
                  (lab, e["coeff"], coh.mean(), rr, "yes" if aff else "no",
                   (gaps < 0).mean()))
        uns_rr = rows[0]["refusal_rate"]
        out["checkpoints"][lab] = dict(
            gate=gate, doses=rows, c50=d.get("c50_massmean"),
            unsteered_refusal=uns_rr,
            budget=budget,
            budget_refusal_drop=(uns_rr - budget["refusal_rate"]) if budget else None)
        print()
    print("=" * 74)
    print("%-11s %9s %11s %12s %12s" %
          ("ckpt", "budget c", "coher_fr", "refusal", "drop from 0"))
    for lab in CKPTS:
        r = out["checkpoints"][lab]
        b = r["budget"]
        print("%-11s %9s %11s %12s %12s" % (
            lab,
            "%.3f" % b["coeff"] if b else "none",
            "%.2f" % b["coherent_frac"] if b else "-",
            "%.2f" % b["refusal_rate"] if b else "-",
            "%+.2f" % (-r["budget_refusal_drop"]) if b else "-"))
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Threshold-free budget: c_half
#
# The BUDGET_FRAC threshold above is arbitrary, and worse, it is evaluated on a
# grid that is c50-RELATIVE: each model was swept at c50 x [0.5, 1, 1.5, ...],
# so the smallest dose ever tested scales with c50 (base 0.136, instruct 0.612,
# 4.5x coarser). For dpo/rlvr/instruct the thresholded budget came out exactly
# equal to their first nonzero dose, i.e. it was bounded above by the grid floor
# and never actually resolved.
#
# c_half instead interpolates each model's own coherence-vs-dose curve and takes
# the dose where the coherent fraction crosses 0.50. It is threshold-free in the
# sense that 0.50 is a property of the curve's midpoint rather than a cutoff
# imposed on a coarse grid, and it uses all the doses rather than the first one
# that clears a bar.
#
# CAVEAT that the fine-grid experiment must fix: instruct's c_half is
# interpolated across a 0.61-wide gap (its only bracketing doses are 0.612 and
# 1.224), so it is suggestive, not resolved. base's is bracketed within 0.136.
# The aligned models are exactly the ones whose budget is least well measured.
def c_half(doses, level=0.50):
    """Dose at which the coherent fraction crosses `level`, by linear interp."""
    import numpy as np
    ds = sorted(doses, key=lambda d: abs(d["coeff"]))
    x = np.array([abs(d["coeff"]) for d in ds])
    y = np.array([d["coherent_frac"] for d in ds])
    if not (y.min() <= level <= y.max()):
        return float("nan")
    return float(np.interp(level, y[::-1], x[::-1]))


def budget_report(path=OUT):
    """c_half, c_cross and their ratio. Ratio >= 1 predicts within-budget steerability."""
    import numpy as np
    cb = json.load(open(path))
    print("%-11s %9s %9s %8s %s" % ("ckpt", "c_half", "c_cross", "ratio", "steerable?"))
    out = {}
    for lab, r in cb["checkpoints"].items():
        ch = c_half(r["doses"])
        cc = abs(r["c50"])
        out[lab] = dict(c_half=ch, c_cross=cc, ratio=ch / cc)
        print("%-11s %9.3f %9.3f %8.2f %s"
              % (lab, ch, cc, ch / cc, "YES" if ch / cc >= 1 else "no"))
    v = np.array([out[l]["c_half"] for l in out])
    print("\nbudget spread %.2fx (%.3f-%.3f) vs c_cross spread %.2fx"
          % (v.max() / v.min(), v.min(), v.max(),
             max(out[l]["c_cross"] for l in out) / min(out[l]["c_cross"] for l in out)))
    cb["c_half"] = out
    json.dump(cb, open(path, "w"), indent=1)
    return out
