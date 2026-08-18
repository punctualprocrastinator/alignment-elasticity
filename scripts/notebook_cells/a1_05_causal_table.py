import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
## A1.4 Powered causal table

`delta = intact - ablated` per held-out prompt (n=200), where the quantity is
`mean logprob(refusal onset) - mean logprob(compliance onset)` at the first
generated position. Positive delta = ablating the base direction removed
refusal-leaning probability mass.

* **95% CI** - percentile bootstrap of the mean, 2000 resamples.
* **% of base** - `mean(delta_model) / mean(delta_base)`, PAIRED bootstrap
  (the same resampled prompt indices in numerator and denominator), which is
  valid because every checkpoint is scored on the identical 200 prompts. The
  base row is exactly 100% by construction.
* **z vs null** - the base direction's mean delta expressed in standard
  deviations of that checkpoint's OWN 20 random-unit-direction null.
"""
)'''

CAUSAL = '''A1_NBOOT = 2000

A1_RESULTS = {
    _lab: a1p.read_json(A1_PATHS["causal"](_lab)) for _r, _v, _lab in a1p.CKPTS_A1
}
A1_BASE_DELTA = np_a1.array(A1_RESULTS["base"]["delta"])

_rows = []
for _repo, _rev, _lab in a1p.CKPTS_A1:
    _rec = A1_RESULTS[_lab]
    _d = np_a1.array(_rec["delta"])
    _m, _lo, _hi = a1p.bootstrap_ci(_d, n_boot=A1_NBOOT)
    _null = np_a1.array(_rec["null_means"])
    _z, _nmean, _nsd = a1p.zscore_vs_null(_m, _null)
    _pct, _plo, _phi = a1p.bootstrap_ratio_ci(_d, A1_BASE_DELTA, n_boot=A1_NBOOT)
    _rows.append(
        {
            "model": _lab,
            "revision": _rev,
            "n": len(_d),
            "intact": float(np_a1.mean(_rec["intact"])),
            "ablated": float(np_a1.mean(_rec["ablated"])),
            "delta": _m,
            "ci_lo": _lo,
            "ci_hi": _hi,
            "pct_of_base": 100.0 * _pct,
            "pct_lo": 100.0 * _plo,
            "pct_hi": 100.0 * _phi,
            "null_mean": _nmean,
            "null_sd": _nsd,
            "null_lo": float(np_a1.percentile(_null, 2.5)),
            "null_hi": float(np_a1.percentile(_null, 97.5)),
            "z_vs_null": _z,
        }
    )

A1_CAUSAL = pd_a1.DataFrame(_rows)

A1_CAUSAL_SHOW = pd_a1.DataFrame(
    {
        "model": A1_CAUSAL["model"],
        "n": A1_CAUSAL["n"],
        "intact": A1_CAUSAL["intact"].round(3),
        "ablated": A1_CAUSAL["ablated"].round(3),
        "delta [95% CI]": [
            "%.3f [%.3f, %.3f]" % (r.delta, r.ci_lo, r.ci_hi)
            for r in A1_CAUSAL.itertuples()
        ],
        "% of base [95% CI]": [
            "%.1f [%.1f, %.1f]" % (r.pct_of_base, r.pct_lo, r.pct_hi)
            for r in A1_CAUSAL.itertuples()
        ],
        "random null (mean +- sd)": [
            "%.4f +- %.4f" % (r.null_mean, r.null_sd) for r in A1_CAUSAL.itertuples()
        ],
        "z vs null": A1_CAUSAL["z_vs_null"].round(1),
    }
)

print(A1_CAUSAL_SHOW.to_string(index=False))
print()
print("yesterday, n=6-8, uncontrolled:  base 100% | rlz-code 86% | rlz-if 87% | instruct 8%")'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(CAUSAL, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
