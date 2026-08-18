import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
### A1.4b Does the CI ordering separate the arms?

Two things are asked of these numbers: (a) is each checkpoint's effect
distinguishable from its own random-direction null, and (b) do the CIs separate
Instruct from the RL-Zero arm. Pairwise differences use a PAIRED bootstrap over
the shared 200 prompts, which is strictly more powerful than eyeballing
overlap of two marginal CIs.

A raw `delta` is a *displacement*, not an *outcome*. A checkpoint that sits at
+7.85 refusal-leaning and is pushed down by 6.8 still ends at +1.04, i.e. it is
still refusing. The **crossing rate** below is the outcome-scaled version: the
fraction of prompts that start refusal-leaning (readout > 0) and end
compliance-leaning (readout < 0) after ablation.
"""
)'''

SEP = '''_pairs = [
    ("instruct", "rlz-math"),
    ("instruct", "rlz-code"),
    ("instruct", "base"),
    ("rlz-math", "base"),
    ("rlz-code", "base"),
    ("rlz-code", "rlz-math"),
]
_rng = np_a1.random.default_rng(a1p.SEED)
_idx = _rng.integers(0, 200, size=(A1_NBOOT, 200))

_rows = []
for _a, _b in _pairs:
    _da = np_a1.array(A1_RESULTS[_a]["delta"])
    _db = np_a1.array(A1_RESULTS[_b]["delta"])
    _boot = _da[_idx].mean(axis=1) - _db[_idx].mean(axis=1)
    _lo, _hi = np_a1.percentile(_boot, [2.5, 97.5])
    _rows.append(
        {
            "comparison": "%s - %s" % (_a, _b),
            "diff": float(_da.mean() - _db.mean()),
            "ci_lo": float(_lo),
            "ci_hi": float(_hi),
            "separated": "YES" if (_lo > 0 or _hi < 0) else "no (CI spans 0)",
        }
    )
A1_PAIRWISE = pd_a1.DataFrame(_rows)

_rows = []
for _repo, _rev, _lab in a1p.CKPTS_A1:
    _rec = A1_RESULTS[_lab]
    _i = np_a1.array(_rec["intact"])
    _ab = np_a1.array(_rec["ablated"])
    _cross = (_i > 0) & (_ab < 0)
    _m, _lo, _hi = a1p.bootstrap_ci(_cross.astype(float), n_boot=A1_NBOOT)
    _rows.append(
        {
            "model": _lab,
            "intact_refusal_leaning": float((_i > 0).mean()),
            "ablated_refusal_leaning": float((_ab > 0).mean()),
            "crossing_rate": _m,
            "cross_lo": _lo,
            "cross_hi": _hi,
        }
    )
A1_CROSS = pd_a1.DataFrame(_rows)

print("PAIRWISE (paired bootstrap on the shared 200 prompts)")
print(A1_PAIRWISE.round(3).to_string(index=False))
print()
print("CROSSING: fraction of prompts pushed from refusal-leaning to compliance-leaning")
print(
    pd_a1.DataFrame(
        {
            "model": A1_CROSS["model"],
            "intact >0": A1_CROSS["intact_refusal_leaning"].round(3),
            "ablated >0": A1_CROSS["ablated_refusal_leaning"].round(3),
            "crossing rate [95% CI]": [
                "%.3f [%.3f, %.3f]" % (r.crossing_rate, r.cross_lo, r.cross_hi)
                for r in A1_CROSS.itertuples()
            ],
        }
    ).to_string(index=False)
)'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(SEP, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
