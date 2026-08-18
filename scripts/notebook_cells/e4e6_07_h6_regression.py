import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
### E6.2 H6 - does training distance add explanatory power once capability is controlled?

Hypothesis H6 from `project.md`: monitors go stale as the model they watch drifts
away from them. The competing account is that detection is limited by the
monitor's own capability, and apparent staleness is a confound.

The regression below predicts catch rate at 5% FPR from three terms:

* `capability` - the monitor's mean AUROC over the *other* seven monitored
  checkpoints (leave-one-out, so the cell being predicted never enters its own
  predictor),
* `difficulty` - the monitored checkpoint's mean AUROC over the *other* three
  monitors (leave-one-out), because generators differ enormously in how legible
  their harmful outputs are,
* `distance` - flow steps between monitor and monitored checkpoint.

CIs are a cluster bootstrap over the 8 monitored checkpoints (1000 reps), which
respects the fact that the 32 cells are not independent.
"""
)'''

H6 = '''_cap, _dif = {}, {}
for _m in E6_MONS:
    for _g in E6_GENS:
        _sub = E6_GRID[(E6_GRID["monitor"] == _m) & (E6_GRID["generator"] != _g)]
        _cap[(_m, _g)] = float(_sub["auroc"].mean())
        _sub2 = E6_GRID[(E6_GRID["generator"] == _g) & (E6_GRID["monitor"] != _m)]
        _dif[(_m, _g)] = float(_sub2["auroc"].mean())

E6_REG = E6_GRID.copy()
E6_REG["capability"] = [_cap[(r.monitor, r.generator)] for r in E6_REG.itertuples()]
E6_REG["difficulty"] = [_dif[(r.monitor, r.generator)] for r in E6_REG.itertuples()]
E6_REG["staleness"] = np_a1.maximum(0, E6_REG["signed_distance"])


def _ols(df, cols, y="catch5"):
    _X = np_a1.column_stack([np_a1.ones(len(df))] + [df[c].values.astype(float) for c in cols])
    _yv = df[y].values.astype(float)
    _beta, _, _, _ = np_a1.linalg.lstsq(_X, _yv, rcond=None)
    _pred = _X @ _beta
    _ss = float(((_yv - _pred) ** 2).sum())
    _tot = float(((_yv - _yv.mean()) ** 2).sum())
    return _beta, 1.0 - _ss / _tot


E6_TERMS = ["capability", "difficulty", "distance"]
_beta_full, _r2_full = _ols(E6_REG, E6_TERMS)
_beta_nod, _r2_nod = _ols(E6_REG, ["capability", "difficulty"])

_rng = np_a1.random.default_rng(42)
_boot = []
for _b in range(1000):
    _pick = _rng.choice(E6_GENS, size=len(E6_GENS), replace=True)
    _df = pd_a1.concat([E6_REG[E6_REG["generator"] == g] for g in _pick], ignore_index=True)
    try:
        _bb, _rr = _ols(_df, E6_TERMS)
        _b2, _r2 = _ols(_df, ["capability", "difficulty"])
        _boot.append(list(_bb) + [_rr - _r2])
    except Exception:
        pass
_boot = np_a1.array(_boot)

E6_H6 = pd_a1.DataFrame({
    "term": ["intercept"] + E6_TERMS + ["partial R2 of distance"],
    "estimate": list(_beta_full) + [_r2_full - _r2_nod],
    "ci_lo": [float(np_a1.percentile(_boot[:, i], 2.5)) for i in range(_boot.shape[1])],
    "ci_hi": [float(np_a1.percentile(_boot[:, i], 97.5)) for i in range(_boot.shape[1])],
})
E6_H6["excludes_zero"] = ["YES" if (r.ci_lo > 0 or r.ci_hi < 0) else "no"
                          for r in E6_H6.itertuples()]

print("H6 regression: catch@5%FPR ~ capability + difficulty + distance   (n=32 cells)")
print(E6_H6.round(4).to_string(index=False))
print()
print("R2 full %.3f | without distance %.3f" % (_r2_full, _r2_nod))
print()
E6_CORR = pd_a1.DataFrame({
    "relationship": ["AUROC vs distance", "AUROC vs staleness",
                     "catch5 vs distance", "catch5 vs capability", "catch5 vs difficulty"],
    "pearson_r": [
        float(np_a1.corrcoef(E6_REG["auroc"], E6_REG["distance"])[0, 1]),
        float(np_a1.corrcoef(E6_REG["auroc"], E6_REG["staleness"])[0, 1]),
        float(np_a1.corrcoef(E6_REG["catch5"], E6_REG["distance"])[0, 1]),
        float(np_a1.corrcoef(E6_REG["catch5"], E6_REG["capability"])[0, 1]),
        float(np_a1.corrcoef(E6_REG["catch5"], E6_REG["difficulty"])[0, 1]),
    ],
})
print(E6_CORR.round(3).to_string(index=False))'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(H6, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
