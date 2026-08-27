def ga_d50_table(summary, which="massmean"):
    """The headline table: where each model sits, what dose moves it, and what
    achieved displacement it takes to move half its prompts across."""
    _bt = summary["bootstrap_" + which]["table"]
    _rows = []
    for _lab in summary["models"]:
        _t = _bt[_lab]
        _rows.append({
            "model": _lab,
            "unsteered_gap": round(_t["unsteered_gap"], 3),
            "pct_refusing": round(100 * _t["frac_refusing_unsteered"], 1),
            "mu_L20": round(_t["mu"], 2),
            "c50": (None if not np.isfinite(_t["c50"]) else round(_t["c50"], 3)),
            "c50_CI": (None if _t["c50_ci"]["lo"] is None else
                       [round(_t["c50_ci"]["lo"], 3), round(_t["c50_ci"]["hi"], 3)]),
            "d50": (None if not np.isfinite(_t["d50"]) else round(_t["d50"], 3)),
            "d50_CI": (None if _t["d50_ci"]["lo"] is None else
                       [round(_t["d50_ci"]["lo"], 3), round(_t["d50_ci"]["hi"], 3)]),
        })
    return _rows


GA_TABLE_MM = ga_d50_table(GA_SUMMARY, "massmean")
GA_TABLE_LR = ga_d50_table(GA_SUMMARY, "logistic")

mo.vstack([
    mo.md("### d_50 table - MASS-MEAN direction (headline)"),
    mo.ui.table(GA_TABLE_MM, selection=None),
    mo.md("### d_50 table - LOGISTIC direction (tol=1e-10)"),
    mo.ui.table(GA_TABLE_LR, selection=None),
    mo.md("### Paired bootstrap d_50 differences (mass-mean)"),
    mo.ui.table(
        [
            {"pair": _k[4:], "d50_diff": (None if _v["point"] is None else round(_v["point"], 3)),
             "CI95": (None if _v["lo"] is None else [round(_v["lo"], 3), round(_v["hi"], 3)]),
             "excludes_zero": _v["excludes_zero"]}
            for _k, _v in GA_SUMMARY["bootstrap_massmean"]["diffs"].items()
            if _k.startswith("d50:")
        ],
        selection=None,
    ),
])
