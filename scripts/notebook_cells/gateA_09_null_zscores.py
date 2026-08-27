def ga_null_rows(summary, which="massmean", labels=None):
    """Base-direction effect as a z-score against 20 random unit directions at
    MATCHED coefficient and matched norm (protocol P3)."""
    _rows = []
    for _lab in (labels or summary["models"]):
        for _r in summary["z_vs_null_" + which][_lab]:
            if _r["c"] not in (-4.0, -2.0, -1.0, -0.5):
                continue
            _rows.append({
                "model": _lab,
                "c": _r["c"],
                "crossing_rate": round(_r["rate"], 3),
                "null_rate_mean": round(_r["null_rate_mean"], 4),
                "rate_z": (None if _r["rate_z"] is None else round(_r["rate_z"], 1)),
                "displacement": round(_r["disp"], 3),
                "null_disp_mean": round(_r["null_disp_mean"], 4),
                "disp_z": (None if _r["disp_z"] is None else round(_r["disp_z"], 1)),
            })
    return _rows


GA_NULL_ROWS = ga_null_rows(GA_SUMMARY, "massmean")

mo.vstack([
    mo.md("### Random-direction null (20 directions per model, matched norm)"),
    mo.ui.table(GA_NULL_ROWS, selection=None),
])
