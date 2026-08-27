GA_VERDICT = GA_SUMMARY["verdict"]

mo.vstack([
    mo.md("## Verdict"),
    mo.md("**Outcome %s**" % GA_VERDICT["outcome"]),
    mo.md(GA_VERDICT["statement"]),
    mo.md(
        "Decisive statistic: `%s`. Between-model spread of the crossing-rate "
        "curve: **%.3f** matched on `c`, **%.3f** matched on achieved "
        "displacement, **%.3f** matched on displacement past each model's own "
        "boundary."
        % (GA_VERDICT["decisive_statistic"],
           GA_VERDICT["mean_spread_by_c"],
           GA_VERDICT["mean_spread_by_disp"],
           GA_VERDICT["mean_spread_by_disp_rel"])
    ),
    mo.md(
        "Raw `d_50` is pinned to baseline depth by construction - crossing the "
        "median prompt requires displacing it by roughly its own unsteered gap - "
        "so equal raw `d_50` is NOT the no-residual-effect null. The excess "
        "`d_50 - median(unsteered gap)` is."
    ),
])
