GA_BEHAV_ROWS = [
    {
        "model": _lab,
        "coeff (own c_50)": round(_b["coeff"], 3),
        "n": _b["n"],
        "max_new_tokens": _b["max_new_tokens"],
        "refusal_unsteered": round(_b["unsteered_refusal_rate"], 3),
        "refusal_steered": round(_b["steered_refusal_rate"], 3),
        "unclosed_think_unsteered": round(_b["unsteered_think"]["frac_unclosed_think"], 3),
        "unclosed_think_steered": round(_b["steered_think"]["frac_unclosed_think"], 3),
    }
    for _lab, _b in GA_SUMMARY["behavioural"].items()
]

mo.vstack([
    mo.md("### Behavioural check - greedy generation at each model's own c_50"),
    mo.ui.table(GA_BEHAV_ROWS, selection=None),
    mo.md(
        "The logit-gap crossing criterion is weaker than a full behavioural "
        "refusal flip, so refusal rates move less than crossing rate at the same "
        "coefficient. Report both; do not substitute one for the other (P4)."
    ),
])
