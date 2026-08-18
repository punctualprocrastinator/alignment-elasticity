import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
## A1.3 Smoke tests (run BEFORE the science)

Five preconditions from `project.md`. Any failure invalidates everything
downstream, so they are checked on the base model right after the direction is
fitted, and the results are stored in `smoke.json`.

1. **Hook liveness** - ablation must actually change the logits. Olmo3 decoder
   layers return a plain `Tensor`, not a tuple; a hook that assumes a tuple
   silently no-ops and produces a beautiful null result.
2. **Per-layer variation** - extracted activations must differ across layers.
3. **Truncation audit** - fraction of prompts hitting `MAX_LEN`.
4. **Degenerate-direction check** - reject any layer whose difference-in-means
   norm is ~0 relative to the activation scale at that layer.
5. **Random-direction control** - available and unit-norm, for the null band.
"""
)'''

SMOKE = '''A1_SMOKE = a1p.read_json(A1_PATHS["smoke"])

A1_SMOKE_DF = pd_a1.DataFrame(
    [
        {
            "test": _t["name"],
            "result": "PASS" if _t["pass"] else "FAIL",
            "evidence": _e,
        }
        for _t, _e in zip(
            A1_SMOKE["tests"],
            [
                "max |dlogit| = %.3f (embed-only hook alone: %.3f)"
                % (
                    A1_SMOKE["tests"][0]["max_abs_logit_delta"],
                    A1_SMOKE["tests"][0]["embed_only_max_abs_delta"],
                ),
                "max off-diagonal layer cosine = %.4f"
                % A1_SMOKE["tests"][1]["max_offdiag_cosine"],
                "%d/%d at MAX_LEN=%d (%.2f%%); len mean %.1f, p95 %.0f, max %d"
                % (
                    A1_SMOKE["tests"][2]["n_at_or_over_max"],
                    A1_SMOKE["tests"][2]["n"],
                    A1_SMOKE["tests"][2]["max_len"],
                    100 * A1_SMOKE["tests"][2]["frac_truncated"],
                    A1_SMOKE["tests"][2]["len_mean"],
                    A1_SMOKE["tests"][2]["len_p95"],
                    A1_SMOKE["tests"][2]["len_max"],
                ),
                "rejected layers: %s; min relative norm %.3f (floor %.2f)"
                % (
                    A1_SMOKE["tests"][3]["rejected_layers"] or "none",
                    min(A1_SMOKE["tests"][3]["relative_norm"]),
                    A1_SMOKE["tests"][3]["rel_floor"],
                ),
                "%d unit directions in R^%d, norms in [%.6f, %.6f]"
                % (
                    A1_SMOKE["tests"][4]["k"],
                    A1_SMOKE["tests"][4]["d_model"],
                    A1_SMOKE["tests"][4]["norm_min"],
                    A1_SMOKE["tests"][4]["norm_max"],
                ),
            ],
        )
    ]
)

A1_LAYER_NORMS = pd_a1.DataFrame(
    {
        "layer": A1_SMOKE["tests"][3]["layers"],
        "diff_in_means_norm": A1_SMOKE["tests"][3]["dim_norm"],
        "act_scale": A1_SMOKE["tests"][3]["act_scale"],
        "relative_norm": A1_SMOKE["tests"][3]["relative_norm"],
    }
)

print("ALL SMOKE TESTS PASS:", A1_SMOKE["all_pass"])
print(A1_SMOKE_DF.to_string(index=False))
print()
print("per-layer difference-in-means (fit layer =", A1_SMOKE["fit_layer"], ")")
print(A1_LAYER_NORMS.to_string(index=False))'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(SMOKE, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
