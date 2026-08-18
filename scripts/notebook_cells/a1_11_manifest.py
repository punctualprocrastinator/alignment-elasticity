import marimo._code_mode as cm

SUMM = '''A1_SUMMARY = {
    "experiment": "A1 - powered causal test of the base-model refusal direction",
    "date": time_a1.strftime("%Y-%m-%d"),
    "pipeline": "/marimo/pipeline.py",
    "seed": a1p.SEED,
    "scaffold": "User: {prompt} + newline + Assistant:",
    "max_len": a1p.MAX_LEN,
    "truncation_side": "left",
    "padding_side": "right",
    "fit_layer": a1p.FIT_LAYER,
    "fit_split": {"harmful": len(A1_HARM_FIT), "benign": len(A1_BEN_FIT)},
    "heldout_harmful": len(A1_HARM_HELD),
    "n_random_controls": 20,
    "n_bootstrap": A1_NBOOT,
    "refusal_ids": A1_RESULTS["base"]["refusal_ids"],
    "refusal_strings": a1p.REFUSAL_STRS,
    "comply_ids": A1_RESULTS["base"]["comply_ids"],
    "comply_strings": a1p.COMPLY_STRS,
    "checkpoints": [
        {"label": l, "repo": r, "revision": v} for r, v, l in a1p.CKPTS_A1
    ],
    "smoke": {t["name"]: t["pass"] for t in A1_SMOKE["tests"]},
    "smoke_all_pass": A1_SMOKE["all_pass"],
    "causal": A1_CAUSAL.to_dict(orient="records"),
    "pairwise": A1_PAIRWISE.to_dict(orient="records"),
    "crossing": A1_CROSS.to_dict(orient="records"),
    "generation": A1_GEN_DF.to_dict(orient="records"),
    "wall_seconds": a1p.read_json(A1_PATHS["status"])["total_seconds"],
    "figure": A1_FIG_PATH,
}

A1_SUMMARY_PATH = a1p.write_json(os_a1.path.join(A1_PATHS["art"], "a1_summary.json"), A1_SUMMARY)
print("wrote", A1_SUMMARY_PATH, os_a1.path.getsize(A1_SUMMARY_PATH), "bytes")
print("smoke all pass:", A1_SUMMARY["smoke_all_pass"], "| wall:", round(A1_SUMMARY["wall_seconds"], 1), "s")
print("artifacts:", sorted(os_a1.listdir(A1_PATHS["art"])))'''

async with cm.get_context() as ctx:
    a = ctx.create_cell('mo_a1.md(r"""' + chr(10) + "## A1.8 Artifact manifest" + chr(10) + '""")', hide_code=False)
    b = ctx.create_cell(SUMM, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
