import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
### A1.4c Readout provenance (exact token ids)

The readout is only interpretable if the onset tokens are pinned. All four
checkpoints share the OLMo 3 tokenizer, so the ids are identical everywhere -
which the table below verifies rather than assumes.
"""
)'''

TOKS = '''_rows = []
for _repo, _rev, _lab in a1p.CKPTS_A1:
    _r = A1_RESULTS[_lab]
    for _role, _key in [("refusal", "refusal"), ("compliance", "comply")]:
        for _s, _i, _sg, _dc in zip(
            _r[_key + "_strings"], _r[_key + "_ids"],
            _r[_key + "_single_token"], _r[_key + "_decoded"],
        ):
            _rows.append({"model": _lab, "role": _role, "string": repr(_s),
                          "token_id": _i, "single_token": _sg, "decoded": repr(_dc)})
A1_TOKENS = pd_a1.DataFrame(_rows)

A1_TOKENS_UNIQUE = A1_TOKENS.drop(columns=["model"]).drop_duplicates()
print("identical token ids across all", len(a1p.CKPTS_A1), "checkpoints:",
      len(A1_TOKENS_UNIQUE) == len(a1p.REFUSAL_STRS) + len(a1p.COMPLY_STRS))
print(A1_TOKENS_UNIQUE.to_string(index=False))
print()
print("pad token:", A1_RESULTS["base"]["pad_token"], A1_RESULTS["base"]["pad_token_id"],
      "| MAX_LEN:", A1_RESULTS["base"]["max_len"], "| truncation side: left | padding: right")
print("held-out truncation:", A1_RESULTS["base"]["held_truncation"]["n_at_or_over_max"],
      "/", A1_RESULTS["base"]["held_truncation"]["n"],
      "(len mean %.1f, max %d)" % (A1_RESULTS["base"]["held_truncation"]["len_mean"],
                                   A1_RESULTS["base"]["held_truncation"]["len_max"]))'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(TOKS, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
