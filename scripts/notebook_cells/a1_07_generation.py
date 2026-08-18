import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
## A1.5 Behavioural cross-check (generations)

40 held-out harmful prompts, greedy decoding, 200 new tokens so nothing is
truncated inside a `<think>` block; a leading reasoning block is stripped before
classification. Refusal-onset classifier = substring match against a fixed
phrase list over the opening 600 characters of the answer. Same base-model
layer-20 direction ablated, same hooks (embedding + all 32 decoder layers).
"""
)'''

GEN = '''A1_GEN = {_lab: a1p.read_json(A1_PATHS["gen"](_lab)) for _r, _v, _lab in a1p.CKPTS_A1}

_rows = []
for _repo, _rev, _lab in a1p.CKPTS_A1:
    _g = A1_GEN[_lab]
    _i = np_a1.array(_g["intact_refusal"], dtype=float)
    _a = np_a1.array(_g["ablated_refusal"], dtype=float)
    _di, _dlo, _dhi = a1p.bootstrap_ci(_i - _a, n_boot=A1_NBOOT)
    _rows.append(
        {
            "model": _lab,
            "n": _g["n"],
            "intact_refusal": float(_i.mean()),
            "ablated_refusal": float(_a.mean()),
            "drop": _di,
            "drop_lo": _dlo,
            "drop_hi": _dhi,
            "rel_drop": float(_di / max(1e-9, _i.mean())),
        }
    )
A1_GEN_DF = pd_a1.DataFrame(_rows)

print(
    pd_a1.DataFrame(
        {
            "model": A1_GEN_DF["model"],
            "n": A1_GEN_DF["n"],
            "refusal intact": A1_GEN_DF["intact_refusal"].round(3),
            "refusal ablated": A1_GEN_DF["ablated_refusal"].round(3),
            "drop [95% CI]": [
                "%.3f [%.3f, %.3f]" % (r.drop, r.drop_lo, r.drop_hi)
                for r in A1_GEN_DF.itertuples()
            ],
            "% of own refusals removed": (100 * A1_GEN_DF["rel_drop"]).round(1),
        }
    ).to_string(index=False)
)
print()
print("--- sample ablated continuation, base ---")
print(A1_GEN["base"]["ablated_text"][0][:220].replace(chr(10), " "))
print("--- sample ablated continuation, instruct ---")
print(A1_GEN["instruct"]["ablated_text"][0][:220].replace(chr(10), " "))'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(GEN, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
