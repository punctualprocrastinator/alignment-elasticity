import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
## A1.1 Data

`harmful` = AdvBench `harmful_behaviors.csv`, column `goal`.
`benign` = `tatsu-lab/alpaca` rows whose `input` field is empty.
Both shuffled with a fixed seed, then split: the first 200 of each are the FIT
split (used only to estimate the base direction), the next 200 harmful are the
HELD-OUT split every causal number is measured on.
"""
)'''

DATA = '''A1_DATA = a1p.load_prompts(n_train=200, n_held=200, cache_path=A1_PATHS["prompts"])

A1_HARM_FIT = A1_DATA["harm_train"]
A1_BEN_FIT = A1_DATA["ben_train"]
A1_HARM_HELD = A1_DATA["harm_held"]

print("pools:", A1_DATA["harm_pool"], "harmful /", A1_DATA["ben_pool"], "benign")
print("fit split:", len(A1_HARM_FIT), "harmful +", len(A1_BEN_FIT), "benign")
print("held-out harmful (all causal numbers):", len(A1_HARM_HELD))
print("fit/held overlap:", len(set(A1_HARM_FIT) & set(A1_HARM_HELD)))
print()
print("--- scaffolded harmful example ---")
print(a1p.fmt(A1_HARM_HELD[0]))
print("--- scaffolded benign example ---")
print(a1p.fmt(A1_BEN_FIT[0]))'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(DATA, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
