import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
## A1.2 Launch (detached, idempotent)

The sweep runs in a kernel-side daemon `threading.Thread`. This matters: a
client disconnect - or any `ctx.run_cell` while a notebook cell is executing -
interrupts a running cell, which killed a full sweep on a sibling box. The
worker writes every artifact to `/marimo/a1` as it goes, so progress is polled
by READING FILES, and re-running these cells reloads from cache in seconds
instead of recomputing.

Per checkpoint the worker does: load -> (base only: fit direction + 5 smoke
tests) -> intact/ablated readout on 200 held-out harmful -> 20 random-direction
controls -> 40 greedy generations at 200 new tokens, intact and ablated ->
write JSON -> free the model and purge that repo's HF cache (~15 GB each).
"""
)'''

LAUNCH = '''A1_LAUNCH = a1p.a1_launch(
    n_train=200,
    n_held=200,
    n_rand=20,
    n_gen=40,
    gen_new_tokens=200,
    batch_size=16,
    gen_batch_size=8,
)
print(A1_LAUNCH)
print("poll:", A1_PATHS["status"], "and", A1_PATHS["log"])'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(LAUNCH, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
