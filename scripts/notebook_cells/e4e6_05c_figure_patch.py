import marimo._code_mode as cm

async with cm.get_context() as ctx:
    src = ctx.cells["xLLj"].code

reps = [
    ('_a.set_ylim(-0.05, 1.0)', '_a.set_ylim(-0.05, 1.34)'),
    ('_a.legend(fontsize=9.5, loc="center", framealpha=0.92)',
     '_a.legend(fontsize=9.5, loc="upper center", ncol=3, framealpha=0.92)'),
]
new = src
for a, b in reps:
    assert new.count(a) == 1, (a[:50], new.count(a))
    new = new.replace(a, b)

async with cm.get_context() as ctx:
    ctx.edit_cell("xLLj", new)
    ctx.run_cell("xLLj")
print("edited")
