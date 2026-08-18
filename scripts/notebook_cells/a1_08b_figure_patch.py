import marimo._code_mode as cm

async with cm.get_context() as ctx:
    src = ctx.cells["EBee"].code

reps = [
    ('figsize=(16.0, 5.6)', 'figsize=(17.2, 5.8)'),
    ('_ax.set_xticklabels([A1_NICE[m] for m in A1_ORDER], fontsize=11)',
     '_ax.set_xticklabels([A1_NICE[m] for m in A1_ORDER], fontsize=10.5)' + chr(10) + '    _ax.set_xlim(-0.55, 3.55)'),
]
new = src
for a, b in reps:
    assert new.count(a) == 1, (a, new.count(a))
    new = new.replace(a, b)

async with cm.get_context() as ctx:
    ctx.edit_cell("EBee", new)
    ctx.run_cell("EBee")
print("edited")
