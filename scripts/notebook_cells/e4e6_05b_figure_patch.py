import marimo._code_mode as cm

async with cm.get_context() as ctx:
    src = ctx.cells["xLLj"].code

reps = [
    ('_a.text(0.12, _t.loc["rlz-math", "asr"] + 0.02,',
     '_a.text(0.05, _t.loc["rlz-math", "asr"] + 0.03,'),
    ('% _t.loc["rlz-math", "asr"], fontsize=10, color="#27ae60")',
     '% _t.loc["rlz-math", "asr"], fontsize=9.5, color="#27ae60")'),
    ('_a.legend(fontsize=10, loc="center right", framealpha=0.92)',
     '_a.legend(fontsize=9.5, loc="center", framealpha=0.92)'),
    ('_a.set_ylim(0, 1.05)' + chr(10) + '_a.set_title("B. Elicited',
     '_a.set_ylim(0, 1.30)' + chr(10) + '_a.set_title("B. Elicited'),
    ('_a.legend(fontsize=10, loc="upper left", framealpha=0.92)',
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
