import marimo._code_mode as cm

async with cm.get_context() as ctx:
    src = ctx.cells["pbjG"].code

old = '''    _mu = [E6_GRID[E6_GRID["monitor"] == m][_c].mean() for m in E6_MONS]
    _lo = [E6_GRID[E6_GRID["monitor"] == m][_c + "_lo"].mean() for m in E6_MONS]
    _hi = [E6_GRID[E6_GRID["monitor"] == m][_c + "_hi"].mean() for m in E6_MONS]
    _a.bar(_xm + (_k - 1) * _w, _mu, _w, color=_col, edgecolor="black", lw=0.6, label=_lab,
           yerr=[np_a1.array(_mu) - np_a1.array(_lo), np_a1.array(_hi) - np_a1.array(_mu)],
           capsize=4, error_kw={"lw": 1.5})'''
new = '''    # Whiskers are the SPREAD across the 8 monitored checkpoints, not a CI:
    # averaging per-cell CI bounds would not be a valid interval for a mean.
    _mu = [E6_GRID[E6_GRID["monitor"] == m][_c].mean() for m in E6_MONS]
    _sd = [E6_GRID[E6_GRID["monitor"] == m][_c].std(ddof=1) for m in E6_MONS]
    _a.bar(_xm + (_k - 1) * _w, _mu, _w, color=_col, edgecolor="black", lw=0.6, label=_lab,
           yerr=_sd, capsize=4, error_kw={"lw": 1.5})'''

assert src.count(old) == 1
body = src.replace(old, new).replace(
    '_a.set_ylabel("score (mean over 8 monitored checkpoints)", fontsize=11)',
    '_a.set_ylabel("score (mean over 8 monitored;" + chr(10) + "whiskers = SD across them)", fontsize=11)')

async with cm.get_context() as ctx:
    ctx.edit_cell("pbjG", body)
    ctx.run_cell("pbjG")
print("edited")
