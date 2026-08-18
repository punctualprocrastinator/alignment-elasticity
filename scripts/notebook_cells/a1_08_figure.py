import marimo._code_mode as cm

FIG = '''A1_COLORS = {
    "base": "#444444",
    "instruct": "#d62728",
    "rlz-math": "#1f77b4",
    "rlz-code": "#2ca02c",
}
A1_ORDER = ["base", "instruct", "rlz-math", "rlz-code"]
A1_NICE = {
    "base": "base" + chr(10) + "Olmo-3-7B",
    "instruct": "Instruct" + chr(10) + "(final)",
    "rlz-math": "RL-Zero-Math" + chr(10) + "step 1900",
    "rlz-code": "RL-Zero-Code" + chr(10) + "step 2900",
}

FIG_A1, A1_AX = plt_a1.subplots(1, 3, figsize=(16.0, 5.6), facecolor="white")
_x = np_a1.arange(len(A1_ORDER))
_ci = A1_CAUSAL.set_index("model")
_cr = A1_CROSS.set_index("model")
_gn = A1_GEN_DF.set_index("model")
_cols = [A1_COLORS[m] for m in A1_ORDER]

_ax = A1_AX[0]
_nlo = float(_ci["null_lo"].min())
_nhi = float(_ci["null_hi"].max())
_ax.axhspan(_nlo, _nhi, color="#bbbbbb", alpha=0.55, zorder=0)
_ax.axhline(0.0, color="#888888", lw=0.8, zorder=1)
for _i, _m in enumerate(A1_ORDER):
    _r = _ci.loc[_m]
    _ax.errorbar(_i, _r["delta"], yerr=[[_r["delta"] - _r["ci_lo"]], [_r["ci_hi"] - _r["delta"]]],
                 fmt="o", ms=11, color=_cols[_i], ecolor=_cols[_i], elinewidth=2.6, capsize=7, zorder=3)
    _ax.annotate("z=%.0f" % _r["z_vs_null"], (_i, _r["ci_hi"]), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=11, color=_cols[_i], fontweight="bold")
_ax.set_ylim(-0.6, 8.2)
_ax.set_title("A. Raw causal effect of ablating the" + chr(10) + "BASE layer-20 refusal direction",
              fontsize=13, fontweight="bold")
_ax.set_ylabel("delta = intact - ablated  (logprob units)", fontsize=12)
_ax.text(0.02, 0.055, "grey band = 20 random unit directions (null)", transform=_ax.transAxes,
         fontsize=10.5, color="#555555")

_ax = A1_AX[1]
for _i, _m in enumerate(A1_ORDER):
    _r = _ci.loc[_m]
    _ax.errorbar(_i, _r["pct_of_base"], yerr=[[_r["pct_of_base"] - _r["pct_lo"]], [_r["pct_hi"] - _r["pct_of_base"]]],
                 fmt="s", ms=10, color=_cols[_i], ecolor=_cols[_i], elinewidth=2.6, capsize=7, zorder=3)
_ax.axhline(100.0, color="#444444", ls="--", lw=1.2)
_ax.scatter([0, 1, 2, 3], [100, 8, 87, 86], marker="x", s=95, color="#999999", zorder=2,
            label="yesterday, n=6-8, no controls")
_ax.set_ylim(-15, 185)
_ax.set_title("B. Effect as % of the base model's" + chr(10) + "own effect (paired bootstrap)",
              fontsize=13, fontweight="bold")
_ax.set_ylabel("% of base delta", fontsize=12)
_ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

_ax = A1_AX[2]
_w = 0.36
_ax.bar(_x - _w / 2, [_cr.loc[m, "crossing_rate"] for m in A1_ORDER], _w, color=_cols,
        edgecolor="black", lw=0.7,
        yerr=[[_cr.loc[m, "crossing_rate"] - _cr.loc[m, "cross_lo"] for m in A1_ORDER],
              [_cr.loc[m, "cross_hi"] - _cr.loc[m, "crossing_rate"] for m in A1_ORDER]],
        capsize=5, error_kw={"lw": 1.8}, label="logit crossing rate (n=200)")
_ax.bar(_x + _w / 2, [_gn.loc[m, "drop"] for m in A1_ORDER], _w, color=_cols, alpha=0.45,
        edgecolor="black", lw=0.7, hatch="//",
        yerr=[[_gn.loc[m, "drop"] - _gn.loc[m, "drop_lo"] for m in A1_ORDER],
              [_gn.loc[m, "drop_hi"] - _gn.loc[m, "drop"] for m in A1_ORDER]],
        capsize=5, error_kw={"lw": 1.8}, label="behavioural refusal drop (n=40)")
_ax.set_ylim(0, 1.02)
_ax.set_title("C. Outcome-scaled effect:" + chr(10) + "the dissociation lives here",
              fontsize=13, fontweight="bold")
_ax.set_ylabel("fraction of prompts flipped", fontsize=12)
_ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

for _ax in A1_AX:
    _ax.set_xticks(_x)
    _ax.set_xticklabels([A1_NICE[m] for m in A1_ORDER], fontsize=11)
    _ax.tick_params(axis="y", labelsize=11)
    _ax.grid(axis="y", alpha=0.25)
    _ax.set_axisbelow(True)

FIG_A1.suptitle(
    "A1: powering the causal claim - base-model refusal direction ablated in 4 OLMo 3 checkpoints",
    fontsize=15, fontweight="bold",
)
FIG_A1.tight_layout(rect=[0, 0, 1, 0.94])
A1_FIG_PATH = os_a1.path.join(A1_FIGDIR, "fig_a1_causal_power.png")
FIG_A1.savefig(A1_FIG_PATH, dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", A1_FIG_PATH, os_a1.path.getsize(A1_FIG_PATH), "bytes")
FIG_A1'''

async with cm.get_context() as ctx:
    a = ctx.create_cell('mo_a1.md(r"""' + chr(10) + "## A1.6 Figure" + chr(10) + '""")', hide_code=False)
    b = ctx.create_cell(FIG, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
