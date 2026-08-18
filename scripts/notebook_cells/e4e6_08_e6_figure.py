import marimo._code_mode as cm

F6 = '''FIG_E6C, _ax6 = plt_a1.subplots(1, 3, figsize=(17.5, 5.8), facecolor="white")
_gn = [E4_NICE[g].replace(chr(10), " ") for g in E6_GENS]

_a = _ax6[0]
_mat = (E6_GRID.pivot(index="monitor", columns="generator", values="auroc")
        .reindex(index=E6_MONS, columns=E6_GENS).values)
_im = _a.imshow(_mat, cmap="viridis", vmin=0.75, vmax=1.0, aspect="auto")
for _i in range(_mat.shape[0]):
    for _j in range(_mat.shape[1]):
        _a.text(_j, _i, "%.2f" % _mat[_i, _j], ha="center", va="center",
                fontsize=9.5, color="white" if _mat[_i, _j] < 0.93 else "black")
_a.set_xticks(range(len(E6_GENS)))
_a.set_xticklabels(_gn, rotation=45, ha="right", fontsize=9.5)
_a.set_yticks(range(len(E6_MONS)))
_a.set_yticklabels(E6_MONS, fontsize=10.5)
_a.set_title("A. Monitor AUROC" + chr(10) + "(rows monitor, cols monitored)",
             fontsize=13, fontweight="bold")
_a.set_xlabel("monitored checkpoint", fontsize=11)
FIG_E6C.colorbar(_im, ax=_a, fraction=0.046)

_a = _ax6[1]
_xm = np_a1.arange(len(E6_MONS))
_w = 0.27
for _k, (_c, _col, _lab) in enumerate([("auroc", "#1f77b4", "AUROC (benign negatives)"),
                                       ("auroc_hard", "#ff7f0e", "AUROC (refusal negatives)"),
                                       ("catch1", "#2ca02c", "catch @ 1% FPR")]):
    _mu = [E6_GRID[E6_GRID["monitor"] == m][_c].mean() for m in E6_MONS]
    _lo = [E6_GRID[E6_GRID["monitor"] == m][_c + "_lo"].mean() for m in E6_MONS]
    _hi = [E6_GRID[E6_GRID["monitor"] == m][_c + "_hi"].mean() for m in E6_MONS]
    _a.bar(_xm + (_k - 1) * _w, _mu, _w, color=_col, edgecolor="black", lw=0.6, label=_lab,
           yerr=[np_a1.array(_mu) - np_a1.array(_lo), np_a1.array(_hi) - np_a1.array(_mu)],
           capsize=4, error_kw={"lw": 1.5})
_a.axhline(0.5, color="#888888", ls=":", lw=1.4)
_a.set_xticks(_xm)
_a.set_xticklabels(E6_MONS, fontsize=10.5)
_a.set_ylim(0, 1.05)
_a.set_ylabel("score (mean over 8 monitored checkpoints)", fontsize=11)
_a.set_title("B. Monitor capability is flat across the flow" + chr(10)
             + "(except on hard refusal negatives)", fontsize=13, fontweight="bold")
_a.legend(fontsize=9.5, loc="lower right", framealpha=0.92)

_a = _ax6[2]
_sc = _a.scatter(E6_REG["distance"], E6_REG["catch5"], c=E6_REG["difficulty"],
                 cmap="coolwarm_r", s=115, edgecolor="black", lw=0.7, zorder=3)
_cb = FIG_E6C.colorbar(_sc, ax=_a, fraction=0.046)
_cb.set_label("monitored-checkpoint legibility" + chr(10) + "(leave-one-out AUROC)", fontsize=9.5)
_z = np_a1.polyfit(E6_REG["distance"], E6_REG["catch5"], 1)
_xx = np_a1.linspace(-0.2, 7.2, 20)
_a.plot(_xx, np_a1.polyval(_z, _xx), color="#c0392b", ls="--", lw=1.8,
        label="raw fit  r = %.2f" % np_a1.corrcoef(E6_REG["distance"], E6_REG["catch5"])[0, 1])
_a.set_xlabel("training distance (flow steps between monitor and monitored)", fontsize=11)
_a.set_ylabel("catch rate @ 5% FPR", fontsize=11)
_a.set_title("C. H6: distance coefficient" + chr(10)
             + "%.4f [%.4f, %.4f] - CI spans 0"
             % (E6_H6.iloc[3]["estimate"], E6_H6.iloc[3]["ci_lo"], E6_H6.iloc[3]["ci_hi"]),
             fontsize=13, fontweight="bold")
_a.legend(fontsize=10, loc="lower left", framealpha=0.92)
_a.grid(alpha=0.25)
_a.set_axisbelow(True)

FIG_E6C.suptitle("E6 clean: 200 judge-confirmed harmful positives per monitored checkpoint "
                 "(elicited: intact / ablated / prefill)", fontsize=14, fontweight="bold")
FIG_E6C.tight_layout(rect=[0, 0, 1, 0.93])
E6_FIG_PATH = os_a1.path.join(A1_FIGDIR, "fig_e6_clean_monitoring.png")
FIG_E6C.savefig(E6_FIG_PATH, dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", E6_FIG_PATH, os_a1.path.getsize(E6_FIG_PATH))
FIG_E6C'''

async with cm.get_context() as ctx:
    a = ctx.create_cell('mo_a1.md(r"""' + chr(10) + "### E6.3 Figure" + chr(10) + '""")', hide_code=False)
    b = ctx.create_cell(F6, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
