import marimo._code_mode as cm

F4 = '''E4_FLOW = ["base", "sft-1k", "sft-20k", "sft-43k", "dpo", "rlvr-25", "rlvr-1375"]
E4_NICE = {"base": "base", "sft-1k": "SFT" + chr(10) + "1k", "sft-20k": "SFT" + chr(10) + "20k",
           "sft-43k": "SFT" + chr(10) + "43k", "dpo": "DPO", "rlvr-25": "RLVR" + chr(10) + "25",
           "rlvr-1375": "RLVR" + chr(10) + "1375", "rlz-math": "RL-Zero" + chr(10) + "Math 1900"}
_t = E4_TABLE.set_index("model")

FIG_E4C, _ax4 = plt_a1.subplots(1, 3, figsize=(17.0, 5.6), facecolor="white")
_xi = np_a1.arange(len(E4_FLOW))

_a = _ax4[0]
_y = [_t.loc[m, "asr"] for m in E4_FLOW]
_el = [_t.loc[m, "asr"] - _t.loc[m, "asr_lo"] for m in E4_FLOW]
_eh = [_t.loc[m, "asr_hi"] - _t.loc[m, "asr"] for m in E4_FLOW]
_a.errorbar(_xi, _y, yerr=[_el, _eh], fmt="o-", color="#c0392b", lw=2.2, ms=9,
            capsize=6, elinewidth=2.0, label="ASR (intact), n=200")
_yr = [_t.loc[m, "refusal"] for m in E4_FLOW]
_rl = [_t.loc[m, "refusal"] - _t.loc[m, "ref_lo"] for m in E4_FLOW]
_rh = [_t.loc[m, "ref_hi"] - _t.loc[m, "refusal"] for m in E4_FLOW]
_a.errorbar(_xi, _yr, yerr=[_rl, _rh], fmt="s-", color="#2471a3", lw=2.2, ms=9,
            capsize=6, elinewidth=2.0, label="refusal rate")
_a.scatter([0, 1, 6], [0.350, 0.000, 0.017], marker="x", s=110, color="#888888",
           zorder=5, label="day 2 (n=60, 44 tokens)")
_a.axhline(_t.loc["rlz-math", "asr"], color="#27ae60", ls=":", lw=2.0)
_a.text(0.12, _t.loc["rlz-math", "asr"] + 0.02, "RL-Zero-Math (off-flow): ASR %.3f"
        % _t.loc["rlz-math", "asr"], fontsize=10, color="#27ae60")
_a.set_ylim(-0.05, 1.0)
_a.set_title("A. Behaviour across the post-training flow", fontsize=13, fontweight="bold")
_a.set_ylabel("rate", fontsize=12)
_a.legend(fontsize=10, loc="center right", framealpha=0.92)

_a = _ax4[1]
_w = 0.27
for _k, (_c, _col, _lab) in enumerate([("asr", "#c0392b", "intact"),
                                       ("asr_ablated", "#e67e22", "refusal-dir ablated"),
                                       ("asr_prefill", "#8e44ad", "prefill")]):
    _a.bar(_xi + (_k - 1) * _w, [_t.loc[m, _c] for m in E4_FLOW], _w,
           color=_col, edgecolor="black", lw=0.6, label=_lab)
_a.set_ylim(0, 1.05)
_a.set_title("B. Elicited harmfulness stays available" + chr(10)
             + "(the E6 positive class comes from here)", fontsize=13, fontweight="bold")
_a.set_ylabel("ASR", fontsize=12)
_a.legend(fontsize=10, loc="upper left", framealpha=0.92)

_a = _ax4[2]
_tr = E4_TRANS.copy()
_yy = np_a1.arange(len(_tr))
_a.errorbar(_tr["d_asr"], _yy,
            xerr=[_tr["d_asr"] - _tr["ci_lo"], _tr["ci_hi"] - _tr["d_asr"]],
            fmt="o", color="#2c3e50", ms=8, capsize=5, elinewidth=2.0)
_a.axvline(0.0, color="#c0392b", ls="--", lw=1.6)
_a.set_yticks(_yy)
_a.set_yticklabels(_tr["transition"], fontsize=10)
_a.invert_yaxis()
_a.set_title("C. Per-transition change in ASR" + chr(10)
             + "(paired bootstrap; CI crossing 0 = flat)", fontsize=13, fontweight="bold")
_a.set_xlabel("delta ASR", fontsize=12)

for _a in _ax4[:2]:
    _a.set_xticks(_xi)
    _a.set_xticklabels([E4_NICE[m] for m in E4_FLOW], fontsize=10)
    _a.grid(axis="y", alpha=0.25)
    _a.set_axisbelow(True)
_ax4[2].grid(axis="x", alpha=0.25)
_ax4[2].set_axisbelow(True)

FIG_E4C.suptitle("E4 clean: 200 HarmBench standard behaviours, greedy, 512 new tokens, "
                 "HarmBench-Llama-2-13b-cls judge", fontsize=14, fontweight="bold")
FIG_E4C.tight_layout(rect=[0, 0, 1, 0.93])
E4_FIG_PATH = os_a1.path.join(A1_FIGDIR, "fig_e4_clean_behaviour.png")
FIG_E4C.savefig(E4_FIG_PATH, dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", E4_FIG_PATH, os_a1.path.getsize(E4_FIG_PATH))
FIG_E4C'''

async with cm.get_context() as ctx:
    a = ctx.create_cell('mo_a1.md(r"""' + chr(10) + "### E4.3 Figure" + chr(10) + '""")', hide_code=False)
    b = ctx.create_cell(F4, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
