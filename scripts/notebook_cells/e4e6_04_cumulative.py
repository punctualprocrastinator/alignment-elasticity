import marimo._code_mode as cm

CUM = '''# Adjacent transitions can each be flat while the cumulative change is real,
# so the "collapse complete by step 1000" claim needs the cumulative contrasts too.
_cum = [("base", "sft-1k"), ("base", "sft-43k"), ("base", "rlvr-1375"),
        ("sft-1k", "sft-43k"), ("sft-1k", "rlvr-1375"), ("sft-43k", "rlvr-1375"),
        ("rlz-math", "base")]
_rows = []
for _a, _b in _cum:
    _d, _lo, _hi = e46.paired_rate_diff_ci(E4_JUD[_b]["intact_harmful"],
                                           E4_JUD[_a]["intact_harmful"],
                                           n_boot=E4_NBOOT, seed=42)
    _rows.append({"contrast": "%s -> %s" % (_a, _b), "d_asr": _d, "ci_lo": _lo, "ci_hi": _hi,
                  "significant": "YES" if (_lo > 0 or _hi < 0) else "no"})
E4_CUMUL = pd_a1.DataFrame(_rows)

_base = E4_TABLE.set_index("model").loc["base", "asr"]
_s1 = E4_TABLE.set_index("model").loc["sft-1k", "asr"]
_last = E4_TABLE.set_index("model").loc["rlvr-1375", "asr"]
E4_COLLAPSE_FRAC = float((_base - _s1) / (_base - _last))

print("cumulative contrasts, paired bootstrap")
print(E4_CUMUL.round(4).to_string(index=False))
print()
print("fraction of the total base->rlvr-1375 ASR collapse achieved by SFT step 1000: %.1f%%"
      % (100 * E4_COLLAPSE_FRAC))
print("day-2 claim was 100%% (ASR 0.000 at step 1000); measured here: %.3f" % _s1)'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(CUM, hide_code=False)
    ctx.run_cell(a)
    print("cell:", a)
