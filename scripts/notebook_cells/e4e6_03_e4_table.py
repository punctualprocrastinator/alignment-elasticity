import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
## E4.2 Behaviour across the flow (200 HarmBench standard behaviours)

ASR = fraction judged harmful by `HarmBench-Llama-2-13b-cls` on the **intact**
model (the shipped behaviour). Refusal rate = refusal-onset classifier on the
post-`</think>` answer. Both with 2000-resample bootstrap CIs; comparisons
between adjacent flow stages use a **paired** bootstrap over the shared 200
behaviours.
"""
)'''

E4T = '''E4_NBOOT = 2000
E4_LABELS = [c[2] for c in e46.CKPTS_E4]

E4_GEN = {l: a1p.read_json(os_a1.path.join(e46.E4_DIR, "gen_%s.json" % l)) for l in E4_LABELS}
E4_JUD = {l: a1p.read_json(os_a1.path.join(e46.E4_DIR, "judged_%s.json" % l)) for l in E4_LABELS}

_rows = []
for _repo, _rev, _lab, _order in e46.CKPTS_E4:
    _g, _j = E4_GEN[_lab], E4_JUD[_lab]
    _asr = np_a1.array(_j["intact_harmful"], dtype=float)
    _ref = np_a1.array(_g["intact_refusal"], dtype=float)
    _a, _alo, _ahi = e46.rate_ci(_asr, n_boot=E4_NBOOT, seed=42)
    _r, _rlo, _rhi = e46.rate_ci(_ref, n_boot=E4_NBOOT, seed=42)
    _rows.append({
        "order": _order, "model": _lab, "revision": _rev,
        "asr": _a, "asr_lo": _alo, "asr_hi": _ahi,
        "refusal": _r, "ref_lo": _rlo, "ref_hi": _rhi,
        "think_unclosed": _g["intact_think_unclosed_frac"],
        "think_blocks": _g["intact_think_blocks"],
        "asr_ablated": _j["ablated_asr"], "asr_prefill": _j["prefill_asr"],
    })
E4_TABLE = pd_a1.DataFrame(_rows).sort_values("order").reset_index(drop=True)

print("E4 - intact behaviour, n=200 HarmBench standard, 512 new tokens")
print(pd_a1.DataFrame({
    "model": E4_TABLE["model"],
    "rev": E4_TABLE["revision"],
    "ASR [95% CI]": ["%.3f [%.3f, %.3f]" % (r.asr, r.asr_lo, r.asr_hi) for r in E4_TABLE.itertuples()],
    "refusal [95% CI]": ["%.3f [%.3f, %.3f]" % (r.refusal, r.ref_lo, r.ref_hi) for r in E4_TABLE.itertuples()],
    "think unclosed": E4_TABLE["think_unclosed"].round(3),
    "ASR abl": E4_TABLE["asr_ablated"].round(3),
    "ASR prefill": E4_TABLE["asr_prefill"].round(3),
}).to_string(index=False))

_pairs = [("base","sft-1k"),("sft-1k","sft-20k"),("sft-20k","sft-43k"),
          ("sft-43k","dpo"),("dpo","rlvr-25"),("rlvr-25","rlvr-1375"),
          ("dpo","rlvr-1375"),("base","rlz-math")]
_rows = []
for _a, _b in _pairs:
    _d, _lo, _hi = e46.paired_rate_diff_ci(E4_JUD[_b]["intact_harmful"],
                                           E4_JUD[_a]["intact_harmful"],
                                           n_boot=E4_NBOOT, seed=42)
    _rows.append({"transition": "%s -> %s" % (_a, _b), "d_asr": _d,
                  "ci_lo": _lo, "ci_hi": _hi,
                  "flat": "FLAT (CI spans 0)" if (_lo <= 0 <= _hi) else "changed"})
E4_TRANS = pd_a1.DataFrame(_rows)
print()
print("paired bootstrap on the shared 200 behaviours")
print(E4_TRANS.round(4).to_string(index=False))'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(E4T, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
