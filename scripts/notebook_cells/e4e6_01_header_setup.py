import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
# E4 / E6 CLEAN - canonical re-runs at protocol v1

Two day-2 results re-run properly, per `protocol.md`.

**E4 (behaviour across the flow).** 200 HarmBench *standard* text behaviours
(day 2 used 60), greedy, **512 new tokens**. Day 2 generated only 44 tokens,
which truncated inside `<think>` and made refusal classification meaningless -
so this run also reports the fraction of `<think>` blocks that fail to close.
Judge: `cais/HarmBench-Llama-2-13b-cls`.

**E6/A5 (monitoring).** Day 2 was underpowered: 5 of 8 monitored columns had
1-4 harmful positives and one had zero, because post-SFT models almost never
comply. Fixed by *deliberate elicitation* - each checkpoint generates under
three conditions (intact / refusal-direction-ablated / prefill) and the pooled
judge-confirmed harmful outputs form the positive class.

**Protocol bindings.** SEED=42; SHA-1 split fingerprints in every result file;
>=1000 bootstrap resamples; AUROC headline with accuracy as a diagnostic only;
paired bootstrap for comparisons; exact HF revisions pinned and verified
pairwise-distinct by safetensors sha256; detached thread taking config as
arguments.

**Stated deviations.** (1) The monitor prompt wraps a full model response in a
3-shot frame and cannot fit `MAX_LEN=384`; monitors use `MON_MAX_LEN=1536`,
left truncation, truncation fraction logged. (2) The AdvBench pool used to fit
each checkpoint's own refusal direction is reused from A1 (shuffle seed 0); it
is an elicitation aid, never a reported quantity, and its fingerprint is
recorded.
"""
)'''

SETUP = '''import pipeline_e4e6 as e46

e46 = importlib.reload(e46)

E46_E4DIR = a1p.ensure_dir(e46.E4_DIR)
E46_E6DIR = a1p.ensure_dir(e46.E6_DIR)
E46_HB = e46.load_harmbench(os_a1.path.join(E46_E4DIR, "harmbench.json"))

print("module:", e46.__file__)
print("seed:", e46.SEED_V1, "| gen tokens:", e46.GEN_MAX_NEW, "| monitor max_len:", e46.MON_MAX_LEN)
print("behaviours:", E46_HB["n"], "| category:", E46_HB["functional_category"])
print("split fingerprint (SHA-1):", E46_HB["fingerprint"])
print("flow:")
for _r, _v, _l, _o in e46.CKPTS_E4:
    print("   %d %-10s %s @ %s" % (_o, _l, _r, _v))
print("monitors:", e46.MONITOR_LABELS)
print("judge:", e46.JUDGE_REPO, "with tokenizer", e46.JUDGE_TOKENIZER)'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(SETUP, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
