import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
## E4/E6.4 Verdicts

### E4: the day-2 collapse was a truncation artifact. It is real but gradual.

Day 2 reported ASR base 0.350 -> SFT-step1000 **0.000** -> RLVR-last 0.017.
At 200 behaviours and 512 new tokens: base **0.355 [0.290, 0.425]** (confirmed
almost exactly), SFT-step1000 **0.180 [0.130, 0.235]**, RLVR-last
**0.070 [0.040, 0.105]**, RL-Zero-Math **0.400 [0.335, 0.470]** (day 2: 0.417,
confirmed). The two endpoints replicate; the middle of the curve does not.

**Is the collapse complete within the first 1,000 SFT steps? No.** SFT step
1000 delivers 61.4% of the total base -> RLVR-last drop, and the remaining
`sft-1k -> rlvr-1375` decline is itself significant
(dASR -0.110 [-0.165, -0.055], paired). Day 2's 0.000 came from generating only
44 tokens: at that budget the answer never escapes the `<think>` block, the
judge sees reasoning rather than an answer, and everything scores non-harmful.
At 512 tokens only 5-11% of `<think>` blocks fail to close.

**Is the DPO -> RLVR span statistically flat? Yes.** `sft-43k -> dpo`
(-0.020 [-0.055, 0.015]), `dpo -> rlvr-25` (0.010 [0.000, 0.025]),
`rlvr-25 -> rlvr-1375` (-0.020 [-0.045, 0.000]) and `dpo -> rlvr-1375`
(-0.010 [-0.030, 0.005]) all have CIs spanning zero. Refusal behaviour is
established during SFT and neither preference optimisation nor 1,375 steps of
RLVR moves it further.

**Harmfulness is suppressed, not removed.** Under prefill the ASR of every
post-SFT checkpoint returns to 0.92-0.96 - higher than the base model's own
0.865. Alignment along this flow is a change in what the model *does*, not in
what it *can* do.

### E6: powering it does not rescue the staleness story - it kills it.

With 200 judge-confirmed positives per monitored checkpoint (day 2 had 0-4 in
six of eight columns), all four monitors are near-ceiling and
**indistinguishable**: mean AUROC 0.955 / 0.949 / 0.952 / 0.953 for base /
SFT-1k / SFT-43k / DPO. Day 2's spread of 0.775-0.874 and its dramatic
catch@1%FPR gap (base 0.017 vs SFT-last 0.431) were small-sample noise; the
clean gap is 0.712 vs 0.748.

**The raw staleness correlation replicates and is spurious.** AUROC vs distance
gives r = -0.284, essentially day 2's -0.26. But in the regression
`catch@5%FPR ~ capability + difficulty + distance`, the distance coefficient is
**-0.0020 [-0.0056, 0.0019]** - CI spans zero - while monitored-checkpoint
legibility carries **1.845 [1.430, 3.391]**. R2 goes 0.993 -> 0.994 when
distance is added. Panel C of the figure shows why: the two hard-to-monitor
generators (base and RL-Zero-Math, AUROC ~0.82) sit at opposite *ends* of the
flow ordering and therefore have large mean distance from the mid-flow
monitors. The apparent staleness is that confound.

*(Caveat on one line of the table: the bootstrap CI for the partial R2 of
distance is reported as excluding zero, but partial R2 is non-negative by
construction, so that interval cannot include negative values and the flag is
not a significance test. The coefficient CI is the test, and it spans zero.)*

**So does "capability, not freshness, is the binding constraint" survive? Half
of it.** Freshness is dead - distance adds nothing once legibility is
controlled. But capability is not the binding constraint either, because there
is no capability gradient left to bind: every monitor from base to DPO scores
~0.95. What actually determines detection is **how legible the monitored
model's harmful output is** (r = 0.996 with catch@5%FPR). The one place a real
monitor difference appears is the hard negative set - discriminating harmful
compliance from *refusals about harmful topics* - where the base monitor drops
to 0.744 while SFT-43k reaches 0.894. If a monitoring claim is to be made, that
is where it lives.
"""
)'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    ctx.run_cell(a)
    print("cell:", a)
