import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
## A1.7 Verdict: the ordering survives, the METRIC that stated it does not

**All five smoke tests pass.** Hook liveness is the one that mattered: ablating
across `embed_tokens` + all 32 decoder layers moves the last-position logits by
up to 13.09, while hooking the embedding alone moves them by 0.41 - the hooks
are live and the embedding contribution is real but small on its own.
Truncation is 0/400 on the fit split and 0/200 held out (token length mean ~17,
max 35), so `MAX_LEN=384` is not binding on AdvBench goals; the left-truncation
policy is in force but costs nothing here.

**Every checkpoint is enormously separated from its own null.** The 20
random-unit-direction controls sit at a mean effect of -0.02 with sd ~0.05; the
base direction scores z = 86 to 164. There is no version of these results in
which the effect is a projection-magnitude artifact.

**The n=6 ordering does NOT replicate on raw effect size - it inverts.**
Yesterday: base 100%, RL-Zero 86-87%, Instruct 8%. At n=200 with paired
bootstrap CIs: base 100%, RL-Zero-Math 104.1% [103.8, 104.4], RL-Zero-Code
111.0% [110.1, 112.0], and **Instruct 156.1% [146.0, 167.1]** - the largest raw
effect of any checkpoint, not the smallest. The CIs separate every pair
(paired bootstrap, all six comparisons exclude 0), so this is not a power
problem; the n=6 estimate was simply wrong about the sign of the contrast.

**The dissociation is real, but it is an OUTCOME effect, not a displacement
effect.** Instruct starts at +7.85 on the readout, is pushed down 6.81, and
lands at +1.04 - still refusal-leaning. Base starts at +0.85, is pushed down
4.37, and lands at -3.51 - decisively compliance-leaning. Scored as *crossing
rate* (prompts flipped from refusal-leaning to compliance-leaning), the picture
snaps back into yesterday's shape and the CIs cleanly separate the arms:

| model | crossing rate (n=200) | behavioural refusal drop (n=40) |
|---|---|---|
| base | 0.670 [0.605, 0.735] | 93.5% of its own refusals |
| RL-Zero-Math | 0.695 [0.630, 0.760] | 93.5% |
| RL-Zero-Code | 0.655 [0.590, 0.720] | 96.7% |
| **Instruct** | **0.350 [0.285, 0.415]** | **15.0%** |

Instruct's crossing-rate CI does not overlap any of the other three. The
generation check is sharper still and lands almost exactly on yesterday's
numbers: 1.00 -> 0.85 refusal for Instruct versus 0.78 -> 0.05 for base and
RL-Zero-Math and 0.75 -> 0.03 for RL-Zero-Code.

**What to write in the paper.** The read/steer dissociation survives powering
and survives a null model - Instruct resists ablation of the base-era refusal
direction while both RL-Zero arms remain as ablatable as the base model. But
the claim must be stated on an outcome-scaled metric (crossing rate or
behavioural refusal), not on raw logit displacement, because Instruct's SFT+DPO
stage does not move the direction out of the residual stream - it *increases*
the model's dependence on refusal-onset logit mass so far that removing the
same amount of it no longer changes the decision. Reporting "Instruct = 8% of
base" would not survive review; reporting "Instruct = 156% of base raw
displacement but 52% of base crossing rate and 16% of base behavioural effect"
would, and it is a more interesting claim.

**RL-Zero does not separate from base.** RL-Zero-Math (104.1%) and RL-Zero-Code
(111.0%) have CIs that formally exclude 100%, but the effect is a few percent
and both arms are behaviourally indistinguishable from base (93.5% / 96.7% vs
93.5%). Do not claim RL-Zero erodes the refusal direction; claim it leaves it
intact.
"""
)'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    ctx.run_cell(a)
    print("cell:", a)
