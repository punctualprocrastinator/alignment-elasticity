import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
# A1 - POWERING THE CAUSAL CLAIMS

Day 3. Yesterday's causal ordering (base 100%, RL-Zero-Code 86%, RL-Zero-IF 87%,
Instruct 8%) rests on n=6-8 prompts and no null model. That will not survive
review. This section re-runs the same intervention with:

* **n = 200 held-out harmful prompts** per checkpoint (fit split is disjoint),
* **bootstrapped 95% CIs** (>=1000 resamples) on every number, including the
  effect-as-percent-of-base ratio (paired bootstrap over prompts),
* a **20 random-direction null band** ablated identically, giving each
  checkpoint a z-score against its own null,
* a **behavioural cross-check**: 40 greedy generations at 200 new tokens.

**Design.** One direction only: the BASE model's layer-20 refusal direction
(normalized difference-in-means, last token, train split). That same vector is
ablated in every checkpoint, so any difference is elasticity of the *base-era*
direction, not a re-fit per model.

**Readout (judge-free).** At the first generated position,
`mean logprob(refusal-onset tokens) - mean logprob(compliance-onset tokens)`.
Causal effect `delta = intact - ablated`; positive means ablation removed
refusal-leaning probability mass.

**Format policy.** Neutral scaffold `User: {prompt}` + newline + `Assistant:`
applied identically to every checkpoint - no chat templates, so the interface
is matched and Instruct is not advantaged by its own template.

All code lives in `/marimo/pipeline.py` (mirrored to the user's machine), not in
cell bodies, so it survives the sandbox.
"""
)'''

SETUP = '''# A1 setup. This notebook already has an E5/E7/E8 import cell that owns the
# short names (mo, np, pd, plt, torch, os, json, ...), and marimo allows exactly
# one owning cell per public name - hence the a1_ / _a1 aliases here.
import sys

if "/marimo" not in sys.path:
    sys.path.insert(0, "/marimo")

import importlib
import json as json_a1
import os as os_a1
import time as time_a1

import marimo as mo_a1
import numpy as np_a1
import pandas as pd_a1
import matplotlib
import matplotlib.pyplot as plt_a1

import pipeline as a1p

a1p = importlib.reload(a1p)
matplotlib.use("Agg")

A1_PATHS = a1p.a1_paths()
A1_FIGDIR = a1p.ensure_dir("/marimo/figures")

print("pipeline:", a1p.__file__)
print("fit layer:", a1p.FIT_LAYER, "| probe layers:", a1p.PROBE_LAYERS)
print("MAX_LEN:", a1p.MAX_LEN, "(left truncation) | seed:", a1p.SEED)
print("checkpoints:")
for _r, _v, _l in a1p.CKPTS_A1:
    print("   ", _l, "->", _r, "@", _v)
print("artifacts:", A1_PATHS["art"], "| figures:", A1_FIGDIR)'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(SETUP, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
