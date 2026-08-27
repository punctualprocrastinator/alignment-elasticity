import json
import os
import sys
import time

import marimo as mo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

mo.md(
    r"""
    # Gate A - does the steering collapse survive boundary-matched dosing?

    **A1 found** that ablating the *base* model's layer-20 refusal direction flips
    0.670 of base prompts to compliance but only 0.350 of Instruct's - while
    producing a *larger* raw logit displacement on Instruct (6.81 vs 4.37).
    Instruct simply starts further inside refusal (unsteered gap +7.85 vs +0.85).

    **Gate A asks:** is that collapse a real property of the post-trained model, or
    an artifact of baseline distance to the behavioural decision boundary?

    Steering uses the persona-vector parameterisation of arXiv 2605.13329 verbatim,
    `h_l <- h_l + c * mu_l * v_hat`, so the input-side comparison is like-for-like.
    The new axis is the **achieved displacement** of the refusal logit gap, which is
    boundary-relative rather than input-relative.

    - **Outcome A (negative methods result):** crossing-rate curves collapse onto one
      another once plotted against achieved displacement -> the decay was baseline
      distance all along.
    - **Outcome B (real residual effect):** Instruct still crosses less at *equal
      achieved displacement*.
    """
)
