# Box A (E1/E2/E3) setup: restore-mode imports from /marimo/scripts
import sys, os, json, time
import numpy as np
import torch
if "/marimo/scripts" not in sys.path:
    sys.path.insert(0, "/marimo/scripts")
import pipeline as P
import gate_a as ga
import gate_a_analysis as gaa
import e1_causal as e1
print("imports OK | torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("gate_a ART", ga.ART, "| E1 ART", e1.E1_ART)
