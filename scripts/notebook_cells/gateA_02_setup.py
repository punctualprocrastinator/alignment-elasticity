GA_DIR = "/marimo/gateA"
if GA_DIR not in sys.path:
    sys.path.insert(0, GA_DIR)

import pipeline as pl
import gate_a as ga
import torch

GA_ENV = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "seed": ga.SEED,
    "split_seed": ga.SPLIT_SEED,
    "fit_layer": ga.FIT_LAYER,
    "steer_layer": ga.STEER_LAYER,
    "c_grid": ga.C_GRID,
    "c_grid_null": ga.C_GRID_NULL,
    "checkpoints": [(lab, repo, br, sha[:10]) for lab, repo, br, sha in ga.CKPTS],
}
GA_ENV
