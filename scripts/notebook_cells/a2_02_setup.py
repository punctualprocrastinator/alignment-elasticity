# A2 setup: configuration, contrast pools, and the exact splits used by the
# published linear sweep. Deliberately self-contained (this session's kernel is
# fresh and the section 1-8 cells above are stale; re-running them would
# re-download ~15 GB checkpoints for no reason). All activations A2 needs are
# already cached on disk from the section-8 sweep.
import os as _os
import re as _re
import numpy as _np
import pandas as _pd
from sklearn.model_selection import train_test_split as _tts

A2_DIR = "/marimo/a2"
A2_ACTS_DIR = "/marimo/e1/acts"
A2_E1_DIR = "/marimo/e1"
A2_SEED = 42  # the seed of the published sweep (section 1 of this
# notebook). NOT pipeline.py's SEED=20260805, which belongs to the A1
# ablation experiment: the fit/eval split is derived from this value, so
# the wrong one silently shifts every AUROC by ~0.01-0.02 and destroys
# comparability with the published formation curve.
A2_LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]
A2_MLP_SEEDS = [0, 1, 2]
A2_WG_PATH = "/marimo/wildguard_contrast.parquet"
_os.makedirs(A2_DIR, exist_ok=True)

# Contrast set: WildGuardMix, 4 pools x 500, deduped on `prompt`, fixed seed.
# Built in section 7 and cached; the gated dataset is not re-downloaded here.
A2_WG = _pd.read_parquet(A2_WG_PATH)
A2_LABELS = A2_WG["label"].to_numpy().astype(int)
_van = ~A2_WG["adversarial"].to_numpy()
_adv = A2_WG["adversarial"].to_numpy()

# Splits reproduce the section-7c/8 protocol byte for byte.
A2_VAN_IDX = _np.where(_van)[0]
A2_ADV_IDX = _np.where(_adv)[0]
A2_VAN_TR, A2_VAN_TE = _tts(
    A2_VAN_IDX, test_size=0.2, stratify=A2_LABELS[A2_VAN_IDX], random_state=A2_SEED
)

# Checkpoints: every revision with cached scaffold activations, ordered by step.
def _a2_step(rev):
    _m = _re.search(r"step(\d+)", rev)
    return int(_m.group(1)) if _m else -1

A2_REVS = sorted(
    [
        _f[len("acts_") : -len("_scaffold.pt")]
        for _f in _os.listdir(A2_ACTS_DIR)
        if _f.startswith("acts_") and _f.endswith("_scaffold.pt")
    ],
    key=lambda _r: (_r.split("-")[0], _a2_step(_r)),
)
A2_STAGE1 = [_r for _r in A2_REVS if _r.startswith("stage1")]
A2_STAGE1 = sorted(A2_STAGE1, key=_a2_step)

A2_SETUP = {
    "n_prompts": int(len(A2_WG)),
    "pools": {
        "vanilla_harmful": int(((~_adv) & (A2_LABELS == 1)).sum()),
        "vanilla_benign": int(((~_adv) & (A2_LABELS == 0)).sum()),
        "adversarial_harmful": int((_adv & (A2_LABELS == 1)).sum()),
        "adversarial_benign": int((_adv & (A2_LABELS == 0)).sum()),
    },
    "dupe_prompts": int(A2_WG["prompt"].duplicated().sum()),
    "n_fit": int(len(A2_VAN_TR)),
    "n_id_eval": int(len(A2_VAN_TE)),
    "n_ood_eval": int(len(A2_ADV_IDX)),
    "n_revisions": len(A2_REVS),
    "n_stage1": len(A2_STAGE1),
    "layers": A2_LAYERS,
}
A2_SETUP