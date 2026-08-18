# E2-clean configuration, protocol v1 (see protocol.md).
# Everything the detached workers need is packed into E2_CFG and passed as a
# THREAD ARGUMENT -- P10: a worker that reads notebook globals dies with
# NameError the moment its owning cell is re-run (observed in A2).
import hashlib as _h2
import os as _os2

import numpy as _np2b
import pandas as _pd2b
from sklearn.model_selection import train_test_split as _tts2b

E2_DIR = "/marimo/e2c"
E2_ACT_DIR = E2_DIR + "/acts"
_os2.makedirs(E2_ACT_DIR, exist_ok=True)

E2_SEED = 42
E2_LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]
E2_POOLINGS = ["last", "mean"]
E2_MAXLEN = 384
E2_BATCH = 16
E2_NBOOT = 1000
E2_NBOOT_REFIT = 200
E2_NULL_SEEDS = [0, 1, 2]
E2_NRAND = 20
E2_SCAFFOLD = "User: {prompt}" + chr(10) + "Assistant:"

# --- contrast set ----------------------------------------------------------
# PROTOCOL DEVIATION (P2): protocol asks for 1,000 prompts per pool. Resampling
# needs the gated allenai/wildguardmix, and HF_TOKEN is NOT set on this box, so
# E2-clean runs on the existing deduped 4x500 pools. Recorded in every result
# file and in the verdict.
E2_WG = _pd2b.read_parquet("/marimo/wildguard_contrast.parquet")
E2_PROMPTS = list(E2_WG["prompt"])
E2_LABELS = E2_WG["label"].to_numpy().astype(int)
_advm = E2_WG["adversarial"].to_numpy()
E2_VAN_IDX = _np2b.where(~_advm)[0]
E2_ADV_IDX = _np2b.where(_advm)[0]
E2_VAN_TR, E2_VAN_TE = _tts2b(
    E2_VAN_IDX, test_size=0.2, stratify=E2_LABELS[E2_VAN_IDX], random_state=E2_SEED
)
E2_SPLIT_FP = _h2.sha1(
    E2_VAN_TR.tobytes() + E2_VAN_TE.tobytes() + E2_ADV_IDX.tobytes()
).hexdigest()[:12]

E2_DEVIATIONS = [
    "P2 sample size: 500 per pool, not 1000 -- allenai/wildguardmix is gated "
    "and HF_TOKEN is not set on this sandbox; resampling was impossible.",
    "P4 bootstrap: refit-based quantities (probe-direction cosine) use 200 "
    "resamples rather than 1000, because each resample requires a fresh "
    "logistic fit; all non-refit quantities (AUROC, paired AUROC difference, "
    "diff-in-means cosine) use the full 1000.",
    "P1 probe seeds: logistic regression on a fixed split is deterministic, so "
    "the >=3-seed rule is satisfied by 3 independent shuffled-null permutations "
    "per (layer, pooling) plus a 1000-resample bootstrap, which is strictly "
    "stronger than 3 point fits.",
]

# --- checkpoints, pinned by branch AND commit (P9) --------------------------
# main differs from the last step branch in Think, Think-SFT and RL-Zero-Math,
# so every revision below is an explicit branch, never main (except the base
# anchor and DPO, which are endpoint-only repos).
E2_PLAN = [
    ("base", "allenai/Olmo-3-1025-7B", "main", "base", 0),
    ("sft_1000", "allenai/Olmo-3-7B-Think-SFT", "step1000", "sft", 1),
    ("sft_15000", "allenai/Olmo-3-7B-Think-SFT", "step15000", "sft", 2),
    ("sft_29000", "allenai/Olmo-3-7B-Think-SFT", "step29000", "sft", 3),
    ("sft_43000", "allenai/Olmo-3-7B-Think-SFT", "step43000", "sft", 4),
    ("dpo", "allenai/Olmo-3-7B-Think-DPO", "main", "dpo", 5),
    ("rlvr_0025", "allenai/Olmo-3-7B-Think", "step_0025", "rlvr", 6),
    ("rlvr_0275", "allenai/Olmo-3-7B-Think", "step_0275", "rlvr", 7),
    ("rlvr_0550", "allenai/Olmo-3-7B-Think", "step_0550", "rlvr", 8),
    ("rlvr_0825", "allenai/Olmo-3-7B-Think", "step_0825", "rlvr", 9),
    ("rlvr_1100", "allenai/Olmo-3-7B-Think", "step_1100", "rlvr", 10),
    ("rlvr_1375", "allenai/Olmo-3-7B-Think", "step_1375", "rlvr", 11),
    ("rlzero_1900", "allenai/Olmo-3-7B-RL-Zero-Math", "step_1900", "rl_zero", 12),
]
E2_COMMITS = {
    "base": "a81bae42db39", "sft_1000": "9a45447cd55e",
    "sft_15000": "57650de2c118", "sft_29000": "9657b7609824",
    "sft_43000": "aaf125a44bae", "dpo": "7b18bf927b43",
    "rlvr_0025": "817b9d38d9cf", "rlvr_0275": "4aa898c7ed12",
    "rlvr_0550": "579c443ed701", "rlvr_0825": "ebeabbcaab2b",
    "rlvr_1100": "f9903ff99bf6", "rlvr_1375": "031240693eb3",
    "rlzero_1900": "8182367150ce",
}
E2_ANCHOR = "base"

E2_CFG = {
    "dir": E2_DIR,
    "act_dir": E2_ACT_DIR,
    "seed": E2_SEED,
    "layers": list(E2_LAYERS),
    "poolings": list(E2_POOLINGS),
    "max_len": E2_MAXLEN,
    "batch_size": E2_BATCH,
    "scaffold": E2_SCAFFOLD,
    "prompts": list(E2_PROMPTS),
    "labels": E2_LABELS.tolist(),
    "van_tr": E2_VAN_TR.tolist(),
    "van_te": E2_VAN_TE.tolist(),
    "adv_idx": E2_ADV_IDX.tolist(),
    "split_fp": E2_SPLIT_FP,
    "plan": [list(_p) for _p in E2_PLAN],
    "commits": dict(E2_COMMITS),
    "anchor": E2_ANCHOR,
    "n_boot": E2_NBOOT,
    "n_boot_refit": E2_NBOOT_REFIT,
    "null_seeds": list(E2_NULL_SEEDS),
    "n_rand": E2_NRAND,
    "n_per_pool": 500,
    "deviations": list(E2_DEVIATIONS),
}

{
    "split_fp": E2_SPLIT_FP,
    "n_prompts": len(E2_PROMPTS),
    "n_fit": len(E2_VAN_TR),
    "n_id_eval": len(E2_VAN_TE),
    "n_ood_eval": len(E2_ADV_IDX),
    "n_checkpoints": len(E2_PLAN),
    "deviations": E2_DEVIATIONS,
}
