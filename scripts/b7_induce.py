# -*- coding: utf-8 -*-
"""B7: can we INDUCE the margin/lever dissociation by fine-tuning?

This is the ICML-grade experiment. Everything else in the project is
observational: we watch margin and lever across checkpoints Ai2 happened to
release, with RL-Zero as a natural control. B7 makes it interventional. Take the
base model, apply refusal-style SFT ourselves, and watch what happens to the two
quantities while holding the steering direction FROZEN at the base fit.

PREDICTION (state it before running; this is the whole point)
    margin  rises monotonically with SFT step, approaching the released
            checkpoints' range
    lever   stays flat within the band we already measured across the flow
            (CV ~0.08, no systematic trend with step)
    cos(v_step, v_base) stays high -- the direction does not rotate away

FALSIFIERS, equally explicit
    F1  margin does NOT open under refusal SFT  -> SFT is not what widens the
        margin, and the RL-Zero contrast in the paper has some other cause.
        This kills the mechanism story outright.
    F2  lever moves WITH the margin (rises or falls together, |corr| high across
        steps) -> lever and load are not separable after all, which contradicts
        the paper's central claim in the one setting where we control the cause.
    F3  the induced margin growth comes with a large direction rotation
        (cos < ~0.7) -> what we induced is not the same phenomenon as the
        released flow, where the direction stays stable.

Any of those is a real result and worth reporting; F1 or F2 means the ICML
mechanism paper does not exist and the work should be rescoped, not rescued.

LEAKAGE (the failure this project keeps almost making)
    The SFT data MUST be disjoint from both the direction-fitting split and the
    held-out evaluation split. We assert this on the raw strings before training
    and refuse to run otherwise. The Gate-A fingerprint is asserted separately.

Cost: LoRA on a 7B is minutes per checkpoint on a Blackwell; the sweep at each
step is the same cost as one Gate-A checkpoint. Budget a few GPU-hours total,
which is why this is worth doing before committing months to the write-up.

Usage (GPU box, from repo root):
    python scripts/b7_induce.py --steps 0,25,50,100,200,400
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P   # noqa: E402
import gate_a as ga    # noqa: E402
import lever as L      # noqa: E402

OUT = "results/b7_induction.json"
GA_ART = "results/gateA_run"
BASE_REPO = "allenai/Olmo-3-1025-7B"
FIT_LAYER, STEER_LAYER = 20, 20
C_GRID = [-3.0, -2.5, -2.0, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
DEFAULT_STEPS = [0, 25, 50, 100, 200, 400]
LORA_R, LORA_ALPHA, LR, BATCH = 16, 32, 1e-4, 8


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def load_eval_split():
    data = P.load_prompts(n_train=200, n_held=200, seed=ga.SPLIT_SEED,
                          cache_path=os.path.join(GA_ART, "prompts.json"))
    fp = ga.split_fingerprint(data)
    if fp != ga.EXPECTED_SPLIT_FP:
        raise RuntimeError("split fingerprint mismatch: %s (STOP)" % fp)
    return data, fp


def build_sft_data(data, n=512, seed=ga.SEED):
    """Refusal-style SFT pairs from prompts DISJOINT from fit and held sets.

    Returns (texts, provenance). Raises if any overlap is detected -- an
    overlapping SFT set would train directly on the evaluation prompts and
    manufacture the very margin growth we are trying to induce.
    """
    import random
    forbidden = set(map(str.strip, list(data["harm_train"]) + list(data["harm_held"])
                        + list(data["ben_train"]) + list(data.get("ben_held", []))))
    pool = [p for p in ga.load_extra_harmful_pool()] if hasattr(ga, "load_extra_harmful_pool") else []
    if not pool:
        raise RuntimeError(
            "No disjoint harmful pool available. Supply one explicitly (e.g. the "
            "AdvBench rows NOT used by the Gate-A split, or a separate safety SFT "
            "set) rather than reusing the evaluation prompts.")
    pool = [p for p in pool if str(p).strip() not in forbidden]
    overlap = forbidden.intersection(map(str.strip, pool))
    if overlap:
        raise RuntimeError("SFT pool overlaps eval split on %d prompts (STOP)" % len(overlap))
    random.Random(seed).shuffle(pool)
    pool = pool[:n]
    refusals = ["I'm sorry, but I can't help with that.",
                "I can't assist with that request.",
                "Sorry, I'm not able to provide that."]
    rng = random.Random(seed + 1)
    texts = [P.fmt(p) + " " + rng.choice(refusals) for p in pool]
    return texts, dict(n=len(texts), source="disjoint harmful pool", seed=seed,
                       overlap_checked=True)


def measure(model, tok, harm_held, v_base, mu):
    """Margin and lever for the current model, frozen base direction."""
    r_ids = P.onset_token_ids(tok, P.REFUSAL_STRS)
    c_ids = P.onset_token_ids(tok, P.COMPLY_STRS)
    gaps = []
    for c in C_GRID:
        if c == 0.0:
            h = []
        else:
            h = P.add_direction(model, v_base, coeff=c * mu, layers=[STEER_LAYER]) \
                if hasattr(P, "add_direction") else None
            if h is None:
                raise RuntimeError("pipeline has no additive-steering helper; "
                                   "wire this to the same hook gate_a.py uses")
        try:
            gaps.append(P.logit_readout(model, tok, harm_held, r_ids, c_ids))
        finally:
            if h:
                P.remove_hooks(h)
    g = np.array(gaps, dtype=float)          # (n_doses, n_prompts)
    c = np.array(C_GRID, dtype=float)
    zi = int(np.where(c == 0.0)[0][0])
    disp = L.displacement(g, zi)
    return dict(margin=L.margin(g, zi), dprime=L.dprime(g, zi),
                lever=L.efficacy(c, disp),
                lever_secant=L.efficacy_secant_mean(c, disp),
                gaps=g.tolist(), c_grid=C_GRID, zero_index=zi)


def run(steps):
    import torch
    from peft import LoraConfig, get_peft_model

    data, fp = load_eval_split()
    harm_held = list(data["harm_held"])
    sft_texts, sft_prov = build_sft_data(data)
    log("split %s | held %d | sft %d (disjoint asserted)" % (fp[:8], len(harm_held), len(sft_texts)))

    z = np.load(os.path.join(GA_ART, "directions.npz"))
    v_base = np.asarray(z["direction"], dtype=np.float64).ravel()
    v_base /= np.linalg.norm(v_base) + 1e-12

    model = P.load_model(BASE_REPO)
    tok = P.load_tokenizer(BASE_REPO)
    mu = float(json.load(open(os.path.join(GA_ART, "..", "gateA_sweep_base.json")))["mu_used"])
    log("frozen base direction loaded; mu = %.3f" % mu)

    out = {"experiment": "B7 induced margin growth", "split_fingerprint": fp,
           "base_repo": BASE_REPO, "c_grid": C_GRID, "sft": sft_prov,
           "lora": dict(r=LORA_R, alpha=LORA_ALPHA, lr=LR, batch=BATCH),
           "prediction": "margin rises, lever flat, cos(v_step,v_base) high",
           "steps": {}}
    if os.path.exists(OUT):
        out = json.load(open(OUT))

    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    done, step = 0, 0
    for target in sorted(steps):
        while step < target:
            batch = sft_texts[(done * BATCH) % len(sft_texts):][:BATCH]
            if len(batch) < BATCH:
                batch = sft_texts[:BATCH]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=P.MAX_LEN).to("cuda")
            loss = model(**enc, labels=enc["input_ids"]).loss
            loss.backward(); opt.step(); opt.zero_grad()
            step += 1; done += 1
        model.eval()
        with torch.no_grad():
            rec = measure(model, tok, harm_held, v_base, mu)
        # did the direction itself move?
        pack = P.extract_activations(model, tok,
                                     list(data["harm_train"]) + list(data["ben_train"]),
                                     [FIT_LAYER], cache_path=None)
        lab = np.array([1] * len(data["harm_train"]) + [0] * len(data["ben_train"]))
        v_now, _ = P.refusal_direction(pack["acts"], lab, layer_index=0)
        rec["cos_to_base"] = float(np.dot(v_now / np.linalg.norm(v_now), v_base))
        rec["sft_loss"] = float(loss.item())
        out["steps"][str(target)] = rec
        json.dump(out, open(OUT, "w"), indent=1)
        log("step %4d  margin %.3f  lever %.3f  cos %.3f  loss %.3f"
            % (target, rec["margin"], rec["lever"], rec["cos_to_base"], rec["sft_loss"]))
        model.train()

    # verdict against the pre-stated falsifiers
    ks = sorted(out["steps"], key=int)
    m = np.array([out["steps"][k]["margin"] for k in ks])
    lv = np.array([out["steps"][k]["lever"] for k in ks])
    cs = np.array([out["steps"][k]["cos_to_base"] for k in ks])
    verdict = []
    if m[-1] / max(m[0], 1e-9) < 2.0:
        verdict.append("F1: margin did not open under refusal SFT")
    if lv.std() / lv.mean() > 0.20 and abs(np.corrcoef(m, lv)[0, 1]) > 0.8:
        verdict.append("F2: lever tracked the margin")
    if cs.min() < 0.7:
        verdict.append("F3: direction rotated substantially")
    out["verdict"] = verdict or ["prediction held: margin opened, lever flat, direction stable"]
    out["margin_fold"] = float(m[-1] / max(m[0], 1e-9))
    out["lever_cv"] = float(lv.std() / lv.mean())
    json.dump(out, open(OUT, "w"), indent=1)
    log("VERDICT: %s" % "; ".join(out["verdict"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(map(str, DEFAULT_STEPS)))
    a = ap.parse_args()
    run([int(s) for s in a.steps.split(",")])
