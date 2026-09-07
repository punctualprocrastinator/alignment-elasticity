# -*- coding: utf-8 -*-
"""Track B, GPU half: WHERE does the margin growth live?

The logit gap this paper measures is
    gap = mean logsoftmax[refusal ids] - mean logsoftmax[comply ids]
        = <h, u>  + const,      u = mean W_U[refusal] - mean W_U[comply]
(the logsumexp cancels in the difference), where h is the post-final-norm hidden
state at the readout position. So the margin factorises exactly:

    margin = ||h|| * ||u|| * cos(h, u)

and the 9.5x growth must be carried by some combination of the three. That is
the decomposition B4 measures. B2 and B3 then ask where along the *residual
stream* the change sits, and whether the steering direction itself rotated.

  B2  projection: <mean unsteered h_L20, v_hat_base> per checkpoint.
      Prediction if alignment TRANSLATES the stream along the refusal axis:
      grows monotonically, tracks the margin, flat for the RL-Zero models
      (which never saw SFT and whose margin never opened).
  B3  rotation: refit the diff-in-means direction at each checkpoint and take
      the cosine to the frozen base direction. High cosine => translation, not
      rotation, which is what a flat lever predicts.
  B4  readout decomposition: ||h||, ||u||, cos(h,u), and their product checked
      against the measured margin.

All three reuse the Gate-A split (fingerprint asserted) and the frozen base
direction, so numbers are comparable to the sweeps. Writes incrementally: a
molab box can be reclaimed mid-run.

Usage (from repo root, on a GPU box):
    python scripts/mechanism.py
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P            # noqa: E402
import gate_a as ga             # noqa: E402

OUT = "results/mechanism_geometry.json"
GA_ART = "results/gateA_run"
FIT_LAYER = 20
PROBE_LAYERS = [8, 12, 16, 20, 24, 28]      # B2 across depth, cheap once loaded
N_TRAIN, N_HELD = 200, 200

# (label, repo, revision) -- revisions pinned from the committed sweeps
CKPTS = [
    ("base",             "allenai/Olmo-3-1025-7B",      "main"),
    ("think-sft-1000",   "allenai/Olmo-3-7B-Think-SFT", "step1000"),
    ("think-sft-15000",  "allenai/Olmo-3-7B-Think-SFT", "step15000"),
    ("think-sft-43000",  "allenai/Olmo-3-7B-Think-SFT", "step43000"),
    ("think-dpo",        "allenai/Olmo-3-7B-Think-DPO", "main"),
    ("think-rlvr-last",  "allenai/Olmo-3-7B-Think",     "step_1375"),
    ("instruct",         "allenai/Olmo-3-7B-Instruct",  "main"),
    ("rlz-math",         "allenai/Olmo-3-7B-RLZero-Math", "main"),
    ("rlz-code",         "allenai/Olmo-3-7B-RLZero-Code", "main"),
]


def log(m):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
    print(line, flush=True)


def load_split():
    """Gate-A's split, with the fingerprint asserted (never re-derive it)."""
    data = P.load_prompts(n_train=N_TRAIN, n_held=N_HELD, seed=ga.SPLIT_SEED,
                          cache_path=os.path.join(GA_ART, "prompts.json"))
    fp = ga.split_fingerprint(data)
    if fp != ga.EXPECTED_SPLIT_FP:
        raise RuntimeError("split fingerprint mismatch: %s (STOP)" % fp)
    return data, fp


def load_base_direction():
    """The frozen base refusal direction the sweeps used."""
    z = np.load(os.path.join(GA_ART, "directions.npz"))
    key = "direction" if "direction" in z else list(z.keys())[0]
    v = np.asarray(z[key], dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-12)


def readout_vector(model, r_ids, c_ids):
    """u = mean W_U[refusal] - mean W_U[comply]; gap = <h, u> + const."""
    import torch
    W = model.get_output_embeddings().weight      # (vocab, d_model)
    with torch.no_grad():
        u = (W[r_ids].float().mean(0) - W[c_ids].float().mean(0))
    return u.detach().cpu().numpy().astype(np.float64)


def final_hidden(model, tok, prompts, batch_size=8):
    """Post-final-norm hidden state at the readout position (last non-pad token).

    This is the vector the unembedding actually sees, so <h, u> reproduces the
    logit gap up to a constant. Uses right padding + attn.sum()-1 exactly as
    pipeline.extract_activations does, so pads cannot contaminate the readout.
    """
    import torch
    outs = []
    for s in range(0, len(prompts), batch_size):
        enc = P.encode_batch(tok, [P.fmt(p) for p in prompts[s:s + batch_size]],
                             device="cuda", side="right")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[-1]
            hs = model.model.norm(hs)             # apply the final norm
        last = enc["attention_mask"].sum(1) - 1
        outs.append(hs[torch.arange(hs.shape[0]), last].float().cpu().numpy())
    return np.concatenate(outs, 0).astype(np.float64)


def run():
    import torch
    data, fp = load_split()
    v_base = load_base_direction()
    harm_tr, ben_tr = list(data["harm_train"]), list(data["ben_train"])
    harm_held = list(data["harm_held"])
    fit_prompts = harm_tr + ben_tr
    fit_labels = np.array([1] * len(harm_tr) + [0] * len(ben_tr))
    log("split %s | fit %d | held %d" % (fp[:8], len(fit_prompts), len(harm_held)))

    out = {"experiment": "mechanism geometry (B2/B3/B4)", "split_fingerprint": fp,
           "fit_layer": FIT_LAYER, "probe_layers": PROBE_LAYERS,
           "seed": ga.SEED, "checkpoints": {}}
    if os.path.exists(OUT):
        out = json.load(open(OUT))                # resume

    for label, repo, rev in CKPTS:
        if label in out["checkpoints"]:
            log("skip %s (done)" % label)
            continue
        log("loading %s @ %s" % (repo, rev))
        model = P.load_model(repo, revision=rev)
        tok = P.load_tokenizer(repo, revision=rev)

        # --- B2/B3: residual-stream activations on fit + held sets ---
        pack_fit = P.extract_activations(model, tok, fit_prompts, PROBE_LAYERS,
                                         cache_path=None)
        li = list(pack_fit["layers"]).index(FIT_LAYER)
        v_ck, _ = P.refusal_direction(pack_fit["acts"], fit_labels, layer_index=li)
        cos_to_base = float(np.dot(v_ck / np.linalg.norm(v_ck), v_base))

        pack_held = P.extract_activations(model, tok, harm_held, PROBE_LAYERS,
                                          cache_path=None)
        proj = {}
        for j, lay in enumerate(pack_held["layers"]):
            A = np.asarray(pack_held["acts"][:, j, :], dtype=np.float64)
            proj["L%d" % lay] = dict(
                mean_proj=float(A.mean(0) @ v_base),         # B2
                mean_norm=float(np.linalg.norm(A, axis=1).mean()),
                proj_per_prompt_sd=float((A @ v_base).std()),
            )

        # --- B4: readout decomposition of the margin ---
        r_ids = P.onset_token_ids(tok, P.REFUSAL_STRS)
        c_ids = P.onset_token_ids(tok, P.COMPLY_STRS)
        u = readout_vector(model, r_ids, c_ids)
        H = final_hidden(model, tok, harm_held)
        hbar = H.mean(0)
        nh, nu = float(np.linalg.norm(hbar)), float(np.linalg.norm(u))
        cos_hu = float(hbar @ u / (nh * nu + 1e-12))
        out["checkpoints"][label] = dict(
            repo=repo, revision=rev,
            cos_direction_to_base=cos_to_base,                 # B3
            projection=proj,                                    # B2
            readout=dict(h_norm=nh, u_norm=nu, cos_h_u=cos_hu,  # B4
                         product=nh * nu * cos_hu,
                         gap_from_dot=float(hbar @ u),
                         h_norm_per_prompt_sd=float(np.linalg.norm(H, axis=1).std())),
        )
        json.dump(out, open(OUT, "w"), indent=1)
        log("%s: cos(v_ck,v_base)=%.3f  projL20=%.3f  <h,u>=%.3f (|h|%.1f |u|%.3f cos%.3f)"
            % (label, cos_to_base, proj["L%d" % FIT_LAYER]["mean_proj"],
               hbar @ u, nh, nu, cos_hu))
        P.free_model(model)
        del tok
        torch.cuda.empty_cache()

    log("done -> %s" % OUT)


if __name__ == "__main__":
    run()
