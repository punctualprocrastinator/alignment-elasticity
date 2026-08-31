# -*- coding: utf-8 -*-
"""TOST / non-inferiority test + per-checkpoint efficacy CIs for the refusal lever.
Zero-GPU: bootstraps the efficacy slope from the per-prompt gaps in the sweep JSONs.
Pre-registered before looking at the bootstrap: equivalence margin = 20% of base
efficacy; 'non-weakening' = each post-training checkpoint's efficacy is non-inferior
to base by that margin (lower 95% bound of the paired difference > -margin)."""
import json, os, numpy as np

ORDER = [("base","base"),("SFT 1k","think-sft-1000"),("SFT 15k","think-sft-15000"),
         ("SFT 43k","think-sft-43000"),("DPO","think-dpo"),("RLVR first","think-rlvr-first"),
         ("RLVR last","think-rlvr-last"),("Instruct","instruct"),
         ("RL-Zero Math","rlz-math"),("RL-Zero Code","rlz-code")]

def load(l):
    for p in ["results/gateA_traj/gateA_sweep_%s.json"%l, "results/gateA_sweep_%s.json"%l]:
        if os.path.exists(p): return json.load(open(p))

def slope(gaps, c, zi, idx=None):
    g = gaps[:, idx] if idx is not None else gaps
    g0 = g[zi]; absc = np.abs(c); near = absc <= 2.0
    disp = (g0[None,:] - g).mean(axis=1)
    A = np.vstack([absc[near], np.ones(near.sum())]).T
    return float(np.linalg.lstsq(A, disp[near], rcond=None)[0][0])

# load all checkpoints; they share the same held-out split (fingerprint 99a7ac88), so
# a single resampled prompt-index vector pairs them.
data = {}
for name, k in ORDER:
    d = load(k)
    data[name] = (np.array(d["gaps_massmean"]), np.array(d["c_grid_massmean"]), d["zero_index_massmean"])
n_prompts = data["base"][0].shape[1]

rng = np.random.default_rng(42)
B = 1000
point = {name: slope(*data[name]) for name,_ in ORDER}
base0 = point["base"]
MARGIN = 0.20 * base0     # pre-registered equivalence margin

boot = {name: np.empty(B) for name,_ in ORDER}
cv_boot = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n_prompts, size=n_prompts)
    effs = []
    for name,_ in ORDER:
        g,c,zi = data[name]
        e = slope(g, c, zi, idx); boot[name][b] = e; effs.append(e)
    effs = np.array(effs)
    cv_boot[b] = effs.std()/effs.mean()

def ci(a): return (float(np.percentile(a,2.5)), float(np.percentile(a,97.5)))

print("Pre-registered equivalence margin delta = 0.20 x base efficacy = %.3f\n" % MARGIN)
print("%-13s %8s  %-18s %-22s" % ("checkpoint","effic.","95%% CI","diff vs base [95%% CI]"))
post = [n for n,_ in ORDER if n!="base"]
noninf = {}
for name,_ in ORDER:
    lo,hi = ci(boot[name])
    if name=="base":
        print("%-13s %8.3f  [%.3f, %.3f]" % (name, point[name], lo, hi)); continue
    diff = boot[name] - boot["base"]        # paired
    dlo,dhi = ci(diff)
    ni = dlo > -MARGIN                       # non-inferiority (non-weakening)
    noninf[name] = ni
    print("%-13s %8.3f  [%.3f, %.3f]   %+.3f [%+.3f, %+.3f] %s" %
          (name, point[name], lo, hi, float(diff.mean()), dlo, dhi,
           "NON-WEAKENING" if ni else "fails"))

pts = np.array([point[n] for n,_ in ORDER])
print("\nCV over the ten point estimates = %.3f, bootstrap 95%% CI [%.3f, %.3f]"
      % (pts.std()/pts.mean(), *ci(cv_boot)))
print("efficacy range %.2f-%.2f (%.2fx spread)" % (pts.min(), pts.max(), pts.max()/pts.min()))
print("\nNon-weakening established (all 9 post-training checkpoints): %s" % all(noninf.values()))
# also the two-sided equivalence verdict vs base for the strongest grower
print("Instruct is significantly ABOVE base (lower diff bound %.3f > 0): %s"
      % (ci(boot["Instruct"]-boot["base"])[0], ci(boot["Instruct"]-boot["base"])[0] > 0))

# save for the paper
out = {"margin_delta": MARGIN, "base_efficacy": base0,
       "efficacy": {n: {"point": point[n], "ci": ci(boot[n])} for n,_ in ORDER},
       "diff_vs_base": {n: {"point": float((boot[n]-boot["base"]).mean()),
                            "ci": ci(boot[n]-boot["base"]), "non_weakening": bool(noninf.get(n,True))}
                        for n,_ in ORDER if n!="base"},
       "cv_point": float(pts.std()/pts.mean()), "cv_ci": ci(cv_boot),
       "all_non_weakening": bool(all(noninf.values())), "n_boot": B, "seed": 42}
json.dump(out, open("results/efficacy_tost.json","w"), indent=1)
print("\nwrote results/efficacy_tost.json")
