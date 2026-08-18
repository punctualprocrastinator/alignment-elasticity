import marimo._code_mode as cm

MD = '''mo_a1.md(
    r"""
# A4 - 32B SCALE REPLICATION

"One lineage, one scale" is the first objection any reviewer raises. This
re-runs the E2-clean elasticity measurement on the **Olmo-3 32B** flow and puts
the 32B drift curve next to the 7B one.

**Scale.** 32B is 64 layers, `d_model = 5120`, ~64.5 GB of bf16 weights per
checkpoint. The layer grid is proportional to the 7B grid (8/64 ... 63/64)
rather than reusing the 7B indices, batch size drops to 8, and each checkpoint
is reduced to a ~330 MB activation cache and then purged before the next
download.

**Noise floor is recomputed, not reused.** The 7B floor of |cos| ~ 0.0117 is a
property of R^4096. In R^5120 it is **0.01115** (analytic sqrt(2/pi*d) =
0.011151). As a check on the implementation, the same code reproduces 0.01172
for R^4096.

**P1b.** Probes are fit at `tol=1e-10`. On this data the default `tol=1e-4`
stops LBFGS at 18 iterations versus 33 - exactly the regime that leaves the
fitted direction rotated by ~0.02 from the optimum, which is fatal when the
headline metric IS a direction cosine.

**Contrast set.** `allenai/wildguardmix` is gated and this box has no
`HF_TOKEN`, so the pools are AdvBench harmful vs Alpaca benign (500 each,
SEED=42, fingerprint recorded). The direction work does not need the
adversarial split. The frozen-vs-refit AUROC arm therefore runs on this
**easier** split and its absolute AUROC values are not comparable to a
WildGuard-based number - only the frozen-vs-refit *gap* within this split is.
"""
)'''

SETUP = '''import pipeline_a4 as a4

a4 = importlib.reload(a4)

A4_DIR = a1p.ensure_dir(a4.A4_DIR)
A4_CONTRAST = a4.load_contrast(cache_path=os_a1.path.join(A4_DIR, "contrast.json"))
A4_FLOOR = a4.random_cos_floor(5120, k=20, seed=42)
A4_REF7B = a1p.read_json(os_a1.path.join(A4_DIR, "ref7b.json"))

print("module:", a4.__file__)
print("layers (64-layer model):", a4.A4_LAYERS, "| poolings:", a4.A4_POOLINGS)
print("pools: %d harmful / %d benign | fingerprint %s"
      % (len(A4_CONTRAST["harmful"]), len(A4_CONTRAST["benign"]), A4_CONTRAST["fingerprint"]))
print("noise floor R^5120: |cos| mean %.5f (p95 %.5f, analytic %.5f)"
      % (A4_FLOOR["abs_cos_mean"], A4_FLOOR["abs_cos_p95"],
         A4_FLOOR["analytic_sqrt_2_over_pi_d"]))
print("7B reference floor R^4096: %.5f" % A4_REF7B["random_direction_reference"]["abs_cos_mean"])
print()
print("checkpoints (branch AND commit pinned):")
for _r, _v, _f, _l, _o in a4.CKPTS_A4:
    print("   %d %-10s %-30s %-16s %s" % (_o, _l, _r, _v, _f))'''

async with cm.get_context() as ctx:
    a = ctx.create_cell(MD, hide_code=False)
    b = ctx.create_cell(SETUP, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
