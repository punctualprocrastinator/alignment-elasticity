import marimo._code_mode as cm

LAUNCH = '''# Guarded launcher (P10): a4_launch_all refuses to spawn a second thread while
# one is alive, and the worker skips any checkpoint whose activation cache
# already exists. Re-running this cell is therefore safe and cheap.
A4_LAUNCH = a4.a4_launch_all(
    worker_kw=dict(
        art_dir=a4.A4_DIR,
        ckpts=a4.CKPTS_A4,
        layers=a4.A4_LAYERS,
        n_per_pool=500,
        batch_size=8,
        max_len=384,
        seed=42,
    ),
    analyze_kw=dict(
        art_dir=a4.A4_DIR,
        ckpts=a4.CKPTS_A4,
        layers=a4.A4_LAYERS,
        n_per_pool=500,
        seed=42,
        n_boot=1000,
        n_boot_refit=60,
        n_perm=3,
        anchor="base",
        tol=1e-10,
    ),
)
print(A4_LAUNCH)
print("poll:", os_a1.path.join(a4.A4_DIR, "log.txt"))'''

async with cm.get_context() as ctx:
    a = ctx.create_cell('mo_a1.md(r"""' + chr(10) + "## A4.1 Launch (detached, guarded, config as arguments)" + chr(10) + '""")', hide_code=False)
    b = ctx.create_cell(LAUNCH, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
