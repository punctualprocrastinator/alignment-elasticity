import marimo._code_mode as cm

LAUNCH = '''# Detached: config passed as thread ARGUMENTS (protocol P10), never read from
# notebook globals - re-running an owning cell would otherwise kill the thread.
E46_LAUNCH = e46.e4e6_launch(
    e4_kw=dict(
        art_dir=e46.E4_DIR,
        ckpts=e46.CKPTS_E4,
        n_behav=200,
        gen_new=512,
        gen_batch=32,
        fit_layer=20,
        n_fit=200,
        seed=42,
        elicit=("intact", "ablated", "prefill"),
        n_benign=200,
    ),
    judge_kw=dict(art_dir=e46.E4_DIR, batch_size=8, max_len=2048),
    e6_kw=dict(
        art_dir=e46.E6_DIR,
        gen_dir=e46.E4_DIR,
        monitors=e46.MONITOR_LABELS,
        n_pos=200,
        n_neg=200,
        batch_size=8,
        seed=42,
    ),
)
print("thread:", E46_LAUNCH.name, "alive:", E46_LAUNCH.is_alive())
print("poll:", os_a1.path.join(e46.E4_DIR, "log.txt"), "|", os_a1.path.join(e46.E6_DIR, "log.txt"))'''

async with cm.get_context() as ctx:
    a = ctx.create_cell('mo_a1.md(r"""' + chr(10) + "## E4/E6.1 Launch (detached, idempotent, config as arguments)" + chr(10) + '""")', hide_code=False)
    b = ctx.create_cell(LAUNCH, hide_code=False)
    ctx.run_cell(a)
    ctx.run_cell(b)
    print("cells:", a, b)
