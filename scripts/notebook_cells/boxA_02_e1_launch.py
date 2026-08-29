# E1 launch (detached worker). Reuses /marimo/gateA/{prompts.json,directions.npz}
# so the split fingerprint reproduces 99a7ac88. Trains a gradient causal L20
# direction on base, then sweeps 3 direction families x 4 checkpoints.
import e1_causal as e1
print(e1.e1_launch())
# Poll: json.load(open('/marimo/e1/status.json')); open('/marimo/e1/log.txt').read()
