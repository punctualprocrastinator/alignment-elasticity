# Idempotent, duplicate-safe launch. gate_a_launch refuses to start a second
# worker while one is alive and refuses to start at all once every sweep
# artifact exists, so re-running this cell is free.
GA_LAUNCH = ga.gate_a_launch(**GA_CONFIG)
GA_LAUNCH
