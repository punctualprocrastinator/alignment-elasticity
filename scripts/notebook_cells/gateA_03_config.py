# Config is passed to the worker as THREAD ARGUMENTS. Nothing inside the worker
# reads a notebook global, so re-running an owning cell cannot kill the run with
# a NameError (protocol P10).
GA_CONFIG = {
    "art": ga.ART,
    "ckpts": ga.CKPTS,
    "n_train": ga.N_TRAIN,
    "n_held": ga.N_HELD,
    "n_rand": ga.N_RAND,
    "n_gen": ga.N_GEN,
    "batch_size": ga.BATCH,
    "gen_batch_size": ga.GEN_BATCH,
    "gen_new_tokens": ga.GEN_NEW_TOKENS,
    "c_grid": ga.C_GRID,
    "c_grid_null": ga.C_GRID_NULL,
    "do_generation": True,
}
GA_CONFIG
