def ga_poll(n_lines=30, art=None):
    """Poll by READING FILES. Never re-run the worker cell to check on it."""
    _p = ga.paths(art or ga.ART)
    _st = pl.read_json(_p["status"]) if os.path.exists(_p["status"]) else {}
    _tail = []
    if os.path.exists(_p["log"]):
        with open(_p["log"]) as _fh:
            _tail = _fh.read().strip().split(chr(10))[-n_lines:]
    _files = sorted(os.listdir(_p["art"])) if os.path.isdir(_p["art"]) else []
    return {
        "stage": _st.get("stage"),
        "done": _st.get("done"),
        "error": _st.get("error"),
        "elapsed_min": round((time.time() - _st.get("t0", time.time())) / 60.0, 1),
        "artifacts": _files,
        "log_tail": _tail,
    }


ga_poll()
