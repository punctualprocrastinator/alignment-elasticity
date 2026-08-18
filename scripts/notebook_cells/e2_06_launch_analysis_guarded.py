# Launch the E2-clean analysis sweep (detached, cfg as a thread argument, P10).
# Guarded: re-running an upstream cell must not spawn a second sweep racing the
# first (that happened once already and both threads then fought over the GPU).
import os as _osg
import threading as _thg

_alive = [_t for _t in _thg.enumerate() if _t.name.startswith('e2-analysis')]
_pending = [
    _r[0] for _r in E2_PLAN
    if _r[0] != E2_ANCHOR
    and not _osg.path.exists(
        E2_DIR + '/res_n' + str(E2_CFG['n_per_pool']) + '_' + _r[0] + '.json'
    )
]
E2_ANALYSIS_THREAD = (
    e2_launch_analysis(E2_CFG) if (not _alive and _pending) else None
)
{'launched': E2_ANALYSIS_THREAD is not None, 'already_running': len(_alive),
 'pending': _pending}
