# Box B setup cell (marimo id ILDf). Puts /marimo/gateA on the path and imports
# the reviewed modules restored to this box. e1_causal + e2_behavioural +
# e4_honesty + e5_family were uploaded byte-identical; pipeline/gate_a/
# gate_a_analysis/gate_a_sentiment already present.
import sys
if "/marimo/gateA" not in sys.path:
    sys.path.insert(0, "/marimo/gateA")
import pipeline, gate_a, e1_causal, e2_behavioural, e4_honesty, e5_family
print("Box B modules loaded from /marimo/gateA")
