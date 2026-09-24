import numpy as np
import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize, stage_labels

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def chained(a: dace.float64[N], b: dace.float64[N], s: dace.float64[1]):
    for i in range(N):
        s[0] = s[0] * 0.5 + a[i]
        s[0] = s[0] * 0.5 + b[i]


labels = stage_labels()
upto = labels[:labels.index('reduction_to_wcr_map')]
sdfg = chained.to_sdfg(simplify=True)
canonicalize(sdfg, stages=upto)
for loop in sdfg.all_control_flow_regions(recursive=True):
    if isinstance(loop, LoopRegion):
        for st in loop.all_states():
            for n in st.nodes():
                if isinstance(n, nodes.Tasklet):
                    print(loop.label, st.label, repr(n.code.as_string))
