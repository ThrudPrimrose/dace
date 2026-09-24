import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize, stage_labels
from ivs_prog import counted

sdfg = counted.to_sdfg(simplify=True)
prev = None
for label in dict.fromkeys(stage_labels()):
    canonicalize(sdfg, stages=[label])
    loops = sum(isinstance(b, LoopRegion) for b in sdfg.all_control_flow_blocks(recursive=True))
    codes = sorted(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet))
    cur = (loops, codes)
    if cur != prev:
        print(label, loops, codes)
    prev = cur
