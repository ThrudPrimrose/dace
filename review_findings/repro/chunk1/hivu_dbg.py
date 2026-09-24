import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import hoist_iv_updates as h
from hivu_probe_prog import cross
sdfg = cross.to_sdfg(simplify=True)
for loop in sdfg.all_control_flow_regions(recursive=True):
    if isinstance(loop, LoopRegion):
        st = loop.nodes()[0]
        for e in st.edges():
            print(e.src, e.src_conn, '->', e.dst, e.dst_conn, e.data)
        for t in st.nodes():
            if isinstance(t, nodes.Tasklet):
                print(t, repr(t.code.as_string), h._is_iv_eligible_tasklet(t, st, loop, sdfg))
