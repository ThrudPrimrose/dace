import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.vectorization.remove_unused_per_lane_symbols import RemoveUnusedPerLaneSymbols

sdfg = dace.SDFG("t_laneid")
sdfg.add_array("a", [16], dace.float64)
sdfg.add_array("b", [16], dace.float64)
sdfg.add_symbol("k_laneid_0", dace.int64)
init = sdfg.add_state("init", is_start_block=True)
loop = LoopRegion("loop", "i < 16", "i", "i = 0", "i = i + 1")
sdfg.add_node(loop)
sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={"k_laneid_0": "3"}))
body = loop.add_state("body", is_start_block=True)
t = body.add_tasklet("cp", {"_i"}, {"_o"}, "_o = _i")
body.add_edge(body.add_read("a"), None, t, "_i", dace.Memlet("a[k_laneid_0]"))
body.add_edge(t, "_o", body.add_write("b"), None, dace.Memlet("b[i]"))
sdfg.validate()
print("before: symbols", list(sdfg.symbols), "assignments", [e.data.assignments for e in sdfg.all_interstate_edges()])
print("pass returned", RemoveUnusedPerLaneSymbols().apply_pass(sdfg, {}))
print("after: symbols", list(sdfg.symbols), "assignments", [e.data.assignments for e in sdfg.all_interstate_edges()])
sdfg.validate()
print("valid")
