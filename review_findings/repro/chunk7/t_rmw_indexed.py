"""LiftMapReductionToReduce(rmw_only=True) lifts a per-element update x[i] = x[i] + a[i] as a scalar reduction."""
import copy
import numpy as np, dace
from drv import compare, run
from dace.transformation.passes.vectorization.lift_map_reduction import LiftMapReductionToReduce

N = dace.symbol('N')


def build() -> dace.SDFG:
    body = dace.SDFG("body")
    body.add_scalar("xin", dace.float64)
    body.add_scalar("ain", dace.float64)
    body.add_scalar("xout", dace.float64)
    bs = body.add_state("s")
    t = bs.add_tasklet("upd", {"_x", "_a"}, {"_o"}, "_o = _x + _a")
    bs.add_edge(bs.add_read("xin"), None, t, "_x", dace.Memlet("xin[0]"))
    bs.add_edge(bs.add_read("ain"), None, t, "_a", dace.Memlet("ain[0]"))
    bs.add_edge(t, "_o", bs.add_write("xout"), None, dace.Memlet("xout[0]"))

    sdfg = dace.SDFG("t_rmw_indexed")
    sdfg.add_array("a", [N], dace.float64)
    sdfg.add_array("x", [N], dace.float64)
    st = sdfg.add_state("s")
    init = st.add_tasklet("init", {}, {"_o"}, "_o = 0.0")
    x_in = st.add_access("x")
    st.add_edge(init, "_o", x_in, None, dace.Memlet("x[0]"))
    me, mx = st.add_map("m", {"i": "0:N"})
    ns = st.add_nested_sdfg(body, {"xin", "ain"}, {"xout"})
    st.add_memlet_path(x_in, me, ns, dst_conn="xin", memlet=dace.Memlet("x[i]"))
    st.add_memlet_path(st.add_read("a"), me, ns, dst_conn="ain", memlet=dace.Memlet("a[i]"))
    st.add_memlet_path(ns, mx, st.add_write("x"), src_conn="xout", memlet=dace.Memlet("x[i]"))
    sdfg.validate()
    return sdfg


n = 19
rng = np.random.default_rng(0)
a = rng.random(n)
x = rng.random(n)
ref = x.copy()
ref[0] = 0.0
ref += a
plain = build()
print("unlifted:", end=" ")
compare({"x": ref}, run(plain, {"a": a, "x": x}, N=n))
lifted = build()
lifted.name = "t_rmw_indexed_lifted"
print("lift returned", LiftMapReductionToReduce(vectorized=True, rmw_only=True).apply_pass(lifted, {}))
lifted.validate()
print("lifted:", end=" ")
compare({"x": ref}, run(lifted, {"a": a, "x": x}, N=n))
