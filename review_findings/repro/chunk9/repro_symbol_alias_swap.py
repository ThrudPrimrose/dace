# PYTHONHASHSEED=0 python repro_symbol_alias_swap.py
import sys
sys.path.insert(0, '.')
import dace
from swapbuild2 import build
from dace.transformation.passes.vectorization.vectorize_multi_dim import normalize_loop_nests

s = build()
ns = next(x for x, _ in s.all_nodes_recursive() if isinstance(x, dace.nodes.NestedSDFG))
print("before: mapping", dict(ns.symbol_mapping), "inner symbols", sorted(ns.sdfg.symbols))
normalize_loop_nests(s)
ns = next(x for x, _ in s.all_nodes_recursive() if isinstance(x, dace.nodes.NestedSDFG))
print("after:  mapping", dict(ns.symbol_mapping), "inner symbols", sorted(ns.sdfg.symbols))
print("inner memlets:", sorted({str(e.data) for st in ns.sdfg.states() for e in st.edges()}))
try:
    s.validate()
    print("valid")
except Exception as ex:
    print("INVALID:", type(ex).__name__, str(ex).splitlines()[0][:200])
