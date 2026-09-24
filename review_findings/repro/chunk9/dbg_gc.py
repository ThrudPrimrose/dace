import sys; sys.path.insert(0, '.')
from drv import *
from dace.transformation.passes.vectorization import widen_accesses as wa
from dace.transformation.passes.vectorization.utils.tile_access import classify_tile_access
orig = wa.WidenAccesses.apply_pass
def patched(self, sdfg, pr):
    for st, ns, me, w in self._body_nsdfgs(sdfg):
        inner = ns.sdfg
        print("MAP", me.map.params, me.map.range)
        for e in inner.all_interstate_edges(): 
            if e.data.assignments: print("  ISE", e.data.assignments)
        for s in inner.states():
            for e in s.edges():
                print("  ", s.label, e.src, "->", e.dst, e.data, getattr(e.src,'code',None) and e.src.code.as_string)
                if e.data.data == 'a':
                    r = classify_tile_access(e.data.subset, iter_vars=(me.map.params[-1],), inner_sdfg=inner, state=s)
                    print("     classify a:", r.per_dim_kind)
    return orig(self, sdfg, pr)
wa.WidenAccesses.apply_pass = patched
from gprogs import *
n=29; rng = np.random.default_rng(0); perm = rng.permutation(n).astype(np.int32)
try:
    run(k_gc, dict(a=rng.random(2*n), idx=perm, b=np.zeros(n), N=n))
except Exception as ex: print("EXC", type(ex).__name__)
