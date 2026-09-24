import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.transformation import pass_pipeline as ppl
s = dace.SDFG.from_file('ib_canon.sdfg')
def libs(s, tag):
    for n, g in s.all_nodes_recursive():
        if isinstance(n, (dace.nodes.LibraryNode,)):
            print(tag, type(n).__name__, n.label, getattr(n, 'wcr', None), getattr(n,'axes',None), [(e.dst_conn, str(e.data)) for e in g.in_edges(n)], [(e.src_conn, str(e.data)) for e in g.out_edges(n)])
        if isinstance(n, dace.nodes.AccessNode):
            for e in g.out_edges(n):
                if isinstance(e.dst, dace.nodes.AccessNode): print(tag, 'AN->AN', n.data, e.dst.data, str(e.data), e.data.wcr)
libs(s, 'CANON')
v = VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR, remainder_strategy='masked_tail', expand_tile_nodes=False))
# instrument passes
for p in v.passes:
    orig = p.apply_pass
    def mk(orig, name):
        def f(sdfg, res):
            r = orig(sdfg, res)
            print('--- after', name); libs(sdfg, '   ')
            return r
        return f
    p.apply_pass = mk(orig, type(p).__name__)
v.apply_pass(s, {})
libs(s, 'FINAL')
