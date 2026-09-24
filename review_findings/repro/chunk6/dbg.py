exec(open('fma_cross_region.py').read().split('sdfg = build()')[0])
from dace.transformation.passes.vectorization.fuse_multiply_add import _binop_tasklet
from dace.transformation.passes.vectorization.utils.tasklets import is_vectorizable_tasklet, is_python_tasklet
s = build(); st = s.start_state
for n in st.nodes():
    if isinstance(n, dace.nodes.Tasklet): print(n, is_vectorizable_tasklet(st, n), is_python_tasklet(n), _binop_tasklet(n,'*'), _binop_tasklet(n,'+'))
print(s.states())
