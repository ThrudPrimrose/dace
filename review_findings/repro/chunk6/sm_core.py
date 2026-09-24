import numpy as np
import dace
from dace.sdfg import infer_types
s = dace.SDFG.from_file('ib_canon.sdfg')
s.name = 'ib_core_only'
infer_types.set_default_schedule_and_storage_types(s, None)
print({k: str(v.storage) for k, v in s.arrays.items()})
m, nn = 5, 12
A = np.random.rand(m, nn); out = np.zeros(m)
s(cnt=np.array([7], np.int32), A=A, out=out, N=nn, M=m)
print('err', np.abs(out - A[:, :7].sum(1)).max())
