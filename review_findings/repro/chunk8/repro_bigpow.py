"""PowerOperatorExpansion unrolls a literal integer exponent into a left-nested product with no cap; x ** 5000 overflows the AST unparser."""
import numpy as np, dace
from drv import *
N = 16
@dace.program
def f(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] ** 5000
s = f.to_sdfg(simplify=True)
try:
    vec(s); print("tile ops", ntile(s))
except BaseException as e:
    print("vectorizer raised", type(e).__name__, str(e)[:200])
