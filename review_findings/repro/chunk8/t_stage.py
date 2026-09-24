import numpy as np, dace, copy, sys, os
from drv import *
import hook
N = 13
@dace.program
def k1(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] + 1.0
        B[i] = B[i] * 3.0
        B[i] = B[i] - A[i]
        C[i] = B[i] * 2.0
def r1(A, B, C):
    B[:] = (A + 1) * 3 - A; C[:] = B * 2

@dace.program
def k2(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] + 1.0
        C[i] = B[i] * 2.0
        B[i] = 7.0
def r2(A, B, C):
    C[:] = (A + 1) * 2; B[:] = 7

@dace.program
def k3(A: dace.float64[N], B: dace.float64[N, 2], C: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i, 0] = A[i] + 1.0
        B[i, 1] = A[i] * 2.0
        C[i] = B[i, 0] + B[i, 1]
        B[i, 0] = B[i, 1] - 1.0
def r3(A, B, C):
    b0 = A + 1; b1 = A * 2; C[:] = b0 + b1; B[:, 0] = b1 - 1; B[:, 1] = b1

@dace.program
def k4(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] + 1.0
        if A[i] > 0.5:
            C[i] = B[i] * 2.0
        else:
            C[i] = B[i] * 4.0
def r4(A, B, C):
    B[:] = A + 1; C[:] = np.where(A > 0.5, B * 2, B * 4)

@dace.program
def k5(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        t = A[i] + 1.0
        B[i] = t
        C[i] = B[i] + t
        B[i] = C[i] * B[i]
def r5(A, B, C):
    t = A + 1; C[:] = 2 * t; B[:] = C * t

progs = {'k1': (k1, r1, (N,)), 'k2': (k2, r2, (N,)), 'k3': (k3, r3, (N, 2)), 'k4': (k4, r4, (N,)), 'k5': (k5, r5, (N,))}
if __name__ == '__main__':
  for name in sys.argv[1:]:
      prog, ref, bshape = progs[name]
      s = prog.to_sdfg(simplify=True)
      try:
          vec(s, **({'remainder_strategy': os.environ['STRAT']} if os.environ.get('STRAT') else {}))
      except Exception as e:
          print(name, 'vectorizer raised', type(e).__name__, str(e)[:300]); continue
      import os
      if os.environ.get('NORUN'): continue
      A = np.random.rand(N); B = np.zeros(bshape); C = np.zeros(N)
      Br = B.copy(); Cr = C.copy(); ref(A, Br, Cr)
      s(A=A, B=B, C=C)
      print(name, 'tiles', ntile(s), 'B ok', np.allclose(B, Br), 'C ok', np.allclose(C, Cr))
