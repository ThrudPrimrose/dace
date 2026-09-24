"""A float literal meeting an INT operand (int symbol, int scalar, or the int64 lane-id tile) is lowered
by ConvertTaskletsToTileOps to a TileBinop without a dtype check; the TileBinop expansion casts the
literal to the int operand dtype: ``(int)(0.25) * (int)(N)`` == 0, ``(int64_t)0.5 * lane`` == 0."""
import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')


@dace.program
def scale_by_quarter_n(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * (N / 4)


@dace.program
def shift_by_half_n_plus_one(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + (N + 1) * 0.5


@dace.program
def ramp_half(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + i * 0.5


@dace.program
def ramp_only(C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = i * 0.5


n = 21
rem = sys.argv[1] if len(sys.argv) > 1 else 'scalar_postamble'
A = np.linspace(0.0, 1.0, n)
for prog, expect in ((scale_by_quarter_n, A * (n / 4)), (shift_by_half_n_plus_one, A + (n + 1) * 0.5),
                     (ramp_half, A + np.arange(n) * 0.5), (ramp_only, np.arange(n) * 0.5)):
    print(f'== {prog.name}  numpy: {expect[:4]}')
    arrays = dict(C=np.zeros(n)) if prog is ramp_only else dict(A=A, C=np.zeros(n))
    run(prog, f'{prog.name}_{rem}', arrays, dict(N=n), remainder=rem, exact=False, lift_copy=False)
