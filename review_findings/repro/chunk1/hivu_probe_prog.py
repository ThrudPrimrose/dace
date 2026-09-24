import dace
N = dace.symbol('N', dtype=dace.int64)


@dace.program
def cross(y: dace.float64[1], x: dace.float64[1], c: dace.float64[N]):
    for i in range(N):
        x[0] = y[0] * 0.5
        y[0] = c[i]
