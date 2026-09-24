"""Probe ArgMaxLift through the full canonicalize pipeline on argmax/argmin variants."""
import sys
import numpy as np
import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def p_const_seed(a: dace.float64[N], out: dace.float64[1]):
    x = 0.5
    for i in range(N):
        if a[i] > x:
            x = a[i]
    out[0] = x


def r_const_seed(a):
    x = 0.5
    for v in a:
        if v > x:
            x = v
    return np.array([x])


@dace.program
def p_idx_high_seed(a: dace.float64[N], out: dace.float64[2]):
    x = 2.0
    idx = -1
    for i in range(N):
        if a[i] > x:
            x = a[i]
            idx = i
    out[0] = x
    out[1] = idx


def r_idx_high_seed(a):
    x, idx = 2.0, -1
    for i, v in enumerate(a):
        if v > x:
            x, idx = v, i
    return np.array([x, idx])


@dace.program
def p_s314_from1(a: dace.float64[N], out: dace.float64[1]):
    x = a[0]
    for i in range(1, N):
        if a[i] > x:
            x = a[i]
    out[0] = x


def r_s314_from1(a):
    x = a[0]
    for v in a[1:]:
        if v > x:
            x = v
    return np.array([x])


@dace.program
def p_s315(a: dace.float64[N], out: dace.float64[2]):
    x = a[0]
    index = 0
    for i in range(N):
        if a[i] > x:
            x = a[i]
            index = i
    out[0] = x
    out[1] = index


def r_s315(a):
    x, index = a[0], 0
    for i, v in enumerate(a):
        if v > x:
            x, index = v, i
    return np.array([x, index])


@dace.program
def p_s315_ge(a: dace.float64[N], out: dace.float64[2]):
    x = a[0]
    index = 0
    for i in range(N):
        if a[i] >= x:
            x = a[i]
            index = i
    out[0] = x
    out[1] = index


def r_s315_ge(a):
    x, index = a[0], 0
    for i, v in enumerate(a):
        if v >= x:
            x, index = v, i
    return np.array([x, index])


@dace.program
def p_pred(a: dace.float64[N], out: dace.float64[1]):
    j = -1
    for i in range(N):
        if a[i] < 0.3:
            j = i
    out[0] = j


def r_pred(a):
    j = -1
    for i, v in enumerate(a):
        if v < 0.3:
            j = i
    return np.array([j])


@dace.program
def p_offset_seed(a: dace.float64[N], out: dace.float64[1]):
    x = a[3]
    for i in range(N):
        if a[i] > x:
            x = a[i]
    out[0] = x


def r_offset_seed(a):
    x = a[3]
    for v in a:
        if v > x:
            x = v
    return np.array([x])


CASES = {
    'const_seed': (p_const_seed, r_const_seed, 1),
    'idx_high_seed': (p_idx_high_seed, r_idx_high_seed, 2),
    's314_from1': (p_s314_from1, r_s314_from1, 1),
    's315': (p_s315, r_s315, 2),
    's315_ge': (p_s315_ge, r_s315_ge, 2),
    'pred': (p_pred, r_pred, 1),
    'offset_seed': (p_offset_seed, r_offset_seed, 1),
}

if __name__ == '__main__':
    names = sys.argv[1:] or list(CASES)
    for name in names:
        prog, ref, nout = CASES[name]
        sdfg = canonicalize(prog.to_sdfg())
        libs = sorted({type(n).__name__ for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)})
        loops = sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if type(r).__name__ == 'LoopRegion')
        csdfg = sdfg.compile()
        bad = []
        for n in ((8, 5) if name == "offset_seed" else (8, 5, 1)):
            for trial in range(3):
                rng = np.random.default_rng(trial)
                a = rng.random(n)
                if trial == 2:
                    a = np.round(a * 2) / 2  # ties
                out = np.zeros(nout)
                csdfg(a=a.copy(), out=out, N=n)
                exp = ref(a)
                if not np.allclose(out, exp):
                    bad.append(f'N={n} trial={trial} a={np.round(a, 2).tolist()} got={out.tolist()} exp={exp.tolist()}')
        print(f'{name}: libnodes={libs} loops={loops} ->', 'ok' if not bad else 'MISMATCH')
        for b in bad[:4]:
            print('    ', b)
