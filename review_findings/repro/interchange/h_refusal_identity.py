"""For every program in the relevant test modules: can_fission never mutates, and fission/push/hoist
returning False leave the SDFG JSON-identical."""
import copy, importlib.util, json, sys, glob, traceback
import dace
from dace.frontend.python.parser import DaceProgram
from dace.sdfg.state import LoopRegion, ConditionalBlock
from dace.transformation.passes.loop_fission import LoopFission
from dace.transformation.passes.move_if_into_loop import MoveIfIntoLoop
from dace.transformation.interstate.move_loop_invariant_if_up import MoveLoopInvariantIfUp

def js(s):
    return json.dumps(s.to_json(), sort_keys=True, default=str)

files = sys.argv[1:]
progs = []
for f in files:
    spec = importlib.util.spec_from_file_location('m' + str(len(progs)), f)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:
        print('import fail', f, e); continue
    progs += [(f.split('/')[-1], k, v) for k, v in vars(m).items() if isinstance(v, DaceProgram)]
print(len(progs), 'programs')
n = {'can': 0, 'fis': 0, 'push': 0, 'hoist': 0}
for fname, name, p in progs:
    for simp in (True, False):
        try:
            sdfg = p.to_sdfg(simplify=simp)
        except Exception as e:
            continue
        regions = [r for r in sdfg.all_control_flow_regions(recursive=True)]
        for idx in range(len(regions)):
            for op in ('can', 'fis', 'push', 'hoist'):
                g = copy.deepcopy(sdfg)
                r = list(g.all_control_flow_regions(recursive=True))[idx]
                before = js(g)
                try:
                    if op == 'can' and isinstance(r, LoopRegion):
                        res = LoopFission.can_fission(r); must = True
                    elif op == 'fis' and isinstance(r, LoopRegion):
                        res = LoopFission.fission(r); must = not res
                    elif op == 'push' and isinstance(r, ConditionalBlock):
                        res = MoveIfIntoLoop.push(r); must = not res
                    elif op == 'hoist' and isinstance(r, LoopRegion):
                        res = MoveLoopInvariantIfUp.hoist(r); must = not res
                    else:
                        continue
                except Exception as e:
                    print('EXC', fname, name, simp, op, r.label, type(e).__name__, e); continue
                n[op] += 1
                if must and js(g) != before:
                    print('MUTATED', fname, name, simp, op, r.label, 'returned', res)
                if not must:
                    try:
                        g.validate()
                    except Exception as e:
                        print('INVALID after', op, fname, name, simp, r.label, type(e).__name__, str(e)[:150])
print(n)
