"""argmax_probe cases with ArgMaxLift disabled (isolates the culprit)."""
import sys
from dace.transformation.passes.canonicalize import arg_max_lift
arg_max_lift.ArgMaxLift.apply_pass = lambda self, sdfg, res: None
import runpy
sys.argv = ['argmax_probe.py'] + sys.argv[1:]
runpy.run_path('argmax_probe.py', run_name='__main__')
