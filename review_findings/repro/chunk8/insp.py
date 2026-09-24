import sys, os
os.environ['NORUN']='1'
import t_stage, hook, dump
from drv import vec
name = sys.argv[1]
s = t_stage.progs[name][0].to_sdfg(simplify=True); vec(s)
print('BEFORE'); dump.dump_nested(hook.SNAP['stage_before']); print('AFTER'); dump.dump_nested(hook.SNAP['stage_after'])
