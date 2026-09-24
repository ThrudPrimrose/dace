#!/bin/bash
source /home/user/.venv/bin/activate
cd /tmp/claude-0/-home-user/1191ce82-d309-5728-89cd-bb12c458f12b/scratchpad/review/chunk7
PYTHONHASHSEED=${SEED:-0} timeout ${TO:-900} python "$@" 2>&1 | grep -v "Warning: Host\|warnings.warn"
