#!/bin/bash
# wait for pid $1 to exit, then run_refs for remaining names
pid=$1; shift
while kill -0 $pid 2>/dev/null; do sleep 15; done
exec ./run_refs.sh "$@"
