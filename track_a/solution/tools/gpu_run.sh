#!/usr/bin/env bash
# Exclusive-GPU job wrapper: serialises all our jobs per GPU so concurrent agents never OOM each other.
# usage: tools/gpu_run.sh [--min-free MiB] [--log FILE] -- <command...>
#   Waits for a free slot (flock per GPU) AND >= min-free MiB actually free (other users), then runs the
#   command with CUDA_VISIBLE_DEVICES=<slot>; the command sees one GPU as cuda:0. Lock released on exit/crash.
#   Default min-free 30000. Slot choice: GPU with the most free memory among unlocked ones.
MINFREE=30000; LOG=""
while [ $# -gt 0 ]; do case "$1" in
  --min-free) MINFREE=$2; shift 2;; --log) LOG=$2; shift 2;; --) shift; break;; *) break;; esac; done
LOCKDIR=/tmp/isu_gpu_locks; mkdir -p $LOCKDIR
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
while true; do
  for idx in $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | cut -d, -f1); do
    free=$(nvidia-smi -i $idx --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$free" -lt "$MINFREE" ] && continue
    exec {fd}>$LOCKDIR/gpu$idx.lock
    if flock -n $fd; then
      echo "[gpu_run] GPU $idx (free ${free} MiB): $*" >&2
      export CUDA_VISIBLE_DEVICES=$idx PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      if [ -n "$LOG" ]; then "$@" >> "$LOG" 2>&1; else "$@"; fi
      rc=$?; echo "[gpu_run] exit $rc" >&2; exit $rc
    fi
    exec {fd}>&-
  done
  sleep 20
done
