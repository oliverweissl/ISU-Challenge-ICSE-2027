#!/usr/bin/env bash
# Periodically launch the `notekeeper` agent (headless) to refresh methodology.md and setup.md.
# Called from a PostToolUse hook. Cheap no-op unless: interval elapsed AND the repo state changed.
# Tunables: NOTEKEEPER_INTERVAL (seconds, default 1800), NOTEKEEPER_DISABLE=1 to switch off.

[ -n "$NOTEKEEPER_RUNNING" ] && exit 0   # never recurse from the agent's own tool calls
[ -n "$NOTEKEEPER_DISABLE" ] && exit 0

ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -z "$ROOT" ] && exit 0
cd "$ROOT" || exit 0

STATE_DIR="$ROOT/.claude/.notekeeper"
mkdir -p "$STATE_DIR"
INTERVAL="${NOTEKEEPER_INTERVAL:-1800}"
NOW=$(date +%s)

# running already?
if [ -f "$STATE_DIR/lock" ] && kill -0 "$(cat "$STATE_DIR/lock" 2>/dev/null)" 2>/dev/null; then exit 0; fi

# interval throttle
LAST=$(cat "$STATE_DIR/last_run" 2>/dev/null || echo 0)
[ $((NOW - LAST)) -lt "$INTERVAL" ] && exit 0

# change detection: tracked/untracked files (excluding the notes themselves) + run configs/reports
FP=$( { git status --porcelain -- . ':!methodology.md' ':!setup.md' ':!.claude/.notekeeper' 2>/dev/null
        find track_a/submissions track_a/track_a_evaluation_output -maxdepth 3 \( -name config.json -o -name report.json \) \
             -printf '%p %T@\n' 2>/dev/null | sort
        find track_a/solution -maxdepth 2 -name '*.py' -printf '%p %T@\n' 2>/dev/null | sort
      } | sha1sum | cut -d' ' -f1)
[ "$FP" = "$(cat "$STATE_DIR/last_fp" 2>/dev/null)" ] && exit 0

echo "$NOW" > "$STATE_DIR/last_run"
echo "$FP" > "$STATE_DIR/last_fp"

PROMPT="Update methodology.md and setup.md now, following your instructions. Inspect current repo state, run configs and evaluation reports. Today is $(date +%F)."

(
  export NOTEKEEPER_RUNNING=1
  nohup claude -p "$PROMPT" --agent notekeeper --permission-mode acceptEdits \
        --allowedTools "Read" "Write" "Edit" "Glob" "Grep" "Bash(git status:*)" "Bash(git diff:*)" "Bash(git log:*)" "Bash(ls:*)" "Bash(cat:*)" \
        --no-session-persistence >> "$STATE_DIR/notekeeper.log" 2>&1 &
  echo $! > "$STATE_DIR/lock"
) >/dev/null 2>&1

exit 0
