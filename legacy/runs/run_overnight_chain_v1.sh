#!/bin/bash
# Overnight chain: run a sequence of run_experiment.py configs sequentially.
# Each entry runs to completion (success or fail) before the next starts.
# Failures are logged but don't abort the chain — we want every experiment
# attempted before morning.
#
# Usage:
#   nohup bash experiment_v2/scripts/run_overnight_chain.sh > experiment_v2/logs/overnight_master_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -u  # unset vars are errors; do NOT use -e (we want to continue past failures)

cd /workspace/thesis-llm-alignment
source env.sh

CONFIGS=(
  "experiment_v2/configs/v2_track_dialog_nojudge.json"
  "experiment_v2/configs/v2_sum_baseline_pairs_nojudge.json"
  "experiment_v2/configs/v2_track_summarization_seed1337_nojudge.json"
)

STATUS_FILE="experiment_v2/logs/overnight_chain_status.json"

mkdir -p experiment_v2/logs

echo "=== overnight chain starting at $(date) ==="
echo "configs:"
for c in "${CONFIGS[@]}"; do echo "  - $c"; done
echo

# Initialise status file.
.venv/bin/python -c "
import json, os
items = [{'config': c, 'status': 'pending', 'started_at': None, 'finished_at': None,
         'exit_code': None, 'exp_dir': None} for c in $(printf '%s\n' "${CONFIGS[@]}" | .venv/bin/python -c "import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))")]
json.dump({'started_at': '$(date -Iseconds)', 'items': items}, open('$STATUS_FILE','w'), indent=2)
"

for IDX in "${!CONFIGS[@]}"; do
  CFG="${CONFIGS[$IDX]}"
  NAME=$(basename "$CFG" .json)
  TS=$(date +%Y%m%d_%H%M%S)
  RUN_LOG="experiment_v2/logs/overnight_${NAME}_${TS}.log"

  echo
  echo "=== [$(date)] starting #$((IDX+1))/${#CONFIGS[@]}: $CFG ==="
  echo "log: $RUN_LOG"

  # Update status: started.
  .venv/bin/python -c "
import json
s = json.load(open('$STATUS_FILE'))
s['items'][$IDX]['status'] = 'running'
s['items'][$IDX]['started_at'] = '$(date -Iseconds)'
s['items'][$IDX]['log_path'] = '$RUN_LOG'
json.dump(s, open('$STATUS_FILE','w'), indent=2)
"

  # Run.
  .venv/bin/python scripts/run_experiment.py --config "$CFG" > "$RUN_LOG" 2>&1
  EC=$?

  # Try to find the exp dir we just created.
  EXP_DIR=$(ls -td results/experiments/${NAME}_* 2>/dev/null | head -1)

  # Update status: finished.
  .venv/bin/python -c "
import json
s = json.load(open('$STATUS_FILE'))
s['items'][$IDX]['status'] = 'completed' if $EC == 0 else 'failed'
s['items'][$IDX]['finished_at'] = '$(date -Iseconds)'
s['items'][$IDX]['exit_code'] = $EC
s['items'][$IDX]['exp_dir'] = '$EXP_DIR'
json.dump(s, open('$STATUS_FILE','w'), indent=2)
"

  if [ $EC -eq 0 ]; then
    echo "=== [$(date)] OK: $NAME (exp dir: $EXP_DIR) ==="
  else
    echo "=== [$(date)] FAILED: $NAME (exit=$EC) — see $RUN_LOG ==="
    echo "  continuing to next config so the rest still runs"
  fi
done

echo
echo "=== overnight chain finished at $(date) ==="
.venv/bin/python -c "
import json
s = json.load(open('$STATUS_FILE'))
for i, it in enumerate(s['items'], 1):
    print(f'  {i}. {it[\"status\"]:10s} ec={it[\"exit_code\"]} {it[\"config\"]}')
"
