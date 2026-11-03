#!/usr/bin/env bash
# Train + benchmark a list of configs, one after another. Resumable: finished runs are skipped.
#   PY=.venv/bin/python scripts/sweep.sh configs/ablation/*.json
set -euo pipefail
PY=${PY:-python}
mkdir -p runs
for cfg in "$@"; do
  name=$(basename "$cfg" .json)
  if [ -f "results/runs/$name.json" ]; then
    echo "[skip] $name (already benchmarked)"
    continue
  fi
  echo "[$(date +%T)] train $name"
  $PY -m llm.train --config "$cfg" --name "$name" > "runs/$name.train.log" 2>&1
  echo "[$(date +%T)] bench $name"
  $PY -m bench.evaluate --run "runs/$name" > "runs/$name.eval.log" 2>&1
done
$PY -m bench.leaderboard > /dev/null
echo "[$(date +%T)] sweep finished"
