#!/usr/bin/env bash
# One-shot reproducer for the Harbor LOCOMO parity numbers on this fork.
# Hardcodes the parity proxy URL + API key (per the user's request — note these
# will be public once this branch is pushed to a public fork).
#
# Usage:  bash scripts/run_harbor_parity.sh
set -euo pipefail

export OPENAI_BASE_URL="REMOVED-ENDPOINT"
export OPENAI_API_KEY="REMOVED-REVOKED-KEY"

: "${MODEL:=gpt-5-mini}"
# All questions per conversation in one call, to match the Harbor side, which
# hands the agent every question at once (no batching knob on the codex agent).
: "${BATCH_SIZE:=200}"
: "${RUNS:=3}"
: "${START_RUN:=1}"  # 1-indexed; bump to 2/3/... when extending a prior session

# Upstream's pinned requirements.txt has openai 0.28; we need v1+.
pip install -q -U 'openai>=1' tiktoken

mkdir -p harbor_parity_out

end_run=$((START_RUN + RUNS - 1))
for run in $(seq "${START_RUN}" "${end_run}"); do
  out_file="harbor_parity_out/run_${run}.json"
  echo "=== Upstream LOCOMO parity, model=${MODEL}, batch_size=${BATCH_SIZE}, run ${run} (${RUNS} requested, START_RUN=${START_RUN}) ==="
  rm -f "${out_file}"  # force fresh run; evaluate_qa.py merges with existing output otherwise
  python task_eval/evaluate_qa.py \
    --data-file data/locomo10.json \
    --out-file "${out_file}" \
    --model "${MODEL}" \
    --batch-size "${BATCH_SIZE}"
done

echo
echo "Done. Per-run predictions: harbor_parity_out/run_{${START_RUN}..${end_run}}.json"
echo "Per-run aggregate stats:    harbor_parity_out/run_{${START_RUN}..${end_run}}_stats.json"
