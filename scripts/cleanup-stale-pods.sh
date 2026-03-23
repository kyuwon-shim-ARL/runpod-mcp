#!/usr/bin/env bash
# Cleanup stale RunPod pods — delete EXITED pods idle longer than GRACE_HOURS.
# Usage: RUNPOD_API_KEY=rp_xxx ./cleanup-stale-pods.sh
# Cron:  0 * * * * RUNPOD_API_KEY=rp_xxx /path/to/cleanup-stale-pods.sh >> /var/log/runpod-cleanup.log 2>&1

set -euo pipefail

GRACE_HOURS="${GRACE_HOURS:-2}"
API_KEY="${RUNPOD_API_KEY:?RUNPOD_API_KEY not set}"
BASE_URL="https://api.runpod.io/v2/pods"
SKIP_PATTERN="keep|persist"
LOG_PREFIX="[runpod-cleanup $(date -Iseconds)]"

pods_json=$(curl -sf -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}" 2>/dev/null) || {
  echo "${LOG_PREFIX} ERROR: Failed to fetch pods"
  exit 1
}

grace_seconds=$((GRACE_HOURS * 3600))
now_epoch=$(date +%s)
deleted=0
skipped=0

echo "${LOG_PREFIX} Checking pods (grace=${GRACE_HOURS}h)..."

echo "${pods_json}" | jq -c '.[]' 2>/dev/null | while read -r pod; do
  id=$(echo "${pod}" | jq -r '.id')
  name=$(echo "${pod}" | jq -r '.name // "unnamed"')
  status=$(echo "${pod}" | jq -r '.desiredStatus // "UNKNOWN"')
  last_change=$(echo "${pod}" | jq -r '.lastStatusChange // empty')

  # Only target EXITED pods
  if [[ "${status}" != "EXITED" ]]; then
    continue
  fi

  # Skip pods with keep/persist in name
  if echo "${name}" | grep -qiE "${SKIP_PATTERN}"; then
    echo "${LOG_PREFIX} SKIP: ${name} (${id}) — name matches keep/persist"
    continue
  fi

  # Skip if no timestamp
  if [[ -z "${last_change}" ]]; then
    echo "${LOG_PREFIX} SKIP: ${name} (${id}) — no lastStatusChange"
    continue
  fi

  # Calculate idle time
  change_epoch=$(date -d "${last_change}" +%s 2>/dev/null) || {
    echo "${LOG_PREFIX} SKIP: ${name} (${id}) — cannot parse timestamp: ${last_change}"
    continue
  }
  idle_seconds=$((now_epoch - change_epoch))

  if [[ ${idle_seconds} -lt ${grace_seconds} ]]; then
    idle_hours=$(echo "scale=1; ${idle_seconds}/3600" | bc)
    echo "${LOG_PREFIX} SKIP: ${name} (${id}) — idle ${idle_hours}h < grace ${GRACE_HOURS}h"
    continue
  fi

  idle_hours=$(echo "scale=1; ${idle_seconds}/3600" | bc)

  # Delete the pod
  http_code=$(curl -sf -o /dev/null -w "%{http_code}" -X DELETE \
    -H "Authorization: Bearer ${API_KEY}" \
    "${BASE_URL}/${id}" 2>/dev/null) || http_code="ERR"

  if [[ "${http_code}" == "200" || "${http_code}" == "204" ]]; then
    echo "${LOG_PREFIX} DELETED: ${name} (${id}) — idle ${idle_hours}h"
  else
    echo "${LOG_PREFIX} FAILED: ${name} (${id}) — HTTP ${http_code}"
  fi
done

echo "${LOG_PREFIX} Done."
