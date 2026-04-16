#!/usr/bin/env bash
# pod_watcher.sh — Background GPU pod idle watcher
# Polls RUNNING pods via SSH, writes events to .omc/gpu-exec/events.jsonl
# Calls RunPod GraphQL API directly (RUNPOD_API_KEY env var, inherited via nohup)
#
# Usage: pod_watcher.sh --pods "id1,id2" --interval N --idle-pct P --idle-checks C \
#                        --mode full|error-only [--expected-completion ISO8601]
#
# PID contract:
#   - On start: checks .omc/gpu-exec/watcher.pid
#     * PID alive → exit 1 (duplicate guard)
#     * PID file exists but process dead → orphan cleanup + WATCHER_RESTART event + continue
#   - On stop (GraphQL stop failure after retry): ERROR event + exit 1 + pid cleanup

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PODS=""
INTERVAL=5
IDLE_PCT=20
IDLE_CHECKS=2
MODE="full"
EXPECTED_COMPLETION=""

# ── Arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pods)                PODS="$2";                shift 2 ;;
    --interval)            INTERVAL="$2";            shift 2 ;;
    --idle-pct)            IDLE_PCT="$2";            shift 2 ;;
    --idle-checks)         IDLE_CHECKS="$2";         shift 2 ;;
    --mode)                MODE="$2";                shift 2 ;;
    --expected-completion) EXPECTED_COMPLETION="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PODS" ]]; then
  echo "ERROR: --pods required" >&2
  exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
mkdir -p .omc/gpu-exec
PID_FILE=".omc/gpu-exec/watcher.pid"
EVENTS_FILE=".omc/gpu-exec/events.jsonl"
SSH_KEY="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
API_URL="https://api.runpod.io/graphql"

# ── PID guard ─────────────────────────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID=$(cat "$PID_FILE" 2>/dev/null || true)
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "ERROR: Watcher already running (PID $EXISTING_PID). Call stop_watching_pods first." >&2
    exit 1
  else
    # Orphan PID — clean up and log WATCHER_RESTART event
    rm -f "$PID_FILE"
    echo "{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"event\":\"WATCHER_RESTART\",\"reason\":\"orphan_cleaned\",\"gapNote\":\"poll gap possible during restart\"}" >> "$EVENTS_FILE"
  fi
fi

# Write our PID
echo $$ > "$PID_FILE"

# Cleanup on exit
cleanup() {
  rm -f "$PID_FILE"
}
trap cleanup EXIT

# ── Helpers ───────────────────────────────────────────────────────────────────
log_event() {
  local pod_id="$1"
  local event="$2"
  local gpu_pct="${3:-0}"
  local extra="${4:-}"
  local ts
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local line="{\"ts\":\"$ts\",\"podId\":\"$pod_id\",\"event\":\"$event\",\"gpuPct\":$gpu_pct"
  [[ -n "$extra" ]] && line="$line,$extra"
  line="$line}"
  echo "$line" >> "$EVENTS_FILE"
}

# Get pod SSH details via RunPod API
get_pod_ssh() {
  local pod_id="$1"
  local query="{ pod(input: {podId: \"$pod_id\"}) { publicIp portMappings } }"
  local resp
  resp=$(curl -sf -X POST "$API_URL?api_key=${RUNPOD_API_KEY:-}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$query\"}" 2>/dev/null) || return 1
  local ip port
  ip=$(echo "$resp" | grep -o '"publicIp":"[^"]*"' | cut -d'"' -f4)
  port=$(echo "$resp" | grep -o '"22":[0-9]*' | cut -d: -f2)
  [[ -z "$ip" || -z "$port" ]] && return 1
  echo "$ip $port"
}

# Query GPU utilization via SSH
get_gpu_util() {
  local ip="$1"
  local port="$2"
  local util
  util=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
    -i "$SSH_KEY" -p "$port" "root@$ip" \
    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader" 2>/dev/null | head -1 | tr -d ' %') || return 1
  echo "${util:-0}"
}

# GraphQL stop mutation with 1 retry
stop_pod_graphql() {
  local pod_id="$1"
  local mutation="mutation { podStop(input: {podId: \"$pod_id\"}) { id desiredStatus } }"
  local resp exit_code

  resp=$(curl -sf -X POST "$API_URL?api_key=${RUNPOD_API_KEY:-}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$mutation\"}" 2>/dev/null)
  exit_code=$?

  if [[ $exit_code -eq 0 ]] && echo "$resp" | grep -q '"podStop"'; then
    return 0
  fi

  # 1 retry after 2s
  sleep 2
  resp=$(curl -sf -X POST "$API_URL?api_key=${RUNPOD_API_KEY:-}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$mutation\"}" 2>/dev/null)
  exit_code=$?

  if [[ $exit_code -eq 0 ]] && echo "$resp" | grep -q '"podStop"'; then
    return 0
  fi

  return 1
}

# Compute sleep interval (reduce to 1min if within 30min of expected completion)
compute_interval() {
  if [[ -z "$EXPECTED_COMPLETION" ]]; then
    echo "$INTERVAL"
    return
  fi
  local now target diff
  now=$(date +%s)
  target=$(date -d "$EXPECTED_COMPLETION" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$EXPECTED_COMPLETION" +%s 2>/dev/null || echo 0)
  if [[ "$target" -eq 0 ]]; then
    echo "$INTERVAL"
    return
  fi
  diff=$(( target - now ))
  if [[ $diff -le 1800 && $diff -ge 0 ]]; then
    echo 1  # 1-minute interval within 30 min of completion
  else
    echo "$INTERVAL"
  fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────
IFS=',' read -ra POD_IDS <<< "$PODS"
declare -A IDLE_COUNTER

# Initialize counters
for pod_id in "${POD_IDS[@]}"; do
  IDLE_COUNTER["$pod_id"]=0
done

# First poll immediately (no initial sleep — reduce restart gap)
FIRST_POLL=true

while true; do
  if [[ "$FIRST_POLL" == "true" ]]; then
    FIRST_POLL=false
  else
    SLEEP_MIN=$(compute_interval)
    sleep $(( SLEEP_MIN * 60 ))
  fi

  for pod_id in "${POD_IDS[@]}"; do
    ssh_info=$(get_pod_ssh "$pod_id") || {
      # Pod not reachable — skip this cycle
      continue
    }
    ip=$(echo "$ssh_info" | cut -d' ' -f1)
    port=$(echo "$ssh_info" | cut -d' ' -f2)

    gpu_pct=$(get_gpu_util "$ip" "$port") || {
      gpu_pct=0
    }

    # Idle check
    if [[ "$gpu_pct" -lt "$IDLE_PCT" ]]; then
      IDLE_COUNTER["$pod_id"]=$(( IDLE_COUNTER["$pod_id"] + 1 ))
      count=${IDLE_COUNTER["$pod_id"]}

      log_event "$pod_id" "IDLE_WARNING" "$gpu_pct" "\"idleCheck\":$count"

      if [[ $count -ge $IDLE_CHECKS ]]; then
        # Attempt stop
        if stop_pod_graphql "$pod_id"; then
          log_event "$pod_id" "AUTO_STOPPED" "$gpu_pct" "\"idleCheck\":$count"
          # Remove from tracking
          IDLE_COUNTER["$pod_id"]=0
        else
          # Stop failed after retry — log ERROR and exit
          log_event "$pod_id" "ERROR" "$gpu_pct" "\"detail\":\"GraphQL stop failed after retry — manual intervention required\""
          echo "{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"event\":\"WATCHER_EXITED\",\"reason\":\"stop_failure\",\"podId\":\"$pod_id\"}" >> "$EVENTS_FILE"
          rm -f "$PID_FILE"
          exit 1
        fi
      fi
    else
      # Reset idle counter on activity
      IDLE_COUNTER["$pod_id"]=0
      if [[ "$MODE" == "full" ]]; then
        log_event "$pod_id" "HEALTH_CHECK" "$gpu_pct"
      fi
      # error-only mode: skip HEALTH_CHECK events entirely
    fi
  done
done
