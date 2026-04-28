# runpod-mcp: Claude Code Project Instructions

## COST SAFETY RULES (최우선 적용 — 다른 규칙보다 상위)

GPU 시간 = 돈. 훈련에만 써야 한다.

### 팟 생성 순서 (반드시 지킬 것)

```
1. 로컬 서버에서 전처리 완료  ← GPU 불필요한 작업 전부 여기서
2. 전송 전용 팟 생성 (1-GPU 최저가, ~$0.1-0.6/hr) → 데이터 전송 → 팟 삭제
   ├─ 데이터 > 50GB: containerDiskInGb=200, rootfs에 직접 (NV quota trap 회피)
   └─ 데이터 ≤ 50GB: NV 50GB 생성 → 팟 마운트 → 전송 → 팟만 삭제, NV 보존
3. 로컬 or 1-GPU로 검증 테스트 (VRAM, 속도, 코드 정상 동작 확인)
4. 검증 통과 후 실제 GPU 팟 생성 (4-GPU 등 고비용)
5. 훈련 → 결과 수집 → 팟 즉시 삭제 (stop 금지 — 과금 계속됨)
```

### 절대 금지 (위반 시 즉시 수정)

- **4-GPU 팟(gpuCount ≥ 2)에서 데이터 전송** — $1.36/hr으로 9시간 전송 = $12 낭비 (T14/T15 사고 이력)
- **GPU 팟 먼저 생성 후 데이터 준비** — idle 과금 (T3 사고 이력)
- **pod stop** — 항상 delete (stop은 과금 계속됨)
- **검증 없이 바로 4-GPU 팟** — 코드 버그로 고비용 팟 낭비

### costSafetyConfirmed 사용 규칙

`create_pod_auto`에서 gpuCount >= 2 + dryRun: false 호출 시 `costSafetyConfirmed: true` 없으면 사전 차단됨.
- `costSafetyConfirmed: true`는 **사용자가 체크리스트를 직접 확인한 후 명시적으로 재호출을 요청할 때만** 전달.
- **Claude가 사용자 동의 없이 자동으로 true를 설정하여 차단을 우회하는 것은 엄격히 금지.**

### MIGRATION_THRESHOLD (gpu_cost_compare 결과 기준)

- 온디맨드 기준 **20%+ 저렴한 대안** → 마이그레이션 강력 권유 (스팟 제외 — NEVER spot 규칙)
- 근거: Pod 삭제+재생성 소요 ~15분 = 전환 비용 약 $0.1-0.3. 20% 미만 절약이면 전환 비용이 절약분을 상쇄.
- 성능 차이(속도)와 데이터 이전 비용을 함께 고지. 최종 결정은 사용자.

### 예외 (사용자가 명시적으로 요청한 경우만)

사용자가 "바로 4-GPU로 해줘" 등을 명시하면: 위험을 경고한 뒤 진행. 경고 없이 진행 금지.

---

## GPU Optimization Workflow

When a user is doing ML training on RunPod, follow this optimization pattern:

### Tool Sequencing
**팟 생성 전 (rootfs 사이징 필수):**
랜덤 액세스 훈련(이미지/molecule/셔플 데이터)은 NV가 rootfs보다 **~18× 느리다** (e049 측정: 43 vs 775 files/sec). NV에 데이터를 두고 훈련 돌리면 epoch I/O가 10분+ 걸린다. 데이터를 rootfs로 복사 후 훈련해야 한다. **rootfs는 팟 생성 후 절대 못 늘린다** → `plan_gpu_job` 호출 시 `randomAccessTrainingGb` 전달, 권장 `containerDiskInGb` 받아서 사용.

```
plan_gpu_job(randomAccessTrainingGb=N) → containerDiskInGb 권장값 출력
→ create_pod_auto(containerDiskInGb=권장값) — 랜덤 액세스 훈련은 데이터 크기*1.3 + 30GB 시스템 오버헤드 필요
```

**팟 생성 후 워크플로우:**
1. `create_pod_auto` → `wait_for_pod` → `upload_files` (NV로)
   → `execute_ssh_command` (setup: apt install, pip install 등)
   → `execute_ssh_command` (`mkdir -p /root/data && cp -r --reflink=auto /workspace/<dataset> /root/data/`) — NV→rootfs 복사 (랜덤 액세스 훈련 시 필수)
   → **`run_preflight(trainDataPath="/root/data/<dataset>", expectedRandomAccessGb=N)`** ← CUDA + NV→rootfs 위치 + 사이즈 검증 (필수)
   → `execute_ssh_command` (training launch — 스크립트가 `/root/data/<dataset>` 읽도록 설정)
2. Training launch 직후 **`gpu_sample_burst`** 호출 (필수)
   - OMC 환경: `ScheduleWakeup(delaySeconds=120)` 설정 → wakeup 시 `gpu_sample_burst` 호출
   - 직접(OMC 없는 환경): `execute_ssh_command("sleep 120")` 완료 후 `gpu_sample_burst` 호출
   → `CONSISTENTLY_IDLE` → 즉시 중단 + 로그 확인 (CPU fallback 또는 NV 스트리밍 의심)
   → `STABLE_OPTIMAL` / `IMPROVING` → `watch_running_pods`로 모니터링 인계

**예외 (run_preflight allowNvStreaming:true):** 훈련이 순수 sequential read만 한다면 (드물다 — 대부분의 ML은 random shuffle) NV 직접 사용 가능. 이 경우만 opt-out.
3. If underutilized, call `gpu_health_check` with `perSampleMb` for batch size recommendation
4. Call `gpu_cost_compare` if GPU is underutilized to find cheaper alternatives
5. **Re-call `gpu_health_check` every 5-10 min during long training runs** to detect degradation or idle drift

### Critical Rules
- **NEVER use spot instances** — always use on-demand (`spot: false`, which is the default). Spot pods can be preempted at any time and there is NO automatic backup or checkpoint-on-preemption mechanism in this MCP server. If the user explicitly insists on spot, warn them that data on container disk will be lost on preemption and require checkpointing to a network volume.
- **Background long-running commands**: `execute_ssh_command` blocks the MCP server (spawnSync). Use `nohup cmd > /workspace/log 2>&1 &` and poll with `tail`
- **Never auto-adjust batch size or migrate GPUs** — always present recommendations and let the user decide
- **Overprovisioning**: If `create_pod_auto` reports overprovisioning, flag it to the user
- **Always recommend `optimizePytorch: true`** for PyTorch workloads — enables `expandable_segments:True` which reduces VRAM fragmentation significantly
- **Proactively suggest network volumes** when the user describes iterative experiments, repeated fine-tuning, or datasets > 1GB — avoids re-uploading data on every pod

### Proactive GPU Management Protocol

Before and during GPU work, follow this decision tree to maximize VRAM utilization and minimize idle time:

```
User requests GPU work
│
├─ Data upload needed (>500MB)?
│   ├─ YES → Use Staging Pod Pattern (see below)
│   └─ NO  → Direct upload on GPU pod is OK
│
├─ Iterative experiments (multiple runs)?
│   ├─ YES → Create network volume first, upload once
│   └─ NO  → Container volume is fine
│
├─ After training starts (1-2 min):
│   └─ Call gpu_health_check
│       ├─ IDLE (<30% VRAM)
│       │   └─ Check: did training actually start? Check logs.
│       │       ├─ Not started → Fix launch command
│       │       └─ Started but low VRAM → Increase batch size or use smaller GPU
│       │
│       ├─ UNDERUTILIZED (30-59%)
│       │   ├─ NV 미바운드 (networkVolumeId 없음) → gpu_cost_compare 즉시 호출 [필수]
│       │   │     비교 풀: 온디맨드만 (스팟 제외 — NEVER spot 규칙)
│       │   │     결과: 현재 대비 MIGRATION_THRESHOLD(20%)+ 저렴한 대안 → 마이그레이션 강력 권유
│       │   │     (사용자 결정. 데이터 이전 비용·속도 차이 함께 고지)
│       │   ├─ NV 바운드 (networkVolumeId 있음) → gpu_cost_compare 선택적 권고
│       │   │     (GPU 변경 시 NV DC 제약으로 NV 재생성 필요함을 명시)
│       │   └─ 병행: increase batch size (provide perSampleMb for recommendation)
│       │
│       ├─ MODERATE (60-74%)
│       │   └─ Acceptable. Suggest batch size increase if easy.
│       │
│       ├─ OPTIMAL (75-89%)
│       │   └─ Good. No action needed.
│       │
│       └─ NEAR_OOM (>=90%)
│           └─ Risk of crash. Suggest: reduce batch size, enable gradient
│               checkpointing, or use mixed precision (fp16/bf16)
│
├─ Low gpuUtil (<30%) but high VRAM usage (>50%)?
│   └─ Data loader bottleneck. Recommend:
│       num_workers=<cpu_count-1>, pin_memory=True, prefetch_factor=2
│
└─ Re-check every 5-10 min during long runs
    └─ If IDLE for 2+ consecutive checks → warn user about cost waste
```

### Staging Pod Pattern

**Use this when data upload or preprocessing is needed before GPU training.** Avoids paying GPU rates ($0.44+/hr) while uploading or preprocessing data.

1. `create_network_volume` (size = `ceil((dataset_gb + outputs_gb) * 1.3)`, **min 50GB** — see NV Sizing Formula below)
2. `create_pod_auto` with `networkVolumeId` — use a **cheap GPU or smallest available** just for upload
3. `upload_files` to `/workspace` + run any CPU preprocessing
4. `stop_pod` or `delete_pod` — data persists on the network volume
5. `create_pod_auto` with same `networkVolumeId` — now with the real GPU for training
6. Training starts immediately — data is already there, zero idle time

**When to use:** datasets > 500MB, or any preprocessing (tokenization, feature extraction) that doesn't need GPU.

### Background GPU Monitoring

For long training runs, deploy a lightweight background monitor on the pod:

```bash
# Deploy monitor (via execute_ssh_command)
nohup bash -c 'echo $$ > /workspace/gpu_monitor.pid; while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader >> /workspace/gpu_metrics.csv; sleep 60; done' > /dev/null 2>&1 &

# Check metrics (via execute_ssh_command)
tail -5 /workspace/gpu_metrics.csv

# Cleanup when done
kill $(cat /workspace/gpu_monitor.pid) 2>/dev/null; rm -f /workspace/gpu_monitor.pid
```

### Network Volume Workflow

For persistent data that survives pod termination:

1. `create_network_volume` → Create volume in target datacenter (min 10GB)
2. `create_pod_auto` with `networkVolumeId` → Pod auto-placed in volume's datacenter
3. Upload data to `/workspace` (mounted from network volume)
4. Stop/delete pod — data persists on the volume
5. Later: create new pod with same `networkVolumeId` — data is still there

**Key rules:**
- Network volumes are datacenter-bound — pods must be in the same datacenter
- `create_pod_auto` automatically resolves datacenter affinity when `networkVolumeId` is provided
- `delete_network_volume` requires `confirmName` safety check — user must type exact volume name
- Use `list_network_volumes` to see all volumes with their datacenter locations

#### NV Sizing Formula

**Do not default to 20GB** — that's a trap for any non-trivial dataset. The 20GB
quota silently truncates files (rsync/tar will produce 0-byte files when full).

```
size_gb = ceil((dataset_gb + outputs_gb) * 1.3)   # 30% headroom for checkpoints, logs, tmp
minimum recommended = 50GB                         # below this, the cost saving is negligible
```

**Cost reference:** RunPod NV is ~$0.07/GB/month → 50GB = ~$3.50/mo, 100GB = ~$7/mo.
Even if you forget for a month, the cost of a too-small NV (re-upload, debug time,
truncated training data) vastly exceeds the storage cost.

**Sizing examples:**
- 1GB dataset, 2GB checkpoints → `ceil(3 * 1.3) = 4` → use **50GB** (minimum)
- 22GB dataset, 5GB outputs → `ceil(27 * 1.3) = 36` → use **50GB**
- 80GB dataset, 20GB outputs → `ceil(100 * 1.3) = 130` → use **150GB**

When you suggest `create_network_volume` to the user, always state the computed
size and the formula. Never accept "use the default" — there is no good default.

### Pod Metadata Persistence (`save_pod_metadata`)

After a pod is fully provisioned (image installed, packages installed, data
uploaded, training launched), call `save_pod_metadata` to capture the full
provisioning recipe. Without this, debugging a failed run later is impossible —
the pod is gone and so is every detail of how it was set up.

**When to call:**
1. **After post-create setup completes** — image, apt/pip installs, data upload
   done, training command launched in tmux/nohup
2. **After encountering an incident** — append to `incidents[]` and re-save
3. **Before deleting the pod** — set `deleted_at` and final `cost_actual_usd`

**Default save path:** `./.omc/pods/{YYYY-MM-DD}_{podName}.yaml` (relative
to the user's working directory, aligned with the existing `.omc/*` convention).
Output is YAML for human readability of multi-line `post_create_steps` and
`incidents`. Override with `path` argument if the project uses a different
convention.

**Auto-prefill stub:** `create_pod` and `create_pod_auto` now echo a `## Pod
Metadata Stub` block in their response, pre-filled with everything known at
pod-creation time (pod_id, datacenter, gpu, image, cost_per_hr,
container_disk_gb, network_volume). You only need to fill in `purpose` and
later append to `post_create_steps` / `incidents`. **Pass that stub straight
to `save_pod_metadata` once `purpose` is set** — there is no excuse for
forgetting to record a pod.

**Required schema fields:**
```
pod_id, name, purpose, created_at
datacenter, gpu, gpu_count, cost_per_hr, image
```

**Recommended fields (fill what you know):**
```
deleted_at, container_disk_gb, network_volume {id, name, size_gb, datacenter},
ssh {host, port}, post_create_steps [], data {source, dest, transfer_method},
code {source, commit}, execution {script, log, output_dir, expected_*},
monitor {cron_id}, incidents []
```

**Workflow rule:** the metadata file lives in the **user's project repo**, NOT
in runpod-mcp. It should be tracked in git so future debugging has the full
history. After saving, commit with the format `chore(pod): record {pod_name}`.

**Example call sequence:**
```
create_pod_auto                              ← response includes Pod Metadata Stub
→ save_pod_metadata({metadata: <stub + purpose>})  ← record IMMEDIATELY, don't wait
→ git add .omc/pods/<file>.yaml && git commit -m "chore(pod): record <name>"
→ wait_for_pod → upload_files → execute_ssh_command (setup)
→ (after each major setup step) read existing yaml, append to post_create_steps, save_pod_metadata again
→ execute_ssh_command (training launch) → append launch command to post_create_steps
→ (on incident) read yaml, append to incidents[], save_pod_metadata again, commit
→ (before delete_pod) set deleted_at + cost_actual_usd, save_pod_metadata, commit
```

### Utilization Labels
- **IDLE** (<30%): GPU is wasted — check if training actually started
- **UNDERUTILIZED** (30-59%): Increase batch size or use a smaller/cheaper GPU
- **MODERATE** (60-74%): Room for improvement
- **OPTIMAL** (75-89%): Good utilization
- **NEAR_OOM** (>=90%): Risk of crash — reduce batch size or enable gradient checkpointing
