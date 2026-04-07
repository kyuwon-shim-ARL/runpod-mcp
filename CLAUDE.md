# runpod-mcp: Claude Code Project Instructions

## GPU Optimization Workflow

When a user is doing ML training on RunPod, follow this optimization pattern:

### Tool Sequencing
1. `create_pod_auto` → `wait_for_pod` → `upload_files` → `execute_ssh_command` (always in this order)
2. After training starts (1-2 min), call `gpu_health_check` to measure utilization
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
│       │   └─ Suggest: increase batch size (provide perSampleMb for recommendation)
│       │       Then: call gpu_cost_compare for cheaper alternatives
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

**Default save path:** `./.runpod/pods/{YYYY-MM-DD}_{podName}.json` (relative
to the user's working directory). Override with `path` argument if the project
uses a different convention.

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
history. After saving, suggest `git add .runpod/pods/<file>.json && git commit`.

**Example call sequence:**
```
create_pod_auto → wait_for_pod → upload_files → execute_ssh_command (setup)
→ execute_ssh_command (training launch)
→ save_pod_metadata({...full provisioning recipe...})
→ git commit .runpod/pods/...
→ (later, on incident) read existing metadata, append to incidents[], save again
```

### Utilization Labels
- **IDLE** (<30%): GPU is wasted — check if training actually started
- **UNDERUTILIZED** (30-59%): Increase batch size or use a smaller/cheaper GPU
- **MODERATE** (60-74%): Room for improvement
- **OPTIMAL** (75-89%): Good utilization
- **NEAR_OOM** (>=90%): Risk of crash — reduce batch size or enable gradient checkpointing
