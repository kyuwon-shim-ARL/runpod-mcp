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
- **Background long-running commands**: `execute_ssh_command` blocks the MCP server (spawnSync). Use `nohup cmd > /workspace/log 2>&1 &` and poll with `tail`
- **Never auto-adjust batch size or migrate GPUs** — always present recommendations and let the user decide
- **Spot instance warning**: Recommend checkpoints on network volumes (not container disk) since spot pods can be preempted
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

1. `create_network_volume` (20GB+, target datacenter)
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

### Utilization Labels
- **IDLE** (<30%): GPU is wasted — check if training actually started
- **UNDERUTILIZED** (30-59%): Increase batch size or use a smaller/cheaper GPU
- **MODERATE** (60-74%): Room for improvement
- **OPTIMAL** (75-89%): Good utilization
- **NEAR_OOM** (>=90%): Risk of crash — reduce batch size or enable gradient checkpointing
