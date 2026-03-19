# runpod-mcp: Claude Code Project Instructions

## GPU Optimization Workflow

When a user is doing ML training on RunPod, follow this optimization pattern:

### Tool Sequencing
1. `create_pod_auto` → `wait_for_pod` → `upload_files` → `execute_ssh_command` (always in this order)
2. After training starts (1-2 min), call `gpu_health_check` to measure utilization
3. If underutilized, call `gpu_health_check` with `perSampleMb` for batch size recommendation
4. Call `gpu_cost_compare` if GPU is underutilized to find cheaper alternatives

### Critical Rules
- **Background long-running commands**: `execute_ssh_command` blocks the MCP server (spawnSync). Use `nohup cmd > /workspace/log 2>&1 &` and poll with `tail`
- **Never auto-adjust batch size or migrate GPUs** — always present recommendations and let the user decide
- **Spot instance warning**: Recommend checkpoints on network volumes (not container disk) since spot pods can be preempted
- **Overprovisioning**: If `create_pod_auto` reports overprovisioning, flag it to the user

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
