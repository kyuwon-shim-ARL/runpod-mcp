---
name: gpu-optimize
description: GPU memory optimization workflow - provision, measure, and optimize GPU utilization in 3 semi-automatic phases
---

# GPU Optimization Workflow

Guide the user through a 3-phase GPU optimization cycle: provision a pod, measure utilization, and recommend cost savings.

## Important Constraints

- **Never auto-migrate GPUs** — present recommendations, let the user decide
- **Never auto-adjust batch size** — recommend values, user confirms before changing
- **Long-running training must be backgrounded** — `execute_ssh_command` uses spawnSync (blocks MCP server). Always use `nohup ... &` and poll logs separately
- **Spot instances can be preempted** — always recommend saving checkpoints to network volumes, not container disk

## Phase 1: Provision (Automatic)

Run these tools in sequence without user intervention:

1. **create_pod_auto** with user's requirements:
   - Set `optimizePytorch: true` for PyTorch workloads
   - Set `minVram` to actual model requirement (avoid overprovisioning)
   - Note any overprovisioning warnings in the response

2. **wait_for_pod** — wait until SSH-ready

3. **upload_files** — upload training code/data to `/workspace`

Report to user: Pod ready, GPU selected, any overprovisioning warnings.

## Phase 2: Measure & Recommend (Semi-automatic)

**Requires user action**: User must start training first.

1. Ask the user to provide their training command. Run it backgrounded:
   ```
   execute_ssh_command: nohup python train.py > /workspace/train.log 2>&1 &
   ```

2. Wait 1-2 minutes, then run **gpu_health_check** (without `perSampleMb` first):
   - If **IDLE**: Training may not have started. Check logs with `execute_ssh_command: tail -20 /workspace/train.log`
   - If **UNDERUTILIZED**: Recommend increasing batch size
   - If **OPTIMAL**: No action needed
   - If **NEAR_OOM**: Recommend reducing batch size or enabling mixed precision

3. If underutilized, help user calculate `perSampleMb`:
   ```
   perSampleMb = current_vram_used_mb / current_batch_size
   ```
   Then run **gpu_health_check** again with `perSampleMb` to get batch size recommendation.

4. **Present recommendation to user** — do NOT auto-apply:
   - Show current vs recommended batch size
   - Warn: "Increase batch size gradually (max 2x per step) and monitor for OOM"
   - For transformer models: "Memory scales O(n^2) with sequence length"

5. If user approves, help them modify training config and restart.

## Phase 3: Cost Report (Report Only)

After training is stable with optimized batch size:

1. Run **gpu_cost_compare** to find cheaper alternatives

2. Present findings as a report:
   - Current GPU cost (hourly + monthly estimate)
   - Cheaper alternatives with savings
   - Stock availability

3. **Do NOT auto-migrate** — if user wants to switch:
   - Warn about spot instance re-acquisition risk
   - Recommend saving checkpoint first
   - Help create new pod and transfer data

## Error Handling

| Error | Action |
|-------|--------|
| No GPU stock | Show alternatives from `list_gpu_types`, let user choose |
| SSH not ready | Re-run `wait_for_pod` |
| Training OOM after batch increase | Reduce batch size by 50%, re-run health check |
| Spot preemption | Check if pod still exists with `get_pod`, help user re-create |
| nvidia-smi not found | Pod image may lack GPU drivers, suggest a different image |

## Quick Reference

```
User: "GPU 최적화 해줘" or "optimize my GPU"
→ Start Phase 1

User: "학습 시작했어" or "training is running"
→ Start Phase 2

User: "비용 비교해줘" or "compare costs"
→ Start Phase 3
```
