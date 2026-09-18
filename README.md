# runpod-tools-mcp

MCP server for RunPod GPU pod management. Full lifecycle control, SSH execution, file transfer, and intelligent GPU selection — all from Claude Code or any MCP-compatible AI tool.

## Features

- **Pod Management**: Create, start, stop, restart, delete pods
- **Smart GPU Selection**: Auto-picks available GPUs based on stock status (GraphQL)
- **Spot Instances**: Create spot pods with bid pricing via GraphQL
- **SSH Execution**: Run commands on pods directly
- **File Transfer**: Upload/download via rsync
- **Wait for Ready**: Polls until pod is SSH-ready (includes TCP probe)
- **GPU Browser**: List all GPU types with pricing and stock status

## Quick Start

### Add to Claude Code

```bash
claude mcp add runpod --scope user \
  -e RUNPOD_API_KEY=rp_xxxxxx \
  -- npx -y runpod-tools-mcp
```

With SSH key:

```bash
claude mcp add runpod --scope user \
  -e RUNPOD_API_KEY=rp_xxxxxx \
  -e SSH_KEY_PATH=~/.ssh/id_ed25519 \
  -- npx -y runpod-tools-mcp
```

### From source

```bash
git clone https://github.com/kyuwon-shim-ARL/runpod-mcp.git
cd runpod-mcp
npm install && npm run build
claude mcp add runpod --scope user \
  -e RUNPOD_API_KEY=rp_xxxxxx \
  -- node /path/to/runpod-mcp/dist/index.js
```

## Tools (19)

| Tool | Description |
|------|-------------|
| `list_pods` | List all pods with status and SSH info |
| `get_pod` | Get detailed pod info |
| `create_pod` | Create a GPU or CPU pod (REST). For CPU, pass `computeType: "CPU"` + `cpuFlavorIds`. |
| `create_pod_auto` | Auto-select GPU based on stock availability. Pass `cpuOnly: true` for CPU pods. |
| `stop_pod` | Stop pod (preserves volume) |
| `start_pod` | Start a stopped pod |
| `restart_pod` | Restart a running pod |
| `delete_pod` | Permanently delete a pod (auto-stops running pods) |
| `wait_for_pod` | Poll until SSH-ready (TCP probe) |
| `cleanup_stale_pods` | Find and delete EXITED pods idle beyond grace period |
| `list_gpu_types` | GPU types with pricing and stock (GraphQL) |
| `list_cpu_types` | CPU pod flavors (cpu3/cpu5 × compute/general/highmem) from static catalog |
| `get_ssh_command` | Get SSH connection command |
| `execute_ssh_command` | Run command on pod via SSH |
| `launch_supervised_training` | Launch training under a watchdog; one STATUS file reports RUNNING/ALERT/DONE/FAILED |
| `upload_files` | Upload files via rsync |
| `download_files` | Download files via rsync |
| `gpu_health_check` | Check GPU utilization with batch size advisor |
| `gpu_sample_burst` | Multi-sample GPU metrics with trend analysis |
| `gpu_cost_compare` | Compare GPU costs against cheaper alternatives |

### CPU pods

For CPU-only workloads (data prep, smina/Vina docking, anything that doesn't touch the GPU),
use `cpuOnly: true` to bypass GPU stock probing and cost safety gates. CPU pods run roughly
$0.10–0.30/hr vs $0.40+/hr for the cheapest GPU pod, so they're cheaper for pure-CPU jobs.

```
list_cpu_types                                    # see flavor options
create_pod_auto({ cpuOnly: true,
                  cpuFamily: "compute",           # or "general" / "highmem"
                  vcpuCount: 16,
                  name: "data-prep" })
```

**Pricing caveat**: RunPod does not expose CPU pod pricing via API. `list_cpu_types` shows
flavor metadata only — verify hourly pricing on the [RunPod Console](https://console.runpod.io/pods)
(CPU tab) before cost-sensitive decisions. The pod metadata stub emitted by `create_pod_auto`
sets `cost_per_hr: 0` for CPU pods; fill it in by hand from the Console.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RUNPOD_API_KEY` | Yes | RunPod API key (starts with `rp_`) |
| `SSH_KEY_PATH` | No | Path to SSH private key for pod access |

## Example Workflow

```
1. "list_gpu_types with 24GB+ VRAM, in stock only"
2. "create_pod_auto named my-training-pod"
3. "wait_for_pod until ready"
4. "upload_files ./data to /workspace/data"
5. "execute_ssh_command: python train.py"
6. "download_files /workspace/results to ./results"
7. "delete_pod when done"
```

## Security

- SSH/rsync commands use `spawnSync` with argument arrays (no shell injection)
- API key sent via `Authorization: Bearer` header only (never in URL)
- GraphQL mutations use parameterized variables (no string interpolation)
- Port mappings normalized on read (`"22/tcp"` → `"22"`)

## Why runpod-tools-mcp?

This server is built for **GPU cost optimization and operational control** — not just pod CRUD.

### vs @runpod/mcp-server (official)

| Dimension | Official | runpod-tools-mcp |
|-----------|----------|-----------------|
| **Tools** | 6 (create, list, get, terminate, endpoints) | 18 (full lifecycle + monitoring) |
| **GPU cost optimization** | None | `gpu_health_check`, `gpu_cost_compare`, `gpu_sample_burst` with batch size advisor and trend analysis |
| **Idle detection** | None | `cleanup_stale_pods` + standalone `runpod-watchdog` daemon for 24/7 cron monitoring |
| **SSH/rsync** | None | Async spawn with args arrays (no shell injection), rsync with skip-compress for ML files |
| **Spot instances** | None | GraphQL spot bidding with `cloudType` (ALL/SECURE/COMMUNITY) selection |
| **Pod lifecycle** | Create + terminate | Create, start, stop, restart, delete (auto-stop before delete), wait-for-ready with TCP probe |
| **Network volumes** | None | List, get, create, delete with datacenter affinity and safety checks |

### Key differentiator: GPU waste prevention

Claude Code reads the [GPU Management Protocol](CLAUDE.md) and proactively:
1. Checks GPU utilization after training starts (`gpu_health_check`)
2. Recommends batch size increases for underutilized GPUs
3. Suggests cheaper GPU alternatives via `gpu_cost_compare`
4. Detects idle pods with `cleanup_stale_pods` and the `runpod-watchdog` daemon

## License

MIT
