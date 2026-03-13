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

## Tools (14)

| Tool | Description |
|------|-------------|
| `list_pods` | List all pods with status and SSH info |
| `get_pod` | Get detailed pod info |
| `create_pod` | Create a GPU pod (REST or GraphQL spot) |
| `create_pod_auto` | Auto-select GPU based on stock availability |
| `stop_pod` | Stop pod (preserves volume) |
| `start_pod` | Start a stopped pod |
| `restart_pod` | Restart a running pod |
| `delete_pod` | Permanently delete a pod |
| `wait_for_pod` | Poll until SSH-ready (TCP probe) |
| `list_gpu_types` | GPU types with pricing and stock (GraphQL) |
| `get_ssh_command` | Get SSH connection command |
| `execute_ssh_command` | Run command on pod via SSH |
| `upload_files` | Upload files via rsync |
| `download_files` | Download files via rsync |

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

## Compared to @runpod/mcp-server

The official RunPod MCP server covers pod/endpoint creation but lacks:

| Feature | Official | This server |
|---------|----------|-------------|
| Pod stop/start/restart | No | Yes |
| SSH execution | No | Yes |
| File transfer (rsync) | No | Yes |
| Wait for ready (TCP probe) | No | Yes |
| GPU stock checking | No | Yes |
| Auto GPU selection | No | Yes |
| Spot bid pricing | No | Yes |

## License

MIT
