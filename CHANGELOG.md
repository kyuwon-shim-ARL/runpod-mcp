# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-04-16

### Added
- `verify_data_on_nv` — SSH-verifies dataset presence on a Network Volume staging pod; issues a 72h readiness token saved to `.omc/gpu-exec/nv_ready_{nvId}.json`. Catches file-missing and data-truncation (minTotalGb check) before the staging pod is deleted.
- `run_preflight` — pre-flight SSH checks on a running pod: disk free space, requirements.txt ML-package pinning (unbounded `>=` detection for peft/transformers/torch etc.), required file existence, system tool availability, and Python import smoke tests. Based on postmortem: 6/8 incident bugs catchable in <5 min.
- `watch_running_pods` — spawns `scripts/pod_watcher.sh` as a background process; polls GPU utilization via SSH + nvidia-smi every N minutes; auto-stops idle pods (gpuUtil < threshold for consecutive checks); supports `error-only` mode for long runs; `expectedCompletionAt` switches to 1-min interval in the final 30 minutes.
- `stop_watching_pods` — kills the background watcher (SIGTERM + 5s waitpid + SIGKILL fallback).
- `get_pipeline_events` — reads `.omc/gpu-exec/events.jsonl`; sticky `WATCHER_EXITED` warning (repeats on every call until watcher is restarted) with manual pod-stop instructions.
- `create_pod_auto`: `nvReadinessToken` parameter — when `gpuCount >= 2` + `networkVolumeId` is set, requires a valid token from `verify_data_on_nv` (TTL 72h). Blocks with `FILE_NOT_FOUND`, `TOKEN_MISMATCH`, or `EXPIRED` to prevent costly multi-GPU pods starting with missing/stale data.
- `plan_gpu_job`: container disk warning — estimates output as `(modelSizeGb + datasetGb×0.1 + 2) × gpuCount` and warns when >70% of RunPod's default 30GB container disk.
- `scripts/pod_watcher.sh` — 224-line bash watcher; PID guard; orphan-PID auto-cleanup; GraphQL stop with 1 retry + `WATCHER_EXITED` on failure; RUNPOD_API_KEY inherited via nohup.
- 43 new test cases (251 → 294 total): token TTL/truncation, nvReadinessToken gate, requirements pinning, disk-free, WATCHER_EXITED, disk-warning boundaries.

### Driven by
- T14/T15 incidents: 4-GPU pods ($1.36/hr) used for data transfer; GPU idle after training undetected for hours.

## [0.4.0] - 2026-04-08

### Added
- `dcPriority` argument on `create_pod_auto` — DC × GPU fallback loop. Outer loop iterates datacenters in priority order, inner loop iterates GPU candidates. Defaults to a built-in list tuned by observed RunPod stock pool sizes (US-GA-1 > US-CA-2 > EU-SE-1 > EU-CZ-1 > AP-JP-1 > US-TX-3 > EU-RO-1). When `networkVolumeId` is set, the NV's DC is used and `dcPriority` is ignored.
- `save_pod_metadata` tool — persists a pod's full provisioning recipe (DC, GPU, image, post-create steps, data layout, incidents) to a YAML file in the user's project repo (default `.omc/pods/{date}_{name}.yaml`). Designed to be git-committed so debugging is possible after the pod is deleted.
- `create_pod` and `create_pod_auto` now echo a `## Pod Metadata Stub` block in their success response — a JSON object pre-filled with everything known at pod-creation time. Pass straight to `save_pod_metadata` after setting `purpose`.
- `upload_files` integrity defense (Patch D) — pre-upload free-space precheck (`df -B1 --output=avail`) and post-upload size match (`du -sb` on local + remote, 95% tolerance). Catches the silent-truncation pattern where rsync produces 0-byte files when the destination quota is exhausted. Opt-out via `verifySize=false`.
- `execute_ssh_command` setup-step nudge — heuristic detects apt/pip/conda/git clone/rsync/tar/wget/etc. and appends a one-line hint to record the command in pod metadata's `post_create_steps`.
- `delete_pod` cost estimate — captures `costPerHr × uptime` before deletion and echoes it in the response, with a reminder to update the pod metadata's `deleted_at` and `cost_actual_usd`.
- Hand-written zero-dep YAML serializer (`toYaml` in `pod-ops.ts`) — handles scalars, nested objects, arrays of strings, multi-line block scalars, and YAML-keyword-aware quoting. Pairs with the metadata schema; no external YAML dep added.
- 46 new test cases (169 → 215 total): DC priority defaults, failure matrix formatter, pod metadata helpers (sanitizePodName, isoToDateStamp, buildPodMetadataPath, buildPodMetadataStub), YAML round-trip, upload integrity helpers (parseDuBytes, parseDfAvailBytes, checkFreeSpace, checkSizeMatch including the exact 22GB→20GB scenario), setup command heuristic, cost estimation edge cases.

### Changed
- `create_network_volume` description and tool-level warning now mandate the sizing formula `ceil((dataset_gb + outputs_gb) * 1.3)` with a practical 50GB minimum. The tool emits a warning when `size < 50` because RunPod's silent quota-truncation makes undersized volumes a real data-loss hazard. Cost reference ($0.07/GB/month) included.
- `create_pod_auto` failure output now shows a `(DC, GPU) → error` matrix grouped by datacenter, plus three escape options when a network volume constrains the DC.
- CLAUDE.md gains "NV Sizing Formula" and "Pod Metadata Persistence" sections documenting the new workflow end-to-end.

### Driven by
- piu-v2 60h training incident (2026-04-07): US-TX-3 stock exhaustion forced manual fallback through AP-JP-1 to EU-RO-1; a 22GB dataset uploaded into a 20GB network volume produced 16996 zero-byte `.npy` files; no record of the pod's setup survived for post-mortem.

## [0.2.0] - 2026-03-30

### Added
- `cleanup_stale_pods` tool — detect and delete idle EXITED pods with grace period and keep/persist skip pattern
- `gpu_sample_burst` tool — multi-sample GPU monitoring with trend verdict (STABLE_OPTIMAL, DEGRADING, etc.)
- `cloudType` parameter for `create_pod` and `create_pod_auto` — select ALL, SECURE, or COMMUNITY cloud
- Standalone watchdog daemon (`runpod-watchdog` bin) for 24/7 GPU idle detection via cron
  - SSH nvidia-smi monitoring with consecutive idle tracking
  - `--auto-stop` mode: stop pods after N consecutive idle checks
  - `--skip-pattern` regex for excluding pods by name
  - Alerter interface for future webhook extension
- Extracted orchestration functions to `pod-ops.ts` for testability:
  `filterStalePods`, `selectGpuCandidates`, `deletePodWithStop`
- Orchestration integration tests and watchdog unit tests (34 new test cases)

### Fixed
- `cleanup_stale_pods`: failed deletions now correctly push to `failed` array (was incorrectly pushing to `deleted`)
- `delete_pod`: auto-stops running pods before deletion (prevents API errors)
- `delete_pod`: handle empty response body from RunPod API

### Changed
- `RunPodClient.config` changed from `private` to `readonly` (enables watchdog access)
- `createSpotPod` default `cloudType` changed from `"SECURE"` to `"ALL"` (breaking: may affect existing spot pod creation)
- README updated to reflect 18 tools (was incorrectly showing 16)

## [0.1.0] - 2026-03-15

### Added
- Initial release with 16 MCP tools
- Pod lifecycle management (list, get, create, start, stop, restart, delete)
- SSH command execution and file transfer (rsync upload/download)
- GPU type listing with pricing and stock status
- GPU health check with nvidia-smi parsing and batch size recommendations
- GPU cost comparison across available GPU types
- Network volume management (list, get, create, delete)
- Proactive GPU Management Protocol in CLAUDE.md
