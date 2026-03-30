# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
