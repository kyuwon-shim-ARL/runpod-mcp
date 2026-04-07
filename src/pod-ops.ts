/**
 * Extracted orchestration functions from index.ts for testability.
 * These are pure functions (or client-dependent but isolated operations)
 * that were previously closures inside MCP tool handlers.
 */
import type { Pod, GpuType } from "./types.js";
import { getStockStatus, isInStock, getSpotPrice, getOnDemandPrice, isOverprovisioned } from "./gpu-utils.js";
import type { RunPodClient } from "./api.js";

// ── cleanup_stale_pods filtering ──

export interface StaleResult {
  pod: Pod;
  idleHours: number;
}

export interface SkippedResult {
  pod: Pod;
  reason: string;
}

export function filterStalePods(
  pods: Pod[],
  graceHours: number,
  now: number = Date.now()
): { stale: StaleResult[]; skipped: SkippedResult[] } {
  const graceMs = graceHours * 60 * 60 * 1000;
  const skipPattern = /keep|persist/i;
  const stale: StaleResult[] = [];
  const skipped: SkippedResult[] = [];

  for (const pod of pods) {
    if (pod.desiredStatus !== "EXITED") {
      skipped.push({ pod, reason: `status=${pod.desiredStatus}` });
      continue;
    }
    if (skipPattern.test(pod.name)) {
      skipped.push({ pod, reason: "name contains keep/persist" });
      continue;
    }
    if (!pod.lastStatusChange) {
      skipped.push({ pod, reason: "no lastStatusChange timestamp" });
      continue;
    }
    const idleMs = now - new Date(pod.lastStatusChange).getTime();
    if (idleMs < graceMs) {
      skipped.push({ pod, reason: `idle ${(idleMs / 3600000).toFixed(1)}h < grace ${graceHours}h` });
      continue;
    }
    stale.push({ pod, idleHours: Math.round(idleMs / 3600000) });
  }

  return { stale, skipped };
}

// ── create_pod_auto DC priority ──

/**
 * Default datacenter priority for create_pod_auto fallback.
 *
 * Order is based on observed RunPod stock pool sizes (largest first):
 * - US-GA-1, US-CA-2: largest US capacity pools, most consistent stock
 * - EU-SE-1, EU-CZ-1: largest EU pools
 * - AP-JP-1: medium pool, variable
 * - US-TX-3, EU-RO-1: smaller pools, easily exhausted but kept as last resort
 *
 * Used only when networkVolumeId is NOT provided (NV constrains the DC).
 */
export const DEFAULT_DC_PRIORITY: string[] = [
  "US-GA-1",
  "US-CA-2",
  "EU-SE-1",
  "EU-CZ-1",
  "AP-JP-1",
  "US-TX-3",
  "EU-RO-1",
];

/**
 * Format a (dc × gpu) failure matrix into a human-readable diagnostic block.
 * Used when create_pod_auto exhausts all DC × GPU combinations.
 */
export function formatDcGpuFailureMatrix(
  attempts: Array<{ dc: string; gpu: string; error: string }>
): string {
  if (attempts.length === 0) return "";
  const byDc = new Map<string, Array<{ gpu: string; error: string }>>();
  for (const a of attempts) {
    if (!byDc.has(a.dc)) byDc.set(a.dc, []);
    byDc.get(a.dc)!.push({ gpu: a.gpu, error: a.error });
  }
  const lines: string[] = [];
  for (const [dc, rows] of byDc) {
    lines.push(`  [${dc}]`);
    for (const r of rows) {
      const truncated = r.error.length > 120 ? r.error.slice(0, 117) + "..." : r.error;
      lines.push(`    - ${r.gpu}: ${truncated}`);
    }
  }
  return lines.join("\n");
}

// ── create_pod_auto GPU selection ──

export interface GpuSelectionOptions {
  gpuPreference: string[];
  minVram: number;
  gpuCount: number;
  spot: boolean;
  maxBidPerGpu: number;
}

export interface GpuCandidate {
  gpu: GpuType;
  gpuId: string;
  stock: string | null;
  ondemandPrice: number;
  bidPrice: number | undefined;
  minBid: number;
  overprovisionWarning: string;
}

export interface GpuSelectionResult {
  candidates: GpuCandidate[];
  errors: string[];
}

export function selectGpuCandidates(
  gpuTypes: GpuType[],
  options: GpuSelectionOptions
): GpuSelectionResult {
  const gpuMap = new Map(gpuTypes.map((g) => [g.id, g]));
  const candidates: GpuCandidate[] = [];
  const errors: string[] = [];

  for (const gpuId of options.gpuPreference) {
    const gpu = gpuMap.get(gpuId);
    if (!gpu) continue;
    if (gpu.memoryInGb < options.minVram) continue;
    const stock = getStockStatus(gpu);
    if (!isInStock(gpu)) continue;

    const ondemandPrice = getOnDemandPrice(gpu) ?? 1.0;
    const minBid = getSpotPrice(gpu) ?? 0;
    const bidPrice = options.spot
      ? Math.min(options.maxBidPerGpu, ondemandPrice * 0.8)
      : undefined;

    if (options.spot && bidPrice != null && minBid > 0 && bidPrice < minBid) {
      errors.push(`${gpu.displayName}: Bid $${bidPrice.toFixed(3)}/hr below minimum $${minBid}/hr, skipped`);
      continue;
    }

    const overprovisionWarning = isOverprovisioned(gpu.memoryInGb, options.minVram)
      ? `\nOverprovisioned: ${gpu.displayName} has ${gpu.memoryInGb}GB VRAM but you requested ${options.minVram}GB minimum.\n` +
        `  Consider a smaller GPU to save cost, or increase your workload to utilize the extra VRAM.\n`
      : "";

    candidates.push({ gpu, gpuId, stock, ondemandPrice, bidPrice, minBid, overprovisionWarning });
  }

  return { candidates, errors };
}

// ── save_pod_metadata helpers ──

/**
 * Sanitize a pod name for use as a filename component.
 * Removes/replaces characters that are unsafe on common filesystems.
 */
export function sanitizePodName(name: string): string {
  return name
    .trim()
    .replace(/[/\\:*?"<>|]/g, "-")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "unnamed-pod";
}

/**
 * Extract YYYY-MM-DD from an ISO timestamp, falling back to today's date in UTC.
 */
export function isoToDateStamp(iso: string | undefined, now: Date = new Date()): string {
  if (iso) {
    const d = new Date(iso);
    if (!isNaN(d.getTime())) return d.toISOString().slice(0, 10);
  }
  return now.toISOString().slice(0, 10);
}

/**
 * Build the canonical save path for a pod metadata file.
 *
 * Pattern: {basePath}/{YYYY-MM-DD}_{sanitized-name}.yaml
 *   - basePath defaults to ".omc/pods" (aligned with existing .omc/* convention)
 *   - date stamp is taken from metadata.created_at, or today if absent
 *   - name is sanitized for cross-platform filesystem safety
 */
export function buildPodMetadataPath(
  metadata: { name?: string; created_at?: string },
  basePath: string = ".omc/pods"
): string {
  const dateStamp = isoToDateStamp(metadata.created_at);
  const safeName = sanitizePodName(metadata.name ?? "unnamed-pod");
  // Use forward slash; Node fs accepts it on all platforms.
  return `${basePath}/${dateStamp}_${safeName}.yaml`;
}

// ── Minimal YAML serializer (zero-dep) ──
//
// Hand-written for the pod metadata schema. NOT a full YAML 1.2 implementation —
// only handles: null, boolean, finite number, string (bare/quoted/block scalar),
// homogeneous arrays of scalars or objects, and nested plain objects.
//
// Why not a real YAML lib: keeps runpod-mcp dependency-free (current deps:
// @modelcontextprotocol/sdk + zod). The schema is small and stable, and the
// output is round-trippable through any standard YAML parser.

function isYamlPlainScalar(s: string): boolean {
  if (s === "") return false;
  if (/^\s|\s$/.test(s)) return false;
  // Special chars that require quoting in YAML 1.2 plain scalars
  if (/[:#&*!|>'"%@`,?\[\]{}]/.test(s)) return false;
  if (s.includes("\n")) return false;
  // Strings that look like YAML keywords or numbers must be quoted to round-trip
  if (/^(null|true|false|yes|no|on|off|~)$/i.test(s)) return false;
  if (/^-?\d+(\.\d+)?(e[+-]?\d+)?$/i.test(s)) return false;
  return true;
}

function emitYamlScalar(value: unknown, indent: number): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "null";
  if (typeof value === "string") {
    if (value.includes("\n")) {
      // Block scalar (literal style preserves newlines)
      const blockIndent = " ".repeat(indent + 2);
      const lines = value.split("\n").map((l) => blockIndent + l).join("\n");
      return "|\n" + lines;
    }
    return isYamlPlainScalar(value) ? value : JSON.stringify(value);
  }
  // Fallback for other types — JSON-encode (rare in our schema)
  return JSON.stringify(value);
}

/**
 * Serialize a plain JS value to YAML. Limited to the pod metadata schema's needs:
 * objects, arrays, strings, numbers, booleans, null. Output is human-readable
 * and parseable by any standard YAML 1.2 library.
 */
export function toYaml(value: unknown, indent: number = 0): string {
  const pad = " ".repeat(indent);

  // Top-level scalar
  if (value === null || value === undefined || typeof value !== "object") {
    return emitYamlScalar(value, indent) + "\n";
  }

  if (Array.isArray(value)) {
    if (value.length === 0) return pad + "[]\n";
    const lines: string[] = [];
    for (const item of value) {
      if (item !== null && typeof item === "object" && !Array.isArray(item)) {
        // Object item: first key inline with "- ", rest indented
        const inner = toYaml(item, indent + 2).trimEnd();
        const innerLines = inner.split("\n");
        // First line: replace leading "  " (the inner pad) with "- "
        lines.push(pad + "- " + innerLines[0].trimStart());
        for (const l of innerLines.slice(1)) lines.push(l);
      } else {
        lines.push(pad + "- " + emitYamlScalar(item, indent));
      }
    }
    return lines.join("\n") + "\n";
  }

  // Plain object
  const entries = Object.entries(value as Record<string, unknown>);
  if (entries.length === 0) return pad + "{}\n";
  const lines: string[] = [];
  for (const [k, v] of entries) {
    if (v === null || v === undefined) {
      lines.push(pad + k + ": null");
    } else if (Array.isArray(v)) {
      if (v.length === 0) {
        lines.push(pad + k + ": []");
      } else {
        lines.push(pad + k + ":");
        lines.push(toYaml(v, indent + 2).trimEnd());
      }
    } else if (typeof v === "object") {
      const subEntries = Object.entries(v as Record<string, unknown>);
      if (subEntries.length === 0) {
        lines.push(pad + k + ": {}");
      } else {
        lines.push(pad + k + ":");
        lines.push(toYaml(v, indent + 2).trimEnd());
      }
    } else {
      lines.push(pad + k + ": " + emitYamlScalar(v, indent));
    }
  }
  return lines.join("\n") + "\n";
}

// ── upload_files: integrity check helpers ──

export interface UploadIntegrityCheck {
  status: "OK" | "FREE_SPACE_LOW" | "SIZE_MISMATCH" | "SKIPPED";
  message: string;
  localBytes?: number;
  remoteBytes?: number;
  availBytes?: number;
  ratio?: number;
}

/**
 * Parse `du -sb` output ("123456\t/path") into a byte count.
 * Returns null on parse failure.
 */
export function parseDuBytes(output: string): number | null {
  const m = output.trim().match(/^(\d+)/);
  if (!m) return null;
  const n = Number(m[1]);
  return Number.isFinite(n) ? n : null;
}

/**
 * Parse `df -B1 --output=avail {path}` output. The first line is the header
 * "Avail" or "Available", second line is the byte count.
 * Returns null on parse failure.
 */
export function parseDfAvailBytes(output: string): number | null {
  const lines = output.trim().split("\n");
  if (lines.length < 2) return null;
  const m = lines[1].trim().match(/^(\d+)/);
  if (!m) return null;
  const n = Number(m[1]);
  return Number.isFinite(n) ? n : null;
}

/**
 * Decide whether a free-space precheck passes.
 * Requires destination to have at least localBytes * safetyMargin available.
 */
export function checkFreeSpace(
  localBytes: number,
  availBytes: number,
  safetyMargin: number = 1.1
): UploadIntegrityCheck {
  const required = Math.ceil(localBytes * safetyMargin);
  if (availBytes >= required) {
    return {
      status: "OK",
      message: `Free space OK: ${formatBytes(availBytes)} available, ${formatBytes(required)} required (with ${Math.round((safetyMargin - 1) * 100)}% margin)`,
      localBytes,
      availBytes,
    };
  }
  const shortBy = required - availBytes;
  return {
    status: "FREE_SPACE_LOW",
    message:
      `Destination has only ${formatBytes(availBytes)} free, but upload needs ${formatBytes(required)} ` +
      `(local data: ${formatBytes(localBytes)} + 10% margin). Short by ${formatBytes(shortBy)}. ` +
      `Increase the network volume size or pick a different destination — uploading anyway WILL silently truncate files (rsync/tar produce 0-byte files when quota fills). ` +
      `If you really know what you're doing, pass verifySize=false to skip this check.`,
    localBytes,
    availBytes,
  };
}

/**
 * Decide whether a post-upload size match passes.
 * Allows up to (1 - tolerance) shortfall before flagging (filesystem block
 * padding can make remote size differ slightly from local).
 */
export function checkSizeMatch(
  localBytes: number,
  remoteBytes: number,
  tolerance: number = 0.95
): UploadIntegrityCheck {
  if (localBytes === 0) {
    return {
      status: "OK",
      message: "Source is empty; nothing to verify",
      localBytes,
      remoteBytes,
      ratio: 1,
    };
  }
  const ratio = remoteBytes / localBytes;
  if (ratio >= tolerance) {
    return {
      status: "OK",
      message: `Size match OK: local ${formatBytes(localBytes)} → remote ${formatBytes(remoteBytes)} (${(ratio * 100).toFixed(1)}%)`,
      localBytes,
      remoteBytes,
      ratio,
    };
  }
  return {
    status: "SIZE_MISMATCH",
    message:
      `Upload looks INCOMPLETE: local ${formatBytes(localBytes)} but remote only ${formatBytes(remoteBytes)} (${(ratio * 100).toFixed(1)}%). ` +
      `This is the silent-truncation pattern (NV quota exceeded → 0-byte files). ` +
      `Inspect the destination, free space, and consider re-uploading after enlarging the volume.`,
    localBytes,
    remoteBytes,
    ratio,
  };
}

function formatBytes(n: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 ? 0 : 1)}${units[i]}`;
}

// ── execute_ssh_command: setup step heuristic ──

const SETUP_COMMAND_PATTERNS = [
  /\bapt(-get)?\s+(install|update|upgrade)\b/,
  /\b(pip|pip3)\s+install\b/,
  /\bconda\s+(install|create)\b/,
  /\bgit\s+clone\b/,
  /\brsync\b/,
  /\btar\s+-?[xc]/,
  /\bunzip\b/,
  /\bwget\b/,
  /\bcurl\s+(-O|.*-o\s)/,
  /\bdocker\s+(pull|run)\b/,
  /\bmake\s+(install|build)\b/,
];

/**
 * Heuristic: does this command look like a setup/install step that should
 * be recorded in post_create_steps? Used to nudge Claude to update pod metadata.
 */
export function looksLikeSetupCommand(command: string): boolean {
  return SETUP_COMMAND_PATTERNS.some((p) => p.test(command));
}

// ── delete_pod: cost estimation ──

/**
 * Estimate cost incurred by a pod given its hourly rate and the time it has
 * been running. Used by delete_pod to echo a final cost so the metadata
 * record can be closed with cost_actual_usd.
 *
 * Returns null if the inputs are insufficient (no start time or no rate).
 */
export function estimatePodCost(
  costPerHr: number | undefined,
  startedAt: string | undefined,
  now: Date = new Date()
): { hours: number; cost: number } | null {
  if (!costPerHr || !startedAt) return null;
  const start = new Date(startedAt).getTime();
  if (!Number.isFinite(start)) return null;
  const hours = (now.getTime() - start) / 3_600_000;
  if (hours < 0) return null;
  return { hours, cost: hours * costPerHr };
}

// ── create_pod_auto: prefill metadata stub ──

export interface PodMetadataStubInput {
  pod_id: string;
  name: string;
  created_at: string;
  datacenter?: string;
  gpu?: string;
  gpu_count?: number;
  cost_per_hr?: number;
  image?: string;
  container_disk_gb?: number;
  network_volume?: { id: string; name: string; size_gb: number; datacenter?: string } | null;
}

/**
 * Build a JSON-formatted pod metadata stub from facts known at pod-creation time.
 * Echoed in create_pod / create_pod_auto responses so Claude can pass it directly
 * to save_pod_metadata after enriching with purpose / post_create_steps / etc.
 */
export function buildPodMetadataStub(input: PodMetadataStubInput): string {
  const stub = {
    pod_id: input.pod_id,
    name: input.name,
    purpose: "<fill in: what this pod is for>",
    created_at: input.created_at,
    deleted_at: null,
    datacenter: input.datacenter ?? null,
    gpu: input.gpu ?? null,
    gpu_count: input.gpu_count ?? 1,
    cost_per_hr: input.cost_per_hr ?? null,
    container_disk_gb: input.container_disk_gb ?? null,
    image: input.image ?? null,
    network_volume: input.network_volume ?? null,
    ssh: null,
    post_create_steps: [],
    incidents: [],
  };
  return JSON.stringify(stub, null, 2);
}

// ── delete_pod with auto-stop ──

export async function deletePodWithStop(
  client: RunPodClient,
  podId: string,
  timeoutMs: number = 60_000
): Promise<{ wasRunning: boolean }> {
  const pod = await client.getPod(podId);
  let wasRunning = false;

  if (pod.desiredStatus === "RUNNING") {
    wasRunning = true;
    await client.stopPod(podId);
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      await new Promise((r) => setTimeout(r, 3_000));
      const current = await client.getPod(podId);
      if (current.desiredStatus !== "RUNNING") break;
    }
  }

  await client.deletePod(podId);
  return { wasRunning };
}
