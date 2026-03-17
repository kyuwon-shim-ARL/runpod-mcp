/** Pure utility functions for GPU monitoring — extracted for testability. */

export interface GpuMetrics {
  index: number;
  name: string;
  totalMb: number;
  usedMb: number;
  freeMb: number;
  usedPct: number;
  gpuUtil: number;
  memUtil: number;
  temp: number;
  label: VramLabel;
}

export type VramLabel = "NEAR_OOM" | "OPTIMAL" | "MODERATE" | "UNDERUTILIZED" | "IDLE" | "UNKNOWN";

/** Safe parseInt that returns fallback (default 0) on NaN. */
function safeInt(value: string, fallback = 0): number {
  const n = parseInt(value, 10);
  return isNaN(n) ? fallback : n;
}

/**
 * Label VRAM utilization based on used/total MiB.
 * - >= 90%: NEAR_OOM
 * - >= 75%: OPTIMAL
 * - >= 60%: MODERATE
 * - >= 30%: UNDERUTILIZED
 * - < 30%: IDLE
 */
export function labelVramUsage(usedMb: number, totalMb: number): VramLabel {
  if (totalMb <= 0) return "IDLE";
  const pct = Math.round((usedMb / totalMb) * 100 * 10) / 10;
  if (pct >= 90) return "NEAR_OOM";
  if (pct >= 75) return "OPTIMAL";
  if (pct >= 60) return "MODERATE";
  if (pct >= 30) return "UNDERUTILIZED";
  return "IDLE";
}

/** Calculate VRAM usage percentage with one decimal precision. */
export function calcVramPct(usedMb: number, totalMb: number): number {
  if (totalMb <= 0) return 0;
  return Math.round((usedMb / totalMb) * 100 * 10) / 10;
}

/**
 * Parse nvidia-smi CSV output into structured GPU metrics.
 * Expects lines from: nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits
 * Filters out non-data lines (warnings, errors) by requiring first field to be numeric.
 */
export function parseNvidiaSmiOutput(csv: string): GpuMetrics[] {
  const lines = csv.split("\n").filter((l) => l.trim());
  const gpus: GpuMetrics[] = [];

  for (const line of lines) {
    const parts = line.split(",").map((s) => s.trim());
    if (parts.length < 8) continue;

    // Filter non-data lines: first field must be a valid GPU index
    const index = safeInt(parts[0], -1);
    if (index < 0) continue;

    const totalMb = safeInt(parts[2]);
    const usedMb = safeInt(parts[3]);

    gpus.push({
      index,
      name: parts[1],
      totalMb,
      usedMb,
      freeMb: safeInt(parts[4]),
      usedPct: calcVramPct(usedMb, totalMb),
      gpuUtil: safeInt(parts[5]),
      memUtil: safeInt(parts[6]),
      temp: safeInt(parts[7]),
      label: labelVramUsage(usedMb, totalMb),
    });
  }

  return gpus;
}

/**
 * Calculate suggested batch size targeting ~82% VRAM utilization.
 * Returns 0 when per-sample memory exceeds the VRAM target.
 */
export function calcSuggestedBatchSize(totalMb: number, perSampleMb: number): number {
  if (perSampleMb <= 0 || totalMb <= 0) return 0;
  return Math.floor((totalMb * 0.82) / perSampleMb);
}

/**
 * Check if a GPU is overprovisioned relative to the VRAM requirement.
 * Returns true when GPU VRAM is >= 2x the requested minimum.
 */
export function isOverprovisioned(gpuVramGb: number, minVramGb: number): boolean {
  return gpuVramGb >= minVramGb * 2;
}

/**
 * Inject PyTorch optimization env vars into an env record.
 * Returns a new object — does not mutate the input.
 * If PYTORCH_CUDA_ALLOC_CONF is already set, preserves the user's value.
 */
export function injectPytorchEnv(
  env: Record<string, string> | undefined,
  optimize: boolean
): Record<string, string> {
  const result = { ...env };
  if (optimize && !result.PYTORCH_CUDA_ALLOC_CONF) {
    result.PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True";
  }
  return result;
}
