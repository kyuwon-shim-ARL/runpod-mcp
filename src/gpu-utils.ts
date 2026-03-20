/** Pure utility functions for GPU monitoring and pricing — extracted for testability. */

import type { GpuType } from "./types.js";

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
export type TrendVerdict = "STABLE_OPTIMAL" | "STABLE_UNDERUTILIZED" | "IMPROVING" | "DEGRADING" | "CONSISTENTLY_IDLE" | "VOLATILE";

/**
 * Summarize a series of GPU metric snapshots into a trend verdict.
 * Each sample is the primary GPU's metrics from one nvidia-smi call.
 */
export function summarizeTrend(samples: GpuMetrics[]): {
  verdict: TrendVerdict;
  avgVramPct: number;
  avgGpuUtil: number;
  minVramPct: number;
  maxVramPct: number;
} {
  if (!samples.length) return { verdict: "CONSISTENTLY_IDLE", avgVramPct: 0, avgGpuUtil: 0, minVramPct: 0, maxVramPct: 0 };

  const vramPcts = samples.map((s) => s.usedPct);
  const gpuUtils = samples.map((s) => s.gpuUtil);
  const avgVramPct = Math.round(vramPcts.reduce((a, b) => a + b, 0) / vramPcts.length * 10) / 10;
  const avgGpuUtil = Math.round(gpuUtils.reduce((a, b) => a + b, 0) / gpuUtils.length);
  const minVramPct = Math.min(...vramPcts);
  const maxVramPct = Math.max(...vramPcts);

  // All samples idle
  if (samples.every((s) => s.label === "IDLE")) return { verdict: "CONSISTENTLY_IDLE", avgVramPct, avgGpuUtil, minVramPct, maxVramPct };

  // Check volatility (>20% swing)
  if (maxVramPct - minVramPct > 20) return { verdict: "VOLATILE", avgVramPct, avgGpuUtil, minVramPct, maxVramPct };

  // Trend direction: compare first half vs second half
  const mid = Math.floor(samples.length / 2);
  const firstHalfAvg = vramPcts.slice(0, mid || 1).reduce((a, b) => a + b, 0) / (mid || 1);
  const secondHalfAvg = vramPcts.slice(mid).reduce((a, b) => a + b, 0) / (samples.length - mid);

  if (secondHalfAvg > firstHalfAvg + 5) return { verdict: "IMPROVING", avgVramPct, avgGpuUtil, minVramPct, maxVramPct };
  if (secondHalfAvg < firstHalfAvg - 5) return { verdict: "DEGRADING", avgVramPct, avgGpuUtil, minVramPct, maxVramPct };

  // Distinguish stable-optimal from stable-underutilized
  if (avgVramPct < 60) return { verdict: "STABLE_UNDERUTILIZED", avgVramPct, avgGpuUtil, minVramPct, maxVramPct };
  return { verdict: "STABLE_OPTIMAL", avgVramPct, avgGpuUtil, minVramPct, maxVramPct };
}

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

// ── GPU Pricing/Stock Fallback Helpers ──

/** Extract stock status with fallback when lowestPrice API is down */
export function getStockStatus(gpu: GpuType): string {
  if (gpu.lowestPrice?.stockStatus) return gpu.lowestPrice.stockStatus;
  if (gpu.communityCloud || gpu.secureCloud) return "available";
  return "unknown";
}

/** Check if GPU is in stock (not "Out of Stock" or "Low") */
export function isInStock(gpu: GpuType): boolean {
  const status = getStockStatus(gpu);
  return status !== "Out of Stock" && status !== "Low";
}

/** Get spot/bid price with fallback to communitySpotPrice */
export function getSpotPrice(gpu: GpuType): number | null {
  return gpu.lowestPrice?.minimumBidPrice ?? gpu.communitySpotPrice ?? null;
}

/** Get on-demand price with fallback to communityPrice/securePrice */
export function getOnDemandPrice(gpu: GpuType): number | null {
  return gpu.lowestPrice?.uninterruptablePrice ?? gpu.communityPrice ?? gpu.securePrice ?? null;
}
