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
