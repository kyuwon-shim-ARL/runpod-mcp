import { describe, it, expect, vi } from "vitest";
import { filterStalePods, selectGpuCandidates, deletePodWithStop } from "../pod-ops.js";
import type { Pod, GpuType } from "../types.js";

// ── Helper factories ──

function makePod(overrides: Partial<Pod> = {}): Pod {
  return {
    id: "pod-1",
    name: "test-pod",
    desiredStatus: "EXITED",
    lastStatusChange: new Date(Date.now() - 5 * 3600_000).toISOString(),
    ...overrides,
  };
}

function makeGpu(overrides: Partial<GpuType> = {}): GpuType {
  return {
    id: "NVIDIA GeForce RTX 3090",
    displayName: "RTX 3090",
    memoryInGb: 24,
    secureCloud: true,
    communityCloud: true,
    lowestPrice: {
      minimumBidPrice: 0.2,
      uninterruptablePrice: 0.44,
      stockStatus: "High",
    },
    ...overrides,
  };
}

// ── Integration: filterStalePods → deletePodWithStop pipeline ──

describe("orchestration: stale detection → deletion pipeline", () => {
  it("filters stale pods then deletes them via deletePodWithStop", async () => {
    const now = Date.now();
    const pods = [
      makePod({ id: "stale-1", name: "old-pod", lastStatusChange: new Date(now - 10 * 3600_000).toISOString() }),
      makePod({ id: "keep-1", name: "keep-important", desiredStatus: "EXITED" }),
      makePod({ id: "running-1", desiredStatus: "RUNNING" }),
    ];

    // Step 1: filter
    const { stale, skipped } = filterStalePods(pods, 2, now);
    expect(stale).toHaveLength(1);
    expect(stale[0].pod.id).toBe("stale-1");
    expect(skipped).toHaveLength(2);

    // Step 2: delete each stale pod
    const client = {
      getPod: vi.fn().mockResolvedValue({ id: "stale-1", desiredStatus: "EXITED" }),
      stopPod: vi.fn().mockResolvedValue(undefined),
      deletePod: vi.fn().mockResolvedValue(undefined),
    } as any;

    const { wasRunning } = await deletePodWithStop(client, stale[0].pod.id);
    expect(wasRunning).toBe(false);
    expect(client.deletePod).toHaveBeenCalledWith("stale-1");
  });

  it("handles stale pod that is still RUNNING (auto-stop)", async () => {
    const now = Date.now();
    // Edge case: pod has EXITED in desiredStatus but is actually RUNNING when we try to delete
    const pods = [
      makePod({ id: "zombie-1", name: "zombie", lastStatusChange: new Date(now - 8 * 3600_000).toISOString() }),
    ];

    const { stale } = filterStalePods(pods, 2, now);
    expect(stale).toHaveLength(1);

    let callCount = 0;
    const client = {
      getPod: vi.fn().mockImplementation(async () => {
        callCount++;
        if (callCount === 1) return { id: "zombie-1", desiredStatus: "RUNNING" };
        return { id: "zombie-1", desiredStatus: "EXITED" };
      }),
      stopPod: vi.fn().mockResolvedValue(undefined),
      deletePod: vi.fn().mockResolvedValue(undefined),
    } as any;

    const { wasRunning } = await deletePodWithStop(client, "zombie-1", 5_000);
    expect(wasRunning).toBe(true);
    expect(client.stopPod).toHaveBeenCalledWith("zombie-1");
    expect(client.deletePod).toHaveBeenCalledWith("zombie-1");
  });
});

// ── Integration: selectGpuCandidates → create_pod_auto flow ──

describe("orchestration: GPU selection → pod creation flow", () => {
  it("selects cheapest viable GPU from preference list", () => {
    const gpus = [
      makeGpu({ id: "GPU-EXPENSIVE", displayName: "A100", memoryInGb: 80, lowestPrice: { minimumBidPrice: 1.0, uninterruptablePrice: 3.0, stockStatus: "High" } }),
      makeGpu({ id: "GPU-CHEAP", displayName: "RTX 3090", memoryInGb: 24, lowestPrice: { minimumBidPrice: 0.2, uninterruptablePrice: 0.44, stockStatus: "High" } }),
      makeGpu({ id: "GPU-TINY", displayName: "RTX 3060", memoryInGb: 12, lowestPrice: { minimumBidPrice: 0.1, uninterruptablePrice: 0.2, stockStatus: "High" } }),
    ];

    // User prefers cheap → expensive
    const { candidates } = selectGpuCandidates(gpus, {
      gpuPreference: ["GPU-CHEAP", "GPU-TINY", "GPU-EXPENSIVE"],
      minVram: 20,
      gpuCount: 1,
      spot: false,
      maxBidPerGpu: 0.5,
    });

    // GPU-TINY (12GB) is below minVram 20GB, so should be filtered
    expect(candidates).toHaveLength(2);
    expect(candidates[0].gpuId).toBe("GPU-CHEAP");
    expect(candidates[1].gpuId).toBe("GPU-EXPENSIVE");
  });

  it("spot bidding: skips GPUs where bid is below minimum", () => {
    const gpus = [
      makeGpu({ id: "GPU-A", displayName: "A", memoryInGb: 24, lowestPrice: { minimumBidPrice: 0.5, uninterruptablePrice: 1.0, stockStatus: "High" } }),
      makeGpu({ id: "GPU-B", displayName: "B", memoryInGb: 24, lowestPrice: { minimumBidPrice: 0.1, uninterruptablePrice: 0.3, stockStatus: "High" } }),
    ];

    const { candidates, errors } = selectGpuCandidates(gpus, {
      gpuPreference: ["GPU-A", "GPU-B"],
      minVram: 12,
      gpuCount: 1,
      spot: true,
      maxBidPerGpu: 0.2, // Below GPU-A's min bid of 0.5
    });

    // GPU-A should be skipped (bid 0.2 < min 0.5), GPU-B should work (bid 0.2 > min 0.1)
    expect(candidates).toHaveLength(1);
    expect(candidates[0].gpuId).toBe("GPU-B");
    expect(errors).toHaveLength(1);
    expect(errors[0]).toContain("A:"); // uses displayName
  });

  it("all GPUs out of stock returns empty candidates", () => {
    const gpus = [
      makeGpu({ id: "GPU-A", lowestPrice: { minimumBidPrice: 0, uninterruptablePrice: 0, stockStatus: "Out of Stock" } }),
      makeGpu({ id: "GPU-B", lowestPrice: { minimumBidPrice: 0, uninterruptablePrice: 0, stockStatus: "Out of Stock" } }),
    ];

    const { candidates, errors } = selectGpuCandidates(gpus, {
      gpuPreference: ["GPU-A", "GPU-B"],
      minVram: 12,
      gpuCount: 1,
      spot: false,
      maxBidPerGpu: 0.5,
    });

    expect(candidates).toHaveLength(0);
    expect(errors).toHaveLength(0);
  });

  it("warns on overprovisioned GPU selection", () => {
    const gpus = [
      makeGpu({ id: "GPU-BIG", displayName: "A100 80GB", memoryInGb: 80 }),
    ];

    const { candidates } = selectGpuCandidates(gpus, {
      gpuPreference: ["GPU-BIG"],
      minVram: 16, // 80GB >> 16GB needed
      gpuCount: 1,
      spot: false,
      maxBidPerGpu: 5.0,
    });

    expect(candidates).toHaveLength(1);
    expect(candidates[0].overprovisionWarning).toContain("Overprovisioned");
    expect(candidates[0].overprovisionWarning).toContain("80GB");
  });
});

// ── Edge cases: deletePodWithStop timeout ──

describe("deletePodWithStop edge cases", () => {
  it("proceeds with delete even if pod never transitions from RUNNING (timeout)", async () => {
    const client = {
      getPod: vi.fn().mockResolvedValue({ id: "stuck-pod", desiredStatus: "RUNNING" }),
      stopPod: vi.fn().mockResolvedValue(undefined),
      deletePod: vi.fn().mockResolvedValue(undefined),
    } as any;

    // Very short timeout to trigger timeout path
    const { wasRunning } = await deletePodWithStop(client, "stuck-pod", 100);
    expect(wasRunning).toBe(true);
    expect(client.stopPod).toHaveBeenCalled();
    expect(client.deletePod).toHaveBeenCalled();
  });

  it("handles delete failure gracefully", async () => {
    const client = {
      getPod: vi.fn().mockResolvedValue({ id: "pod-1", desiredStatus: "EXITED" }),
      stopPod: vi.fn(),
      deletePod: vi.fn().mockRejectedValue(new Error("API error: pod not found")),
    } as any;

    await expect(deletePodWithStop(client, "pod-1")).rejects.toThrow("API error: pod not found");
  });
});
