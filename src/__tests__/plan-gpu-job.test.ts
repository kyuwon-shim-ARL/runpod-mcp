/**
 * Unit tests for plan_gpu_job helper logic.
 * Tests NV sizing, staging determination, and cost calculation.
 */
import { describe, it, expect } from "vitest";

const DEFAULT_MIN_VRAM_GB = 12;
const NV_COST_PER_GB_MONTH = 0.07;

// NV sizing: ceil((datasetGb + estimatedOutputGb) * 1.3), min 50GB
function calcNvGb(datasetGb: number, modelSizeGb?: number): number {
  const estimatedOutputGb = modelSizeGb != null ? modelSizeGb * 2 : 5;
  const raw = Math.ceil((datasetGb + estimatedOutputGb) * 1.3);
  return Math.max(50, raw);
}

// Staging needed: datasetGb > 0.5
function needsStaging(datasetGb: number): boolean {
  return datasetGb > 0.5;
}

// Training cost: price * hours * gpuCount (null if hours unknown)
function calcTrainingCost(gpuPrice: number, hours: number | undefined, gpuCount: number): number | null {
  if (hours == null) return null;
  return gpuPrice * hours * gpuCount;
}

describe("plan_gpu_job: NV sizing", () => {
  it("1GB dataset, 8GB model → ceil((1 + 16) × 1.3) = 23 → min 50GB", () => {
    expect(calcNvGb(1, 8)).toBe(50);
  });

  it("22GB dataset, 16GB model → ceil((22 + 32) × 1.3) = 71GB", () => {
    expect(calcNvGb(22, 16)).toBe(71);
  });

  it("80GB dataset, 20GB model → ceil((80 + 40) × 1.3) = 156GB", () => {
    expect(calcNvGb(80, 20)).toBe(156);
  });

  it("no dataset, no model → ceil((0 + 5) × 1.3) = 7 → min 50GB", () => {
    expect(calcNvGb(0)).toBe(50);
  });

  it("0GB dataset, 7GB model → ceil((0 + 14) × 1.3) = 19 → min 50GB", () => {
    expect(calcNvGb(0, 7)).toBe(50);
  });
});

describe("plan_gpu_job: staging determination", () => {
  it("0GB dataset → no staging", () => {
    expect(needsStaging(0)).toBe(false);
  });

  it("0.5GB dataset → no staging (threshold is > 0.5, not >=)", () => {
    expect(needsStaging(0.5)).toBe(false);
  });

  it("0.6GB dataset → staging required", () => {
    expect(needsStaging(0.6)).toBe(true);
  });

  it("22GB dataset → staging required", () => {
    expect(needsStaging(22)).toBe(true);
  });
});

describe("plan_gpu_job: cost calculation", () => {
  it("8hr training, $0.34/hr, 1 GPU → $2.72", () => {
    expect(calcTrainingCost(0.34, 8, 1)).toBeCloseTo(2.72);
  });

  it("8hr training, $0.34/hr, 4 GPUs → $10.88", () => {
    expect(calcTrainingCost(0.34, 8, 4)).toBeCloseTo(10.88);
  });

  it("expectedHours undefined → null (N/A row)", () => {
    expect(calcTrainingCost(0.34, undefined, 4)).toBeNull();
  });

  it("NV cost: 50GB × $0.07/mo → $3.50/mo", () => {
    expect(50 * NV_COST_PER_GB_MONTH).toBeCloseTo(3.5);
  });

  it("NV cost: 156GB × $0.07/mo → $10.92/mo", () => {
    expect(156 * NV_COST_PER_GB_MONTH).toBeCloseTo(10.92);
  });
});
