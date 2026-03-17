import { describe, it, expect } from "vitest";
import {
  labelVramUsage,
  calcVramPct,
  parseNvidiaSmiOutput,
  calcSuggestedBatchSize,
  isOverprovisioned,
  injectPytorchEnv,
} from "../gpu-utils.js";

// ── labelVramUsage ──

describe("labelVramUsage", () => {
  // NEAR_OOM boundary: >= 90%
  it("returns NEAR_OOM at exactly 90.0%", () => {
    // 9216/10240 = 90.0%
    expect(labelVramUsage(9216, 10240)).toBe("NEAR_OOM");
  });
  it("returns NEAR_OOM above 90%", () => {
    expect(labelVramUsage(9500, 10240)).toBe("NEAR_OOM");
  });
  it("returns NEAR_OOM at 99%", () => {
    expect(labelVramUsage(24000, 24576)).toBe("NEAR_OOM");
  });

  // OPTIMAL boundary: >= 75% and < 90%
  it("returns OPTIMAL at exactly 75%", () => {
    expect(labelVramUsage(7680, 10240)).toBe("OPTIMAL");
  });
  it("returns OPTIMAL at 89.9%", () => {
    // 9205/10240 ≈ 89.9% (rounds to 89.9)
    expect(labelVramUsage(9205, 10240)).toBe("OPTIMAL");
  });

  // MODERATE boundary: >= 60% and < 75%
  it("returns MODERATE at exactly 60%", () => {
    expect(labelVramUsage(6144, 10240)).toBe("MODERATE");
  });
  it("returns MODERATE at 74.9%", () => {
    // 7669/10240 * 1000 = 749.12 → rounds to 749 → 74.9%
    expect(labelVramUsage(7669, 10240)).toBe("MODERATE");
  });

  // UNDERUTILIZED boundary: >= 30% and < 60%
  it("returns UNDERUTILIZED at exactly 30%", () => {
    expect(labelVramUsage(3072, 10240)).toBe("UNDERUTILIZED");
  });
  it("returns UNDERUTILIZED at 59.9%", () => {
    // 6133/10240 * 1000 = 598.93 → rounds to 599 → 59.9%
    expect(labelVramUsage(6133, 10240)).toBe("UNDERUTILIZED");
  });

  // IDLE boundary: < 30%
  it("returns IDLE below 30%", () => {
    // 3061/10240 * 1000 = 298.93 → rounds to 299 → 29.9%
    expect(labelVramUsage(3061, 10240)).toBe("IDLE");
  });
  it("returns IDLE at 0% usage", () => {
    expect(labelVramUsage(0, 24576)).toBe("IDLE");
  });

  // Edge cases
  it("returns IDLE when total is 0 (division guard)", () => {
    expect(labelVramUsage(0, 0)).toBe("IDLE");
  });
  it("returns IDLE when total is negative", () => {
    expect(labelVramUsage(100, -1)).toBe("IDLE");
  });
});

// ── calcVramPct ──

describe("calcVramPct", () => {
  it("calculates percentage with one decimal", () => {
    expect(calcVramPct(19661, 24576)).toBeCloseTo(80.0, 0);
  });
  it("returns 0 when total is 0", () => {
    expect(calcVramPct(100, 0)).toBe(0);
  });
  it("handles exact 50%", () => {
    expect(calcVramPct(5120, 10240)).toBe(50.0);
  });
});

// ── parseNvidiaSmiOutput ──

describe("parseNvidiaSmiOutput", () => {
  const SINGLE_GPU_IDLE = "0, NVIDIA GeForce RTX 3090, 24576, 512, 24064, 2, 1, 35";
  const SINGLE_GPU_OPTIMAL = "0, NVIDIA GeForce RTX 3090, 24576, 19661, 4915, 95, 88, 78";
  const SINGLE_GPU_NEAR_OOM = "0, NVIDIA A100-SXM4-80GB, 81920, 75899, 6021, 99, 97, 82";
  const MULTI_GPU = [
    "0, NVIDIA GeForce RTX 3090, 24576, 19661, 4915, 95, 88, 72",
    "1, NVIDIA GeForce RTX 3090, 24576, 18000, 6576, 90, 82, 70",
  ].join("\n");

  it("parses single GPU idle correctly", () => {
    const gpus = parseNvidiaSmiOutput(SINGLE_GPU_IDLE);
    expect(gpus).toHaveLength(1);
    expect(gpus[0].index).toBe(0);
    expect(gpus[0].name).toBe("NVIDIA GeForce RTX 3090");
    expect(gpus[0].totalMb).toBe(24576);
    expect(gpus[0].usedMb).toBe(512);
    expect(gpus[0].label).toBe("IDLE");
    expect(gpus[0].temp).toBe(35);
  });

  it("parses optimal GPU correctly", () => {
    const gpus = parseNvidiaSmiOutput(SINGLE_GPU_OPTIMAL);
    expect(gpus[0].label).toBe("OPTIMAL");
    expect(gpus[0].gpuUtil).toBe(95);
  });

  it("parses near-OOM GPU correctly", () => {
    const gpus = parseNvidiaSmiOutput(SINGLE_GPU_NEAR_OOM);
    expect(gpus[0].label).toBe("NEAR_OOM");
    expect(gpus[0].totalMb).toBe(81920);
  });

  it("parses multi-GPU output", () => {
    const gpus = parseNvidiaSmiOutput(MULTI_GPU);
    expect(gpus).toHaveLength(2);
    expect(gpus[0].index).toBe(0);
    expect(gpus[1].index).toBe(1);
  });

  it("filters out non-data lines (warnings)", () => {
    const withWarning = "WARNING: infoROM is corrupted\n" + SINGLE_GPU_IDLE;
    const gpus = parseNvidiaSmiOutput(withWarning);
    expect(gpus).toHaveLength(1);
    expect(gpus[0].name).toBe("NVIDIA GeForce RTX 3090");
  });

  it("filters out short lines (< 8 fields)", () => {
    const shortLine = "0, RTX 3090, 24576";
    const gpus = parseNvidiaSmiOutput(shortLine);
    expect(gpus).toHaveLength(0);
  });

  it("returns empty array for empty input", () => {
    expect(parseNvidiaSmiOutput("")).toHaveLength(0);
    expect(parseNvidiaSmiOutput("  \n  ")).toHaveLength(0);
  });

  it("handles N/A fields gracefully (falls back to 0)", () => {
    const naLine = "0, Tesla T4, 16384, 8000, 8384, [N/A], [N/A], 45";
    const gpus = parseNvidiaSmiOutput(naLine);
    expect(gpus).toHaveLength(1);
    expect(gpus[0].gpuUtil).toBe(0); // NaN → 0
    expect(gpus[0].memUtil).toBe(0);
    expect(gpus[0].temp).toBe(45);
    expect(gpus[0].usedMb).toBe(8000);
  });

  it("handles GPU names with extra spaces", () => {
    const spaceyName = "0,  Tesla V100-SXM2-32GB , 32768, 20000, 12768, 85, 70, 65";
    const gpus = parseNvidiaSmiOutput(spaceyName);
    expect(gpus[0].name).toBe("Tesla V100-SXM2-32GB");
  });

  it("handles nvidia-smi not found output", () => {
    const gpus = parseNvidiaSmiOutput("nvidia-smi: command not found");
    expect(gpus).toHaveLength(0);
  });
});

// ── calcSuggestedBatchSize ──

describe("calcSuggestedBatchSize", () => {
  it("calculates correct batch size for RTX 3090 with 256 MiB samples", () => {
    // floor(24576 * 0.82 / 256) = floor(78.72) = 78
    expect(calcSuggestedBatchSize(24576, 256)).toBe(78);
  });

  it("calculates correct batch size for 10GB GPU with 128 MiB samples", () => {
    // floor(10240 * 0.82 / 128) = floor(65.6) = 65
    expect(calcSuggestedBatchSize(10240, 128)).toBe(65);
  });

  it("returns 0 when per-sample exceeds VRAM target", () => {
    // floor(24576 * 0.82 / 30000) = floor(0.67) = 0
    expect(calcSuggestedBatchSize(24576, 30000)).toBe(0);
  });

  it("returns 0 when per-sample equals VRAM", () => {
    // floor(10240 * 0.82 / 10240) = floor(0.82) = 0
    expect(calcSuggestedBatchSize(10240, 10240)).toBe(0);
  });

  it("returns 0 for zero totalMb", () => {
    expect(calcSuggestedBatchSize(0, 256)).toBe(0);
  });

  it("returns 0 for zero perSampleMb", () => {
    expect(calcSuggestedBatchSize(24576, 0)).toBe(0);
  });

  it("returns 0 for negative perSampleMb", () => {
    expect(calcSuggestedBatchSize(24576, -10)).toBe(0);
  });

  it("handles A100 80GB with small samples", () => {
    // floor(81920 * 0.82 / 64) = floor(1049.6) = 1049
    expect(calcSuggestedBatchSize(81920, 64)).toBe(1049);
  });
});

// ── isOverprovisioned ──

describe("isOverprovisioned", () => {
  it("returns false when exact 1x", () => {
    expect(isOverprovisioned(12, 12)).toBe(false);
  });

  it("returns true when exactly 2x (>= threshold)", () => {
    expect(isOverprovisioned(24, 12)).toBe(true);
  });

  it("returns false when just below 2x", () => {
    expect(isOverprovisioned(23, 12)).toBe(false);
  });

  it("returns true when well above 2x", () => {
    expect(isOverprovisioned(80, 12)).toBe(true);
  });

  it("returns true for A100 vs 12GB min", () => {
    expect(isOverprovisioned(80, 12)).toBe(true);
  });

  it("returns false when equal", () => {
    expect(isOverprovisioned(24, 24)).toBe(false);
  });

  it("handles 0 minVram (edge)", () => {
    expect(isOverprovisioned(24, 0)).toBe(true);
  });
});

// ── injectPytorchEnv ──

describe("injectPytorchEnv", () => {
  it("injects env var when optimize=true and key absent", () => {
    const result = injectPytorchEnv({}, true);
    expect(result.PYTORCH_CUDA_ALLOC_CONF).toBe("expandable_segments:True");
  });

  it("preserves user-supplied value when key already exists", () => {
    const result = injectPytorchEnv({ PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:128" }, true);
    expect(result.PYTORCH_CUDA_ALLOC_CONF).toBe("max_split_size_mb:128");
  });

  it("does not inject when optimize=false", () => {
    const result = injectPytorchEnv({}, false);
    expect(result.PYTORCH_CUDA_ALLOC_CONF).toBeUndefined();
  });

  it("preserves all other env vars", () => {
    const result = injectPytorchEnv({ MY_VAR: "hello", OTHER: "world" }, true);
    expect(result.MY_VAR).toBe("hello");
    expect(result.OTHER).toBe("world");
    expect(result.PYTORCH_CUDA_ALLOC_CONF).toBe("expandable_segments:True");
  });

  it("does not mutate original env object", () => {
    const original = { FOO: "bar" };
    const result = injectPytorchEnv(original, true);
    expect(original).not.toHaveProperty("PYTORCH_CUDA_ALLOC_CONF");
    expect(result.PYTORCH_CUDA_ALLOC_CONF).toBe("expandable_segments:True");
  });

  it("handles undefined env", () => {
    const result = injectPytorchEnv(undefined, true);
    expect(result.PYTORCH_CUDA_ALLOC_CONF).toBe("expandable_segments:True");
  });

  it("handles undefined env with optimize=false", () => {
    const result = injectPytorchEnv(undefined, false);
    expect(Object.keys(result)).toHaveLength(0);
  });
});
