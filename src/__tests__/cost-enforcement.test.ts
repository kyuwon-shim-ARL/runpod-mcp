/**
 * Unit tests for GPU Cost Safety Phase 1 enforcement logic.
 * Covers: nvReadinessToken gate, verify_data_on_nv token TTL,
 * run_preflight checks, get_pipeline_events WATCHER_EXITED,
 * plan_gpu_job container disk warning.
 */
import { describe, it, expect } from "vitest";

// ── Constants (mirrors src/index.ts) ──────────────────────────────────────
const TOKEN_TTL_HOURS = 72;
const COST_GATE_GPU_COUNT = 2;
const NV_READY_DIR = ".omc/gpu-exec";
const DEFAULT_CONTAINER_DISK_GB = 30;
const CRITICAL_ML = ["peft", "transformers", "torch", "torchaudio", "torchvision", "bitsandbytes", "accelerate", "datasets"];

// ══════════════════════════════════════════════════════════════════════════
// verify_data_on_nv — token logic
// ══════════════════════════════════════════════════════════════════════════

type TokenData = { token: string; nvId: string; verifiedAt: string; totalGb: number; paths: string[] };

function validateTokenTTL(tokenData: TokenData, providedToken: string, nowMs: number): "PASS" | "TOKEN_MISMATCH" | "EXPIRED" {
  if (tokenData.token !== providedToken) return "TOKEN_MISMATCH";
  const ageHours = (nowMs - new Date(tokenData.verifiedAt).getTime()) / 3_600_000;
  if (ageHours > TOKEN_TTL_HOURS) return "EXPIRED";
  return "PASS";
}

function checkTruncation(totalGb: number, minTotalGb: number | undefined): boolean {
  if (minTotalGb == null) return false;
  return totalGb < minTotalGb;
}

describe("verify_data_on_nv: token TTL validation", () => {
  const base: TokenData = {
    token: "abc-123",
    nvId: "nv-001",
    verifiedAt: "2026-04-16T10:00:00Z",
    totalGb: 25.5,
    paths: ["data/train.jsonl", "data/val.jsonl"],
  };
  const baseNow = new Date("2026-04-16T10:00:00Z").getTime();

  it("matching token, 0h old → PASS", () => {
    expect(validateTokenTTL(base, "abc-123", baseNow)).toBe("PASS");
  });

  it("matching token, 71h old → PASS (within 72h TTL)", () => {
    const now = baseNow + 71 * 3_600_000;
    expect(validateTokenTTL(base, "abc-123", now)).toBe("PASS");
  });

  it("matching token, 72.1h old → EXPIRED", () => {
    const now = baseNow + 72.1 * 3_600_000;
    expect(validateTokenTTL(base, "abc-123", now)).toBe("EXPIRED");
  });

  it("wrong token → TOKEN_MISMATCH (regardless of age)", () => {
    expect(validateTokenTTL(base, "wrong-token", baseNow)).toBe("TOKEN_MISMATCH");
  });

  it("empty string token → TOKEN_MISMATCH", () => {
    expect(validateTokenTTL(base, "", baseNow)).toBe("TOKEN_MISMATCH");
  });
});

describe("verify_data_on_nv: truncation detection", () => {
  it("totalGb >= minTotalGb → no truncation", () => {
    expect(checkTruncation(25.5, 20)).toBe(false);
  });

  it("totalGb < minTotalGb → truncation detected", () => {
    expect(checkTruncation(15.0, 20)).toBe(true);
  });

  it("minTotalGb undefined → no check, no truncation", () => {
    expect(checkTruncation(0.001, undefined)).toBe(false);
  });

  it("totalGb exactly equals minTotalGb → no truncation (boundary)", () => {
    expect(checkTruncation(20.0, 20)).toBe(false);
  });
});

// ══════════════════════════════════════════════════════════════════════════
// create_pod_auto — nvReadinessToken gate
// ══════════════════════════════════════════════════════════════════════════

type NvTokenGateArgs = {
  gpuCount: number;
  networkVolumeId?: string;
  nvReadinessToken?: string;
  dryRun: boolean;
};

type NvTokenGateState = {
  tokenData?: TokenData;
  fileExists: boolean;
};

function nvTokenGate(
  args: NvTokenGateArgs,
  state: NvTokenGateState,
  nowMs: number
): "PASS" | "TOKEN_REQUIRED" | "FILE_NOT_FOUND" | "TOKEN_MISMATCH" | "EXPIRED" {
  if (args.gpuCount < COST_GATE_GPU_COUNT || !args.networkVolumeId || args.dryRun) return "PASS";
  if (!args.nvReadinessToken) return "TOKEN_REQUIRED";
  if (!state.fileExists || !state.tokenData) return "FILE_NOT_FOUND";
  return validateTokenTTL(state.tokenData, args.nvReadinessToken, nowMs);
}

describe("create_pod_auto: nvReadinessToken gate", () => {
  const nowMs = new Date("2026-04-16T12:00:00Z").getTime();
  const goodToken: TokenData = {
    token: "tok-xyz",
    nvId: "nv-aaa",
    verifiedAt: "2026-04-16T10:00:00Z",
    totalGb: 30,
    paths: ["data/train.jsonl"],
  };

  it("gpuCount=1, no NV → gate skipped (PASS)", () => {
    expect(nvTokenGate({ gpuCount: 1, networkVolumeId: "nv-aaa", nvReadinessToken: undefined, dryRun: false }, { fileExists: false }, nowMs)).toBe("PASS");
  });

  it("gpuCount=2, no NV → gate skipped (PASS)", () => {
    expect(nvTokenGate({ gpuCount: 2, networkVolumeId: undefined, nvReadinessToken: undefined, dryRun: false }, { fileExists: false }, nowMs)).toBe("PASS");
  });

  it("gpuCount=2, NV set, dryRun=true → gate skipped (PASS)", () => {
    expect(nvTokenGate({ gpuCount: 2, networkVolumeId: "nv-aaa", nvReadinessToken: undefined, dryRun: true }, { fileExists: false }, nowMs)).toBe("PASS");
  });

  it("gpuCount=2, NV set, no token → TOKEN_REQUIRED", () => {
    expect(nvTokenGate({ gpuCount: 2, networkVolumeId: "nv-aaa", nvReadinessToken: undefined, dryRun: false }, { fileExists: true, tokenData: goodToken }, nowMs)).toBe("TOKEN_REQUIRED");
  });

  it("gpuCount=2, NV set, token provided but file missing → FILE_NOT_FOUND", () => {
    expect(nvTokenGate({ gpuCount: 2, networkVolumeId: "nv-aaa", nvReadinessToken: "tok-xyz", dryRun: false }, { fileExists: false }, nowMs)).toBe("FILE_NOT_FOUND");
  });

  it("gpuCount=2, NV set, valid token → PASS", () => {
    expect(nvTokenGate({ gpuCount: 2, networkVolumeId: "nv-aaa", nvReadinessToken: "tok-xyz", dryRun: false }, { fileExists: true, tokenData: goodToken }, nowMs)).toBe("PASS");
  });

  it("gpuCount=4, NV set, valid token → PASS (≥2 GPU all require token)", () => {
    expect(nvTokenGate({ gpuCount: 4, networkVolumeId: "nv-aaa", nvReadinessToken: "tok-xyz", dryRun: false }, { fileExists: true, tokenData: goodToken }, nowMs)).toBe("PASS");
  });

  it("gpuCount=2, NV set, expired token → EXPIRED", () => {
    const expiredNow = nowMs + 80 * 3_600_000;
    expect(nvTokenGate({ gpuCount: 2, networkVolumeId: "nv-aaa", nvReadinessToken: "tok-xyz", dryRun: false }, { fileExists: true, tokenData: goodToken }, expiredNow)).toBe("EXPIRED");
  });

  it("gpuCount=2, NV set, wrong token → TOKEN_MISMATCH", () => {
    expect(nvTokenGate({ gpuCount: 2, networkVolumeId: "nv-aaa", nvReadinessToken: "bad-tok", dryRun: false }, { fileExists: true, tokenData: goodToken }, nowMs)).toBe("TOKEN_MISMATCH");
  });
});

// ══════════════════════════════════════════════════════════════════════════
// run_preflight — requirements.txt pinning check
// ══════════════════════════════════════════════════════════════════════════

type PinResult = { pkg: string; issue: "unpinned" | "unbounded" } | null;

function checkRequirementsLine(line: string): PinResult {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) return null;
  const match = trimmed.match(/^([a-zA-Z0-9_-]+)(.*)$/);
  if (!match) return null;
  const pkg = match[1].toLowerCase();
  if (!CRITICAL_ML.includes(pkg)) return null;
  const spec = match[2].trim();
  if (!spec) return { pkg, issue: "unpinned" };
  if (/^>=/.test(spec) && !spec.includes(",<")) return { pkg, issue: "unbounded" };
  return null;
}

describe("run_preflight: requirements.txt pinning", () => {
  it("torch==2.1.0 → no issue (exact pin)", () => {
    expect(checkRequirementsLine("torch==2.1.0")).toBeNull();
  });

  it("torch>=2.0.0,<3.0.0 → no issue (bounded range)", () => {
    expect(checkRequirementsLine("torch>=2.0.0,<3.0.0")).toBeNull();
  });

  it("torch (no spec) → unpinned warning", () => {
    expect(checkRequirementsLine("torch")).toEqual({ pkg: "torch", issue: "unpinned" });
  });

  it("transformers>=4.30.0 (no upper bound) → unbounded warning", () => {
    expect(checkRequirementsLine("transformers>=4.30.0")).toEqual({ pkg: "transformers", issue: "unbounded" });
  });

  it("requests>=2.28 → no issue (not in CRITICAL_ML)", () => {
    expect(checkRequirementsLine("requests>=2.28")).toBeNull();
  });

  it("# comment line → no issue", () => {
    expect(checkRequirementsLine("# torch==2.1.0")).toBeNull();
  });

  it("peft==0.6.0 → no issue", () => {
    expect(checkRequirementsLine("peft==0.6.0")).toBeNull();
  });

  it("bitsandbytes → unpinned", () => {
    expect(checkRequirementsLine("bitsandbytes")).toEqual({ pkg: "bitsandbytes", issue: "unpinned" });
  });
});

describe("run_preflight: disk free check", () => {
  function diskCheck(freeGb: number, minDiskFreeGb: number): "PASS" | "FAIL" {
    if (freeGb < 0) return "FAIL";
    return freeGb < minDiskFreeGb ? "FAIL" : "PASS";
  }

  it("30GB free, min=10GB → PASS", () => {
    expect(diskCheck(30, 10)).toBe("PASS");
  });

  it("5GB free, min=10GB → FAIL", () => {
    expect(diskCheck(5, 10)).toBe("FAIL");
  });

  it("exactly 10GB free, min=10GB → PASS (boundary)", () => {
    expect(diskCheck(10, 10)).toBe("PASS");
  });

  it("query fails (freeGb=-1) → FAIL", () => {
    expect(diskCheck(-1, 10)).toBe("FAIL");
  });
});

// ══════════════════════════════════════════════════════════════════════════
// get_pipeline_events — WATCHER_EXITED sticky warning
// ══════════════════════════════════════════════════════════════════════════

type PipelineEvent = { event: string; podId?: string; ts?: string; gpuPct?: number; idleCheck?: number; detail?: string; reason?: string };

function detectWatcherExited(events: PipelineEvent[]): PipelineEvent | null {
  const exitedEvents = events.filter(e => e.event === "WATCHER_EXITED");
  return exitedEvents.length > 0 ? exitedEvents[exitedEvents.length - 1] : null;
}

function filterByPod(events: PipelineEvent[], podId: string | undefined): PipelineEvent[] {
  if (!podId) return events;
  return events.filter(e => e.podId === podId);
}

describe("get_pipeline_events: WATCHER_EXITED detection", () => {
  it("no WATCHER_EXITED in events → null", () => {
    const events: PipelineEvent[] = [
      { event: "HEALTH_CHECK", podId: "pod-1", gpuPct: 85 },
      { event: "HEALTH_CHECK", podId: "pod-1", gpuPct: 82 },
    ];
    expect(detectWatcherExited(events)).toBeNull();
  });

  it("WATCHER_EXITED present → returns last occurrence", () => {
    const events: PipelineEvent[] = [
      { event: "HEALTH_CHECK", podId: "pod-1", gpuPct: 85 },
      { event: "WATCHER_EXITED", podId: "pod-1", reason: "GraphQL stop failure" },
    ];
    const result = detectWatcherExited(events);
    expect(result).not.toBeNull();
    expect(result?.event).toBe("WATCHER_EXITED");
    expect(result?.reason).toBe("GraphQL stop failure");
  });

  it("multiple WATCHER_EXITED → returns the last one", () => {
    const events: PipelineEvent[] = [
      { event: "WATCHER_EXITED", podId: "pod-1", reason: "first" },
      { event: "WATCHER_EXITED", podId: "pod-1", reason: "second" },
    ];
    expect(detectWatcherExited(events)?.reason).toBe("second");
  });
});

describe("get_pipeline_events: pod filter", () => {
  const events: PipelineEvent[] = [
    { event: "HEALTH_CHECK", podId: "pod-1", gpuPct: 80 },
    { event: "HEALTH_CHECK", podId: "pod-2", gpuPct: 55 },
    { event: "AUTO_STOPPED", podId: "pod-2" },
  ];

  it("no filter → all events returned", () => {
    expect(filterByPod(events, undefined)).toHaveLength(3);
  });

  it("filter pod-1 → 1 event", () => {
    expect(filterByPod(events, "pod-1")).toHaveLength(1);
  });

  it("filter pod-2 → 2 events", () => {
    expect(filterByPod(events, "pod-2")).toHaveLength(2);
  });

  it("filter unknown pod → 0 events", () => {
    expect(filterByPod(events, "pod-999")).toHaveLength(0);
  });
});

// ══════════════════════════════════════════════════════════════════════════
// plan_gpu_job — container disk warning
// ══════════════════════════════════════════════════════════════════════════

function calcDiskEstimate(modelSizeGb: number, datasetGb: number, gpuCount: number): number {
  return (modelSizeGb + datasetGb * 0.1 + 2) * gpuCount;
}

function diskWarn(modelSizeGb: number, datasetGb: number, gpuCount: number): { warn: boolean; estimateGb: number; recommendedDisk: number } {
  const estimateGb = calcDiskEstimate(modelSizeGb, datasetGb, gpuCount);
  const threshold = DEFAULT_CONTAINER_DISK_GB * 0.7;
  const warn = estimateGb > threshold;
  const recommendedDisk = Math.ceil(estimateGb * 1.5);
  return { warn, estimateGb, recommendedDisk };
}

describe("plan_gpu_job: container disk warning", () => {
  it("small model (5GB), 10GB data, 1 GPU → estimate=8GB, < 21GB threshold → no warn", () => {
    const r = diskWarn(5, 10, 1);
    expect(r.warn).toBe(false);
    expect(r.estimateGb).toBeCloseTo(8.0);
  });

  it("model=10GB, data=100GB, 1 GPU → estimate=22GB > 21GB threshold → warn", () => {
    const r = diskWarn(10, 100, 1);
    expect(r.warn).toBe(true);
    expect(r.estimateGb).toBeCloseTo(22.0);
  });

  it("model=5GB, data=10GB, 4 GPUs → estimate=32GB > 21GB → warn", () => {
    const r = diskWarn(5, 10, 4);
    expect(r.warn).toBe(true);
    expect(r.estimateGb).toBeCloseTo(32.0);
  });

  it("warn → recommendedDisk = ceil(estimateGb * 1.5)", () => {
    const r = diskWarn(10, 100, 1);
    expect(r.recommendedDisk).toBe(Math.ceil(22 * 1.5)); // 33
  });

  it("default 30GB disk, 70% threshold = 21GB boundary: estimate=21 → no warn (boundary)", () => {
    // (modelSizeGb + datasetGb*0.1 + 2) * gpuCount = 21
    // modelSizeGb=5, datasetGb=140, gpuCount=1: (5 + 14 + 2) = 21
    const r = diskWarn(5, 140, 1);
    expect(r.estimateGb).toBeCloseTo(21.0);
    expect(r.warn).toBe(false);
  });

  it("estimate just over threshold (21.1GB) → warn", () => {
    // (5 + 14.1 + 2) * 1 = 21.1
    const r = diskWarn(5, 141, 1);
    expect(r.estimateGb).toBeCloseTo(21.1);
    expect(r.warn).toBe(true);
  });
});

// ══════════════════════════════════════════════════════════════════════════
// run_preflight — CUDA availability check (EXP-047)
// Mirrors the result-processing logic in src/index.ts run_preflight tool.
// ══════════════════════════════════════════════════════════════════════════

type CudaSpawnResult = { status: number | null; stdout: string; stderr: string };
type CudaCheckResult = { label: "CUDA"; status: "✅" | "⚠️" | "❌"; detail: string };

function processCudaResult(r: CudaSpawnResult): { result: CudaCheckResult; isFail: boolean; isWarn: boolean } {
  if (r.status === null) {
    return {
      result: { label: "CUDA", status: "⚠️", detail: "CUDA check timed out (60s) — pod may be cold-starting. Re-run run_preflight." },
      isFail: false,
      isWarn: true,
    };
  }
  const lines = (r.stdout ?? "").split("\n");
  const cudaLine = lines.find(l => l.startsWith("CUDA:OK") || l.startsWith("CUDA:FAIL")) ?? "";
  const stderrFull = r.stderr ?? "";
  const isModuleErr = stderrFull.includes("ModuleNotFoundError") || stderrFull.includes("No module named");
  const stderrDisplay = stderrFull.substring(0, 200);

  if (cudaLine.startsWith("CUDA:FAIL")) {
    return { result: { label: "CUDA", status: "❌", detail: cudaLine }, isFail: true, isWarn: false };
  } else if (r.status !== 0 || !cudaLine) {
    const detail = isModuleErr
      ? `torch not installed: ${stderrDisplay}`
      : `check failed (exit ${r.status}): ${stderrDisplay || cudaLine}`;
    return { result: { label: "CUDA", status: "❌", detail }, isFail: true, isWarn: false };
  } else {
    return { result: { label: "CUDA", status: "✅", detail: cudaLine.replace("CUDA:OK ", "") }, isFail: false, isWarn: false };
  }
}

describe("run_preflight: CUDA availability check", () => {
  // Case 1: GPU visible — normal happy path
  it("CUDA:OK stdout → ✅, isFail=false", () => {
    const r = processCudaResult({ status: 0, stdout: "CUDA:OK cuda_build=12.4 driver=550.54 torch=2.3.0", stderr: "" });
    expect(r.result.status).toBe("✅");
    expect(r.result.detail).toBe("cuda_build=12.4 driver=550.54 torch=2.3.0");
    expect(r.isFail).toBe(false);
    expect(r.isWarn).toBe(false);
  });

  // Case 2: pip install overwrote CUDA torch (the actual incident trigger)
  it("CUDA:FAIL stdout → ❌, isFail=true", () => {
    const r = processCudaResult({ status: 0, stdout: "CUDA:FAIL torch=2.11.0+cu130 cuda_build=13.0", stderr: "" });
    expect(r.result.status).toBe("❌");
    expect(r.result.detail).toContain("CUDA:FAIL");
    expect(r.isFail).toBe(true);
  });

  // Case 3: torch not installed at all (pip install not yet run, or failed)
  it("status=1 + ModuleNotFoundError in stderr → ❌ 'torch not installed'", () => {
    const r = processCudaResult({ status: 1, stdout: "", stderr: "ModuleNotFoundError: No module named 'torch'" });
    expect(r.result.status).toBe("❌");
    expect(r.result.detail).toContain("torch not installed");
    expect(r.isFail).toBe(true);
  });

  // Case 4: SSH timeout / cold pod — non-blocking warning
  it("status=null (timeout) → ⚠️, isWarn=true, isFail=false", () => {
    const r = processCudaResult({ status: null, stdout: "", stderr: "" });
    expect(r.result.status).toBe("⚠️");
    expect(r.result.detail).toContain("timed out");
    expect(r.isFail).toBe(false);
    expect(r.isWarn).toBe(true);
  });

  // Case 5: nvidia-smi absent but CUDA available (Python handles FileNotFoundError → drv='nvidia-smi-missing')
  it("nvidia-smi missing but CUDA:OK → ✅, detail includes 'nvidia-smi-missing'", () => {
    const r = processCudaResult({ status: 0, stdout: "CUDA:OK cuda_build=12.4 driver=nvidia-smi-missing torch=2.3.0", stderr: "" });
    expect(r.result.status).toBe("✅");
    expect(r.result.detail).toContain("nvidia-smi-missing");
    expect(r.isFail).toBe(false);
  });

  // SSH banner / conda init noise before CUDA line — line parser must skip it
  it("SSH banner before CUDA:OK → ✅ (line parser skips noise)", () => {
    const stdout = "Welcome to RunPod!\nLast login: Mon Apr 22\nCUDA:OK cuda_build=12.4 driver=550.54 torch=2.3.0";
    const r = processCudaResult({ status: 0, stdout, stderr: "" });
    expect(r.result.status).toBe("✅");
  });
});
