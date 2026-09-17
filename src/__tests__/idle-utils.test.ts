import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  isWorking,
  judgeIdle,
  formatDuration,
  idleCostUsd,
  pruneState,
  parseWorkProbe,
  loadIdleState,
  saveIdleState,
  type IdleRecord,
} from "../idle-utils.js";

describe("isWorking", () => {
  it("gpuUtil > 0 → true", () => {
    expect(isWorking({ gpuUtil: 5, usedMb: 0, totalMb: 24576, computeProcs: 0 })).toBe(true);
  });
  it("usedMb > 100 with 0% util → true", () => {
    expect(isWorking({ gpuUtil: 0, usedMb: 6592, totalMb: 24576, computeProcs: 0 })).toBe(true);
  });
  it("computeProcs > 0 with 0% util & 1MiB → true (CPU-phase case)", () => {
    expect(isWorking({ gpuUtil: 0, usedMb: 1, totalMb: 24576, computeProcs: 1 })).toBe(true);
  });
  it("all zero → false", () => {
    expect(isWorking({ gpuUtil: 0, usedMb: 0, totalMb: 24576, computeProcs: 0 })).toBe(false);
  });
  it("null gpu (CPU pod) + procs 0 → false", () => {
    expect(isWorking({ gpuUtil: null, usedMb: null, totalMb: null, computeProcs: 0 })).toBe(false);
  });
  it("null gpu + procs 2 → true", () => {
    expect(isWorking({ gpuUtil: null, usedMb: null, totalMb: null, computeProcs: 2 })).toBe(true);
  });
});

describe("judgeIdle", () => {
  const now = new Date("2026-09-18T12:00:00.000Z");

  it("working sample resets lastNonIdleAt to now", () => {
    const prev: IdleRecord = { lastNonIdleAt: "2026-09-18T10:00:00.000Z", lastSampleAt: "2026-09-18T11:00:00.000Z" };
    const v = judgeIdle("p1", { gpuUtil: 50, usedMb: 1000, totalMb: 24576, computeProcs: 1 }, prev, now, 10);
    expect(v.working).toBe(true);
    expect(v.idleMinutes).toBe(0);
    expect(v.sustained).toBe(false);
    expect(v.record.lastNonIdleAt).toBe(now.toISOString());
  });

  it("first idle observation → firstObservation true, sustained false, idleMinutes 0", () => {
    const v = judgeIdle("p1", { gpuUtil: 0, usedMb: 0, totalMb: 24576, computeProcs: 0 }, undefined, now, 10);
    expect(v.firstObservation).toBe(true);
    expect(v.sustained).toBe(false);
    expect(v.idleMinutes).toBe(0);
    expect(v.record.lastNonIdleAt).toBe(now.toISOString());
  });

  it("prev 30min ago + threshold 10 → idleMinutes 30, sustained true, lastNonIdleAt unchanged", () => {
    const prev: IdleRecord = { lastNonIdleAt: "2026-09-18T11:30:00.000Z", lastSampleAt: "2026-09-18T11:55:00.000Z" };
    const v = judgeIdle("p1", { gpuUtil: 0, usedMb: 0, totalMb: 24576, computeProcs: 0 }, prev, now, 10);
    expect(v.idleMinutes).toBe(30);
    expect(v.sustained).toBe(true);
    expect(v.record.lastNonIdleAt).toBe(prev.lastNonIdleAt);
    expect(v.firstObservation).toBe(false);
  });

  it("prev 5min ago + threshold 10 → sustained false", () => {
    const prev: IdleRecord = { lastNonIdleAt: "2026-09-18T11:55:00.000Z", lastSampleAt: "2026-09-18T11:55:00.000Z" };
    const v = judgeIdle("p1", { gpuUtil: 0, usedMb: 0, totalMb: 24576, computeProcs: 0 }, prev, now, 10);
    expect(v.idleMinutes).toBe(5);
    expect(v.sustained).toBe(false);
  });

  it("a single 0% sample must NOT be sustained", () => {
    const v = judgeIdle("p1", { gpuUtil: 0, usedMb: 0, totalMb: 24576, computeProcs: 0 }, undefined, now, 10);
    expect(v.sustained).toBe(false);
  });
});

describe("formatDuration", () => {
  it("0 → 0m", () => expect(formatDuration(0)).toBe("0m"));
  it("47 → 47m", () => expect(formatDuration(47)).toBe("47m"));
  it("872 → 14h32m", () => expect(formatDuration(872)).toBe("14h32m"));
  it("1500 → 1d1h0m", () => expect(formatDuration(1500)).toBe("1d1h0m"));
});

describe("idleCostUsd", () => {
  it("872 minutes at $0.22/hr → 3.20", () => {
    expect(idleCostUsd(872, 0.22)).toBe(3.2);
  });
});

describe("pruneState", () => {
  it("drops dead pods, keeps live", () => {
    const state = {
      alive: { lastNonIdleAt: "x", lastSampleAt: "x" },
      dead: { lastNonIdleAt: "y", lastSampleAt: "y" },
    };
    const pruned = pruneState(state, ["alive"]);
    expect(pruned).toEqual({ alive: state.alive });
  });
});

describe("parseWorkProbe", () => {
  it("normal GPU output", () => {
    const out =
      "0, NVIDIA A100, 24576, 6592, 17984, 97, 40, 65\n" +
      "---PROCS---\n" +
      "5\n";
    const s = parseWorkProbe(out);
    expect(s.gpuUtil).toBe(97);
    expect(s.usedMb).toBe(6592);
    expect(s.totalMb).toBe(24576);
    expect(s.computeProcs).toBe(5);
  });

  it("nvidia-smi absent output (CPU pod)", () => {
    const out = "NO_NVIDIA_SMI\n---PROCS---\n0\n";
    const s = parseWorkProbe(out);
    expect(s.gpuUtil).toBeNull();
    expect(s.usedMb).toBeNull();
    expect(s.totalMb).toBeNull();
    expect(s.computeProcs).toBe(0);
  });

  it("procs count line with no match found (pgrep exit 1, empty)", () => {
    const out = "NO_NVIDIA_SMI\n---PROCS---\n";
    const s = parseWorkProbe(out);
    expect(s.computeProcs).toBe(0);
  });
});

describe("load/save idle state", () => {
  let dir: string;
  let path: string;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "idle-state-"));
    path = join(dir, "nested", "idle-state.json");
  });

  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it("round-trips state", async () => {
    const state = { p1: { lastNonIdleAt: "2026-09-18T00:00:00.000Z", lastSampleAt: "2026-09-18T00:05:00.000Z" } };
    await saveIdleState(state, path);
    const loaded = await loadIdleState(path);
    expect(loaded).toEqual(state);
  });

  it("missing file → {}", async () => {
    const loaded = await loadIdleState(path);
    expect(loaded).toEqual({});
  });

  it("corrupt JSON → {}", async () => {
    const { mkdir, writeFile } = await import("node:fs/promises");
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, "{not json", "utf8");
    const loaded = await loadIdleState(path);
    expect(loaded).toEqual({});
  });
});

function dirname(p: string): string {
  return p.split("/").slice(0, -1).join("/");
}
