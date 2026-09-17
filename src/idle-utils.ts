/** Pure utility functions for idle/work detection on RunPod pods — extracted for testability. */

import { mkdir, readFile, writeFile, rename } from "node:fs/promises";
import { dirname } from "node:path";
import { parseNvidiaSmiOutput } from "./gpu-utils.js";

export interface WorkSample {
  gpuUtil: number | null;
  usedMb: number | null;
  totalMb: number | null;
  computeProcs: number;
}

export interface IdleRecord {
  lastNonIdleAt: string; // ISO string
  lastSampleAt: string; // ISO string
}

export type IdleState = Record<string, IdleRecord>;

export interface IdleVerdict {
  working: boolean;
  idleMinutes: number;
  sustained: boolean;
  firstObservation: boolean;
  record: IdleRecord;
}

const VRAM_FLOOR_MB_DEFAULT = 100;

/** True if the sample shows the pod is doing GPU/compute work. */
export function isWorking(s: WorkSample, vramFloorMb = VRAM_FLOOR_MB_DEFAULT): boolean {
  if (s.gpuUtil != null && s.gpuUtil > 0) return true;
  if (s.usedMb != null && s.usedMb > vramFloorMb) return true;
  if (s.computeProcs > 0) return true;
  return false;
}

/** Judge idle/working state for a pod, given the previous record (if any). */
export function judgeIdle(
  podId: string,
  s: WorkSample,
  prev: IdleRecord | undefined,
  now: Date,
  thresholdMinutes: number
): IdleVerdict {
  const nowIso = now.toISOString();
  const working = isWorking(s);

  if (working) {
    return {
      working: true,
      idleMinutes: 0,
      sustained: false,
      firstObservation: !prev,
      record: { lastNonIdleAt: nowIso, lastSampleAt: nowIso },
    };
  }

  if (!prev) {
    return {
      working: false,
      idleMinutes: 0,
      sustained: false,
      firstObservation: true,
      record: { lastNonIdleAt: nowIso, lastSampleAt: nowIso },
    };
  }

  const idleMs = now.getTime() - new Date(prev.lastNonIdleAt).getTime();
  const idleMinutes = Math.round(idleMs / 60000);
  const sustained = idleMinutes >= thresholdMinutes;

  return {
    working: false,
    idleMinutes,
    sustained,
    firstObservation: false,
    record: { lastNonIdleAt: prev.lastNonIdleAt, lastSampleAt: nowIso },
  };
}

/** Format minutes as a human-friendly duration: "0m", "47m", "14h32m", "1d1h0m". */
export function formatDuration(minutes: number): string {
  if (minutes < 60) return `${minutes}m`;
  const totalHours = Math.floor(minutes / 60);
  const remMinutes = minutes % 60;
  if (totalHours < 24) return `${totalHours}h${remMinutes}m`;
  const days = Math.floor(totalHours / 24);
  const remHours = totalHours % 24;
  return `${days}d${remHours}h${remMinutes}m`;
}

/** Estimated cost (USD) accrued while idle, rounded to 2 decimals. */
export function idleCostUsd(idleMinutes: number, costPerHr: number): number {
  const usd = (idleMinutes / 60) * costPerHr;
  return Math.round(usd * 100) / 100;
}

/** Drop state entries for pods that are no longer in the live pod list. */
export function pruneState(state: IdleState, livePodIds: string[]): IdleState {
  const live = new Set(livePodIds);
  const pruned: IdleState = {};
  for (const [podId, record] of Object.entries(state)) {
    if (live.has(podId)) pruned[podId] = record;
  }
  return pruned;
}

/**
 * Parse the combined work-probe output:
 * nvidia-smi CSV (or NO_NVIDIA_SMI marker on CPU pods), then "---PROCS---", then a process count.
 */
export function parseWorkProbe(stdout: string): WorkSample {
  const marker = "---PROCS---";
  const idx = stdout.indexOf(marker);
  const smiPart = idx >= 0 ? stdout.slice(0, idx) : stdout;
  const procsPart = idx >= 0 ? stdout.slice(idx + marker.length) : "";

  let gpuUtil: number | null = null;
  let usedMb: number | null = null;
  let totalMb: number | null = null;

  if (!smiPart.includes("NO_NVIDIA_SMI")) {
    const gpus = parseNvidiaSmiOutput(smiPart);
    if (gpus.length) {
      gpuUtil = gpus[0].gpuUtil;
      usedMb = gpus[0].usedMb;
      totalMb = gpus[0].totalMb;
    }
  }

  const procMatch = procsPart.match(/-?\d+/);
  let computeProcs = procMatch ? parseInt(procMatch[0], 10) : 0;
  if (isNaN(computeProcs) || computeProcs < 0) computeProcs = 0;

  return { gpuUtil, usedMb, totalMb, computeProcs };
}

export const IDLE_STATE_PATH = ".omc/gpu-exec/idle-state.json";

export async function loadIdleState(path: string = IDLE_STATE_PATH): Promise<IdleState> {
  try {
    const raw = await readFile(path, "utf8");
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") return parsed as IdleState;
    return {};
  } catch {
    return {};
  }
}

export async function saveIdleState(state: IdleState, path: string = IDLE_STATE_PATH): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tmpPath = `${path}.tmp-${process.pid}-${Date.now()}`;
  await writeFile(tmpPath, JSON.stringify(state, null, 2), "utf8");
  await rename(tmpPath, path);
}

/** The single-line SSH probe: nvidia-smi CSV (or NO_NVIDIA_SMI marker) + compute process count. */
export const WORK_PROBE_CMD =
  "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo NO_NVIDIA_SMI; " +
  "echo \"---PROCS---\"; " +
  "(pgrep -c -f 'python|torchrun|accelerate' 2>/dev/null || echo 0)";

/** Render the "Work: ..." line for a pod, given its probe verdict (or null if the probe failed/was skipped). */
export function renderWorkLine(
  verdict: IdleVerdict | null,
  sample: WorkSample | null,
  costPerHr: number | undefined,
  thresholdMinutes: number,
  probeError?: string
): string {
  if (probeError) return `Work: signal unavailable (${probeError})`;
  if (!verdict || !sample) return "Work: signal unavailable";

  const gpuPart =
    sample.gpuUtil == null || sample.usedMb == null || sample.totalMb == null
      ? "no nvidia-smi"
      : `GPU ${sample.gpuUtil}% · ${sample.usedMb}/${sample.totalMb} MiB · procs ${sample.computeProcs}`;

  if (verdict.working) {
    return `Work: ACTIVE · ${gpuPart}`;
  }

  if (verdict.firstObservation) {
    return `Work: quiet (first observation — clock started) · ${gpuPart}`;
  }

  if (verdict.sustained) {
    const costSuffix =
      costPerHr != null ? ` · idle cost so far ~$${idleCostUsd(verdict.idleMinutes, costPerHr).toFixed(2)}` : "";
    return `Work: ⚠️ IDLE ${formatDuration(verdict.idleMinutes)} · ${gpuPart}${costSuffix}`;
  }

  return `Work: quiet ${formatDuration(verdict.idleMinutes)} (below ${thresholdMinutes}m threshold) · ${gpuPart}`;
}
