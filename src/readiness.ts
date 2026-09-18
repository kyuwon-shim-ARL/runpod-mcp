/**
 * Import-readiness gates for pods.
 *
 * Incident (2026-09-18): two pods were created before anyone checked that the first one
 * could import the project's modules. Hours later both hit `ModuleNotFoundError`. A markdown
 * rule ("verify imports on the first pod, then create the rest") did not stop it.
 *
 * The gate makes the rule refusable: when `create_pod_auto` is called with `imports`, the pod
 * is recorded as NOT READY, and the next `create_pod_auto` is refused until `run_preflight`
 * verifies those imports. The pod is still created and its id still returned — the gate never
 * risks an orphaned, unreported, billing pod.
 */

import { mkdir, readFile, writeFile, rename } from "node:fs/promises";
import { dirname } from "node:path";

export interface ReadinessGate {
  imports: string[];
  createdAt: string; // ISO
  verifiedAt: string | null; // ISO once run_preflight passed every import
}

export type ReadinessState = Record<string, ReadinessGate>;

export interface PendingGate extends ReadinessGate {
  podId: string;
}

export const READINESS_STATE_PATH = ".omc/gpu-exec/readiness.json";

/** Gates that have not been satisfied yet. */
export function pendingGates(state: ReadinessState): PendingGate[] {
  return Object.entries(state)
    .filter(([, gate]) => gate.verifiedAt == null)
    .map(([podId, gate]) => ({ podId, ...gate }));
}

/** Record a new unverified gate for a pod. */
export function recordGate(
  state: ReadinessState,
  podId: string,
  imports: string[],
  now: Date
): ReadinessState {
  return {
    ...state,
    [podId]: { imports: [...imports], createdAt: now.toISOString(), verifiedAt: null },
  };
}

/** Stamp a pod's gate as satisfied. No-op when the pod has no gate. */
export function markVerified(state: ReadinessState, podId: string, now: Date): ReadinessState {
  const gate = state[podId];
  if (!gate) return state;
  return { ...state, [podId]: { ...gate, verifiedAt: now.toISOString() } };
}

/** Drop gates for pods that are no longer in the live pod list. */
export function pruneGates(state: ReadinessState, livePodIds: string[]): ReadinessState {
  const live = new Set(livePodIds);
  const pruned: ReadinessState = {};
  for (const [podId, gate] of Object.entries(state)) {
    if (live.has(podId)) pruned[podId] = gate;
  }
  return pruned;
}

function preflightCall(podId: string, imports: string[]): string {
  const list = imports.map((i) => `"${i}"`).join(", ");
  return `run_preflight(podId="${podId}", importSmokes=[${list}])`;
}

/** The NOT-READY block appended to a create_pod_auto success when `imports` was given. */
export function renderGateBlock(podId: string, imports: string[]): string {
  return [
    ``,
    `⚠️ NOT READY — import 미검증. 팟은 생성됐지만 프로젝트 코드를 돌릴 수 있는지 아직 모른다.`,
    `다음 호출로 검증하기 전까지 이 팟에서 훈련을 시작하지 말 것:`,
    `  ${preflightCall(podId, imports)}`,
    `검증 전에는 다음 create_pod_auto 호출이 거부된다 (형제 팟 동시 생성 사고 방지).`,
  ].join("\n");
}

/** The refusal returned when a new pod is requested while gates are still pending. */
export function renderPendingRefusal(pending: PendingGate[]): string {
  const lines = [
    `❌ 팟 생성 거부 — import 미검증 팟이 ${pending.length}개 있다.`,
    ``,
    `첫 팟에서 import를 검증하기 전에 형제 팟을 만들면, 누락 모듈을 몇 시간 뒤에 모든 팟에서 동시에 발견하게 된다 (2026-09-18 사고).`,
    ``,
  ];
  for (const g of pending) {
    lines.push(`- ${g.podId} (생성 ${g.createdAt}) — 미검증 imports: ${g.imports.join(", ")}`);
    lines.push(`  → ${preflightCall(g.podId, g.imports)}`);
  }
  lines.push(``);
  lines.push(`검증이 통과하면 게이트는 자동으로 해제된다.`);
  lines.push(`의도적으로 건너뛰려면 skipReadinessGate: true 를 전달한다 (사용자가 직접 요청한 경우에만).`);
  return lines.join("\n");
}

export async function loadReadinessState(
  path: string = READINESS_STATE_PATH
): Promise<ReadinessState> {
  try {
    const raw = await readFile(path, "utf8");
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as ReadinessState;
    }
    return {};
  } catch {
    return {};
  }
}

export async function saveReadinessState(
  state: ReadinessState,
  path: string = READINESS_STATE_PATH
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tmpPath = `${path}.tmp-${process.pid}-${Date.now()}`;
  await writeFile(tmpPath, JSON.stringify(state, null, 2), "utf8");
  await rename(tmpPath, path);
}

/**
 * Label for an import smoke statement. `"import torch"` → `torch`, `"from x import y"` → `x`,
 * and a bare `"torch"` → `torch` (a bare module name is a valid expression statement, and the
 * old `split(" ")[1]` rendered it as `undefined`).
 */
export function importStatementLabel(statement: string): string {
  const words = statement.trim().split(/\s+/);
  if (words.length === 1) return words[0];
  if (words[0] === "import" || words[0] === "from") return words[1];
  return statement.trim();
}

/**
 * Which of a pod's gated imports are still unverified, given the set that just passed.
 * A gate must only open for the imports it actually demanded — verifying `["os"]` must not
 * release a gate recorded for `["torch","kornia"]`, or the gate would certify a check that
 * never happened, which is the incident it exists to prevent.
 */
export function unmetImports(gate: ReadinessGate, passed: string[]): string[] {
  const passedLabels = new Set(passed.map(importStatementLabel));
  return gate.imports.filter((required) => !passedLabels.has(importStatementLabel(required)));
}
