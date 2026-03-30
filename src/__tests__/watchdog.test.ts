import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { loadState, saveState, checkPod, runWatchdog } from "../watchdog.js";
import type { WatchdogOptions, Alerter, PodIdleState } from "../watchdog.js";
import { writeFileSync, unlinkSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { tmpdir } from "node:os";

// ── State persistence ──

describe("watchdog state management", () => {
  const testFile = resolve(tmpdir(), `.runpod-watchdog-test-${Date.now()}.json`);

  afterEach(() => {
    try { unlinkSync(testFile); } catch {}
  });

  it("loadState returns empty object for missing file", () => {
    const state = loadState("/tmp/nonexistent-watchdog-state-xyz.json");
    expect(state).toEqual({});
  });

  it("loadState returns empty object for invalid JSON", () => {
    writeFileSync(testFile, "not json{{{");
    const state = loadState(testFile);
    expect(state).toEqual({});
  });

  it("saveState + loadState roundtrip", () => {
    const state: PodIdleState = {
      "pod-1": { consecutiveIdle: 3, lastChecked: "2025-01-01T00:00:00Z", lastStatus: "IDLE" },
      "pod-2": { consecutiveIdle: 0, lastChecked: "2025-01-01T00:00:00Z", lastStatus: "OPTIMAL" },
    };
    saveState(testFile, state);
    const loaded = loadState(testFile);
    expect(loaded).toEqual(state);
  });

  it("saveState overwrites previous state", () => {
    saveState(testFile, { "pod-1": { consecutiveIdle: 1, lastChecked: "", lastStatus: "" } });
    saveState(testFile, { "pod-2": { consecutiveIdle: 5, lastChecked: "", lastStatus: "" } });
    const loaded = loadState(testFile);
    expect(loaded).not.toHaveProperty("pod-1");
    expect(loaded["pod-2"].consecutiveIdle).toBe(5);
  });
});

// ── runWatchdog logic ──

describe("runWatchdog", () => {
  const stateFile = resolve(tmpdir(), `.runpod-watchdog-test-run-${Date.now()}.json`);

  afterEach(() => {
    try { unlinkSync(stateFile); } catch {}
  });

  function makeOptions(overrides: Partial<WatchdogOptions> = {}): WatchdogOptions {
    return {
      autoStop: false,
      skipPattern: /keep|persist/i,
      idleThreshold: 3,
      stateFile,
      ...overrides,
    };
  }

  function makeAlerter(): Alerter & { warns: string[]; actions: string[] } {
    const warns: string[] = [];
    const actions: string[] = [];
    return {
      warns,
      actions,
      warn(podName, podId, idleCount, message) { warns.push(`${podName}:${idleCount}:${message}`); },
      action(podName, podId, action) { actions.push(`${podName}:${action}`); },
    };
  }

  it("skips pods matching skipPattern", async () => {
    const client = {
      listPods: vi.fn().mockResolvedValue([
        { id: "p1", name: "keep-this", desiredStatus: "RUNNING" },
        { id: "p2", name: "persist-data", desiredStatus: "RUNNING" },
      ]),
    } as any;

    const alerter = makeAlerter();
    const result = await runWatchdog(client, makeOptions(), alerter);
    expect(result.checked).toBe(0); // Both skipped
    expect(alerter.warns).toHaveLength(0);
  });

  it("only processes RUNNING pods", async () => {
    const client = {
      listPods: vi.fn().mockResolvedValue([
        { id: "p1", name: "stopped-pod", desiredStatus: "EXITED" },
        { id: "p2", name: "errored-pod", desiredStatus: "ERROR" },
      ]),
    } as any;

    const result = await runWatchdog(client, makeOptions());
    expect(result.checked).toBe(0);
  });

  it("cleans up state for pods no longer running", async () => {
    // Pre-populate state with a pod that no longer exists
    saveState(stateFile, {
      "old-pod": { consecutiveIdle: 5, lastChecked: "2025-01-01T00:00:00Z", lastStatus: "IDLE" },
    });

    const client = {
      listPods: vi.fn().mockResolvedValue([]),
    } as any;

    await runWatchdog(client, makeOptions());
    const state = loadState(stateFile);
    expect(state).not.toHaveProperty("old-pod");
  });
});
