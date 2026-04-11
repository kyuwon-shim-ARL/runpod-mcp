import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { loadState, saveState, checkPod, runWatchdog } from "../watchdog.js";
import { spawnAsync } from "../api.js";
import { writeFileSync, unlinkSync } from "node:fs";
import { resolve } from "node:path";
import { tmpdir } from "node:os";
vi.mock("../api.js", async () => {
    const actual = await vi.importActual("../api.js");
    return {
        ...actual,
        spawnAsync: vi.fn(),
    };
});
const mockSpawnAsync = vi.mocked(spawnAsync);
// ── State persistence ──
describe("watchdog state management", () => {
    const testFile = resolve(tmpdir(), `.runpod-watchdog-test-${Date.now()}.json`);
    afterEach(() => {
        try {
            unlinkSync(testFile);
        }
        catch { }
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
        const state = {
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
        try {
            unlinkSync(stateFile);
        }
        catch { }
    });
    function makeOptions(overrides = {}) {
        return {
            autoStop: false,
            skipPattern: /keep|persist/i,
            idleThreshold: 3,
            stateFile,
            ...overrides,
        };
    }
    function makeAlerter() {
        const warns = [];
        const actions = [];
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
        };
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
        };
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
        };
        await runWatchdog(client, makeOptions());
        const state = loadState(stateFile);
        expect(state).not.toHaveProperty("old-pod");
    });
    it("accumulates idle count and triggers warn alerter", async () => {
        // 8-col nvidia-smi output: index, name, total, used, free, gpuUtil, memUtil, temp
        // Low usage = IDLE (< 30% VRAM)
        const idleOutput = "0, RTX 3090, 24576, 2000, 22576, 5, 3, 45";
        mockSpawnAsync.mockResolvedValue({ stdout: idleOutput, stderr: "", status: 0 });
        const client = {
            listPods: vi.fn().mockResolvedValue([
                { id: "p1", name: "training-run", desiredStatus: "RUNNING", costPerHr: 0.44 },
            ]),
            getPod: vi.fn().mockResolvedValue({
                id: "p1", name: "training-run", desiredStatus: "RUNNING",
                publicIp: "1.2.3.4", portMappings: { "22": 10022 },
            }),
            config: { sshKeyPath: "/tmp/key" },
        };
        const alerter = makeAlerter();
        // First run
        await runWatchdog(client, makeOptions(), alerter);
        expect(alerter.warns).toHaveLength(1);
        expect(alerter.warns[0]).toContain("training-run");
        expect(alerter.warns[0]).toContain(":1:"); // idle count = 1
        // Second run — idle accumulates
        await runWatchdog(client, makeOptions(), alerter);
        expect(alerter.warns).toHaveLength(2);
        expect(alerter.warns[1]).toContain(":2:"); // idle count = 2
    });
    it("auto-stops pod after reaching idle threshold", async () => {
        const idleOutput = "0, RTX 3090, 24576, 1000, 23576, 2, 1, 40";
        mockSpawnAsync.mockResolvedValue({ stdout: idleOutput, stderr: "", status: 0 });
        const stopPod = vi.fn().mockResolvedValue({});
        const client = {
            listPods: vi.fn().mockResolvedValue([
                { id: "p1", name: "idle-pod", desiredStatus: "RUNNING", costPerHr: 0.44 },
            ]),
            getPod: vi.fn().mockResolvedValue({
                id: "p1", name: "idle-pod", desiredStatus: "RUNNING",
                publicIp: "1.2.3.4", portMappings: { "22": 10022 },
            }),
            stopPod,
            config: { sshKeyPath: "/tmp/key" },
        };
        const alerter = makeAlerter();
        const opts = makeOptions({ autoStop: true, idleThreshold: 2 });
        // Run 1: idle count = 1, no stop yet
        await runWatchdog(client, opts, alerter);
        expect(stopPod).not.toHaveBeenCalled();
        // Run 2: idle count = 2 >= threshold, auto-stop triggered
        const result = await runWatchdog(client, opts, alerter);
        expect(stopPod).toHaveBeenCalledWith("p1");
        expect(result.stopped).toBe(1);
        expect(alerter.actions).toHaveLength(1);
        expect(alerter.actions[0]).toContain("Auto-stopped");
    });
});
// ── checkPod ──
describe("checkPod", () => {
    beforeEach(() => {
        mockSpawnAsync.mockReset();
    });
    it("returns NO_SSH when pod has no public IP", async () => {
        const client = {
            getPod: vi.fn().mockResolvedValue({
                id: "p1", publicIp: null, portMappings: {},
            }),
        };
        const result = await checkPod(client, "p1", "test-pod", undefined);
        expect(result.status).toBe("NO_SSH");
        expect(result.vramPercent).toBeNull();
    });
    it("returns SSH_FAILED when SSH command fails", async () => {
        const client = {
            getPod: vi.fn().mockResolvedValue({
                id: "p1", publicIp: "1.2.3.4", portMappings: { "22": 10022 },
            }),
        };
        mockSpawnAsync.mockResolvedValue({ stdout: "", stderr: "Connection refused", status: 255 });
        const result = await checkPod(client, "p1", "test-pod", "/tmp/key");
        expect(result.status).toBe("SSH_FAILED");
    });
    it("returns UNKNOWN when nvidia-smi output is empty", async () => {
        const client = {
            getPod: vi.fn().mockResolvedValue({
                id: "p1", publicIp: "1.2.3.4", portMappings: { "22": 10022 },
            }),
        };
        mockSpawnAsync.mockResolvedValue({ stdout: "", stderr: "", status: 0 });
        const result = await checkPod(client, "p1", "test-pod", undefined);
        expect(result.status).toBe("UNKNOWN");
    });
    it("returns IDLE label for low VRAM usage", async () => {
        const client = {
            getPod: vi.fn().mockResolvedValue({
                id: "p1", publicIp: "1.2.3.4", portMappings: { "22": 10022 },
            }),
        };
        // 8-col format: index, name, total, used, free, gpuUtil, memUtil, temp
        // 2000 / 24576 = ~8% → IDLE
        mockSpawnAsync.mockResolvedValue({
            stdout: "0, RTX 3090, 24576, 2000, 22576, 5, 3, 45",
            stderr: "", status: 0,
        });
        const result = await checkPod(client, "p1", "test-pod", "/tmp/key");
        expect(result.status).toBe("IDLE");
        expect(result.vramPercent).toBe(8);
    });
});
//# sourceMappingURL=watchdog.test.js.map