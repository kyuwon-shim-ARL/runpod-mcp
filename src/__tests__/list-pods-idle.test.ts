import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Pod } from "../types.js";

const spawnAsync = vi.fn();
const listPods = vi.fn();

vi.mock("../api.js", async () => {
  const actual = await vi.importActual<typeof import("../api.js")>("../api.js");
  class FakeRunPodClient {
    listPods = listPods;
    getPod = vi.fn();
    getSshArgs(pod: Pod) {
      if (!pod.publicIp || !pod.portMappings?.["22"]) return null;
      return ["ssh", "-p", String(pod.portMappings["22"]), `root@${pod.publicIp}`];
    }
    getSshCommandString(pod: Pod) {
      const args = this.getSshArgs(pod);
      return args ? args.join(" ") : null;
    }
  }
  return { ...actual, spawnAsync, RunPodClient: FakeRunPodClient };
});

// loadIdleState/saveIdleState default to the real implementation; individual tests
// can override them to reject, to verify list_pods stays fail-soft on state I/O errors.
const loadIdleState = vi.fn();
const saveIdleState = vi.fn();

vi.mock("../idle-utils.js", async () => {
  const actual = await vi.importActual<typeof import("../idle-utils.js")>("../idle-utils.js");
  loadIdleState.mockImplementation(actual.loadIdleState);
  saveIdleState.mockImplementation(actual.saveIdleState);
  return { ...actual, loadIdleState, saveIdleState };
});

process.env.RUNPOD_API_KEY = "rp_test_key";

const { server } = await import("../index.js");

const ACTIVE_POD: Pod = {
  id: "pod-active",
  name: "active-pod",
  desiredStatus: "RUNNING",
  publicIp: "1.2.3.4",
  portMappings: { "22": 40001 },
  costPerHr: 0.44,
};

const IDLE_POD: Pod = {
  id: "pod-idle",
  name: "idle-pod",
  desiredStatus: "RUNNING",
  publicIp: "1.2.3.5",
  portMappings: { "22": 40002 },
  costPerHr: 0.22,
};

const ok = (stdout: string) => ({ status: 0, stdout, stderr: "", error: undefined });
const fail = (message: string) => ({ status: null, stdout: "", stderr: "", error: new Error(message) });

const activeOutput = "0, NVIDIA A100, 24576, 6592, 17984, 97, 40, 65\n---PROCS---\n5\n";
const idleOutput = "0, NVIDIA A100, 24576, 1, 24575, 0, 0, 40\n---PROCS---\n0\n";

async function callListPods(args: Record<string, unknown> = {}) {
  const tool = (server as any)._registeredTools["list_pods"];
  const withDefaults = {
    probe: true,
    idleThresholdMinutes: 10,
    probeTimeoutSeconds: 15,
    ...args,
  };
  return tool.handler(withDefaults, {});
}

describe("list_pods idle signal", () => {
  let dir: string;
  const originalCwd = process.cwd();

  beforeEach(async () => {
    spawnAsync.mockReset();
    listPods.mockReset();
    const { loadIdleState: actualLoad, saveIdleState: actualSave } =
      await vi.importActual<typeof import("../idle-utils.js")>("../idle-utils.js");
    loadIdleState.mockReset().mockImplementation(actualLoad);
    saveIdleState.mockReset().mockImplementation(actualSave);
    dir = await mkdtemp(join(tmpdir(), "list-pods-idle-"));
    process.chdir(dir);
  });

  afterEach(async () => {
    process.chdir(originalCwd);
    await rm(dir, { recursive: true, force: true });
  });

  it("shows the job_group line and live siblings from .omc/pods records", async () => {
    const { mkdir, writeFile } = await import("node:fs/promises");
    const { toYaml } = await import("../pod-ops.js");
    await mkdir(".omc/pods", { recursive: true });
    await writeFile(
      ".omc/pods/a.yaml",
      toYaml({ pod_id: "pod-active", name: "a", job_group: "lopo-s123" }),
      "utf8"
    );
    await writeFile(
      ".omc/pods/b.yaml",
      toYaml({ pod_id: "pod-idle", name: "b", job_group: "lopo-s123" }),
      "utf8"
    );
    listPods.mockResolvedValueOnce([ACTIVE_POD, IDLE_POD]);
    spawnAsync.mockResolvedValue(ok(activeOutput));

    const result = await callListPods();
    const output = result.content[0].text as string;
    expect(output).toContain("Group: lopo-s123 (siblings live: pod-idle)");
    expect(output).toContain("Group: lopo-s123 (siblings live: pod-active)");
  });

  it("omits the group line when there are no pod records", async () => {
    listPods.mockResolvedValueOnce([ACTIVE_POD]);
    spawnAsync.mockResolvedValueOnce(ok(activeOutput));

    const result = await callListPods();
    expect(result.content[0].text as string).not.toContain("Group:");
  });

  it("shows the group line even with probe disabled", async () => {
    const { mkdir, writeFile } = await import("node:fs/promises");
    const { toYaml } = await import("../pod-ops.js");
    await mkdir(".omc/pods", { recursive: true });
    await writeFile(".omc/pods/a.yaml", toYaml({ pod_id: "pod-active", name: "a", job_group: "g1" }), "utf8");
    listPods.mockResolvedValueOnce([ACTIVE_POD]);

    const result = await callListPods({ probe: false });
    const output = result.content[0].text as string;
    expect(output).toContain("Group: g1 (no live siblings)");
    expect(spawnAsync).not.toHaveBeenCalled();
  });

  it("shows ACTIVE line for a working pod", async () => {
    listPods.mockResolvedValueOnce([ACTIVE_POD]);
    spawnAsync.mockResolvedValueOnce(ok(activeOutput));

    const result = await callListPods();
    const output = result.content[0].text as string;
    expect(output).toContain("Work: ACTIVE");
    expect(output).toContain("GPU 97%");
  });

  it("shows sustained IDLE line + trailing summary given a pre-seeded 30min-old state", async () => {
    const { saveIdleState } = await import("../idle-utils.js");
    const thirtyMinAgo = new Date(Date.now() - 30 * 60_000).toISOString();
    await saveIdleState({ [IDLE_POD.id]: { lastNonIdleAt: thirtyMinAgo, lastSampleAt: thirtyMinAgo } });

    listPods.mockResolvedValueOnce([IDLE_POD]);
    spawnAsync.mockResolvedValueOnce(ok(idleOutput));

    const result = await callListPods({ idleThresholdMinutes: 10 });
    const output = result.content[0].text as string;
    expect(output).toContain("⚠️ IDLE");
    expect(output).toContain("idle cost so far");
    expect(output).toContain("consider delete_pod");
  });

  it("ssh failure on one pod shows signal unavailable, other pod still rendered", async () => {
    listPods.mockResolvedValueOnce([ACTIVE_POD, IDLE_POD]);
    spawnAsync.mockImplementation(async (_cmd: string, args: string[]) => {
      const target = args.join(" ");
      if (target.includes("1.2.3.4")) return ok(activeOutput);
      return fail("ssh timed out");
    });

    const result = await callListPods();
    const output = result.content[0].text as string;
    expect(output).toContain("Work: ACTIVE");
    expect(output).toContain("signal unavailable (ssh failed)");
    expect(output).toContain(ACTIVE_POD.name);
    expect(output).toContain(IDLE_POD.name);
  });

  it("stays fail-soft when idle-state file I/O throws", async () => {
    loadIdleState.mockRejectedValueOnce(new Error("EACCES: permission denied"));
    saveIdleState.mockRejectedValueOnce(new Error("EACCES: permission denied"));

    listPods.mockResolvedValueOnce([ACTIVE_POD]);
    spawnAsync.mockResolvedValueOnce(ok(activeOutput));

    const result = await callListPods();
    expect(result.isError).not.toBe(true);
    const output = result.content[0].text as string;
    expect(output).toContain(ACTIVE_POD.name);
    expect(output).toContain("Work: ACTIVE");
    expect(output).not.toContain("Error");
    expect(output).not.toContain("EACCES");
  });

  it("probe:false skips SSH entirely", async () => {
    listPods.mockResolvedValueOnce([ACTIVE_POD]);

    const result = await callListPods({ probe: false });
    const output = result.content[0].text as string;
    expect(spawnAsync).not.toHaveBeenCalled();
    expect(output).not.toContain("Work:");
  });
});
