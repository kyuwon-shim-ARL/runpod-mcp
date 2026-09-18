import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Pod } from "../types.js";

const spawnAsync = vi.fn();
const getPod = vi.fn();

vi.mock("../api.js", async () => {
  const actual = await vi.importActual<typeof import("../api.js")>("../api.js");
  class FakeRunPodClient {
    listPods = vi.fn();
    getPod = getPod;
    getSshArgs(pod: Pod) {
      if (!pod.publicIp || !pod.portMappings?.["22"]) return null;
      return ["ssh", "-p", String(pod.portMappings["22"]), `root@${pod.publicIp}`];
    }
    getSshCommandString() {
      return null;
    }
  }
  return { ...actual, spawnAsync, RunPodClient: FakeRunPodClient };
});

process.env.RUNPOD_API_KEY = "rp_test_key";

const { server } = await import("../index.js");
const { loadReadinessState, saveReadinessState, recordGate } = await import("../readiness.js");

const POD: Pod = {
  id: "pod-gated",
  name: "gated",
  desiredStatus: "RUNNING",
  publicIp: "1.2.3.4",
  portMappings: { "22": 40001 },
  costPerHr: 0.22,
};

const ok = (stdout: string) => ({ status: 0, stdout, stderr: "", error: undefined });

async function callPreflight(args: Record<string, unknown>) {
  const tool = (server as any)._registeredTools["run_preflight"];
  const result = await tool.handler(
    { podId: "pod-gated", minDiskFreeGb: 10, strict: false, allowNvStreaming: false, ...args },
    {}
  );
  return result.content[0].text as string;
}

/**
 * run_preflight issues several SSH calls (CUDA, disk, NV, ...) before the import smokes,
 * and their order is an implementation detail. Dispatch on the command text instead of on
 * call order, and decide each import by the module it names.
 */
function stubSsh(failingModules: string[] = []) {
  spawnAsync.mockReset();
  spawnAsync.mockImplementation(async (_cmd: string, args: string[]) => {
    const command = args[args.length - 1] ?? "";
    if (command.includes("torch.cuda.is_available") || command.includes("CUDA:")) {
      return ok("CUDA:OK cuda_build=11.8 driver=550 torch=2.1.0");
    }
    if (command.startsWith("python3 -c") && command.includes("__IMPORT_OK__")) {
      const failed = failingModules.find((m) => command.includes(m));
      return failed
        ? ok(`ModuleNotFoundError: No module named '${failed}'\n__IMPORT_FAIL__\n`)
        : ok("__IMPORT_OK__\n");
    }
    return ok("");
  });
}

describe("run_preflight releases the import-readiness gate", () => {
  let dir: string;
  const originalCwd = process.cwd();

  beforeEach(async () => {
    getPod.mockReset().mockResolvedValue(POD);
    dir = await mkdtemp(join(tmpdir(), "preflight-gate-"));
    process.chdir(dir);
  });

  afterEach(async () => {
    process.chdir(originalCwd);
    await rm(dir, { recursive: true, force: true });
  });

  it("opens the gate when every gated import passes", async () => {
    await saveReadinessState(recordGate({}, "pod-gated", ["torch", "kornia"], new Date()));
    stubSsh();

    const output = await callPreflight({ importSmokes: ["torch", "kornia"] });

    expect(output).toContain("게이트 해제");
    const state = await loadReadinessState();
    expect(state["pod-gated"].verifiedAt).not.toBeNull();
  });

  it("keeps the gate shut when a gated import fails", async () => {
    await saveReadinessState(recordGate({}, "pod-gated", ["torch", "kornia"], new Date()));
    stubSsh(["kornia"]);

    const output = await callPreflight({ importSmokes: ["torch", "kornia"] });

    expect(output).toContain("❌");
    const state = await loadReadinessState();
    expect(state["pod-gated"].verifiedAt).toBeNull();
  });

  it("does NOT open the gate when a different, easier import is verified instead", async () => {
    // The regression this guards: verifying ["os"] must not release a gate demanding
    // ["torch","kornia"] — that would certify a check that never ran.
    await saveReadinessState(recordGate({}, "pod-gated", ["torch", "kornia"], new Date()));
    stubSsh();

    const output = await callPreflight({ importSmokes: ["os"] });

    expect(output).toContain("게이트 유지");
    expect(output).toContain("kornia");
    const state = await loadReadinessState();
    expect(state["pod-gated"].verifiedAt).toBeNull();
  });

  it("opens the gate when the verified set is a superset of what it demanded", async () => {
    await saveReadinessState(recordGate({}, "pod-gated", ["torch"], new Date()));
    stubSsh();

    await callPreflight({ importSmokes: ["torch", "pandas"] });

    const state = await loadReadinessState();
    expect(state["pod-gated"].verifiedAt).not.toBeNull();
  });

  it("matches gated imports written as statements against bare-name smokes", async () => {
    await saveReadinessState(recordGate({}, "pod-gated", ["import torch"], new Date()));
    stubSsh();

    await callPreflight({ importSmokes: ["torch"] });

    const state = await loadReadinessState();
    expect(state["pod-gated"].verifiedAt).not.toBeNull();
  });

  it("says nothing about gates for a pod that has none", async () => {
    stubSsh();

    const output = await callPreflight({ importSmokes: ["torch"] });

    expect(output).not.toContain("게이트");
    expect(await loadReadinessState()).toEqual({});
  });
});
