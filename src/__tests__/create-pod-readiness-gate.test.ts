import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Pod } from "../types.js";

const listPods = vi.fn();
const listGpuTypes = vi.fn();
const createPod = vi.fn();

vi.mock("../api.js", async () => {
  const actual = await vi.importActual<typeof import("../api.js")>("../api.js");
  class FakeRunPodClient {
    listPods = listPods;
    listGpuTypes = listGpuTypes;
    createPod = createPod;
    getPod = vi.fn();
    getSshArgs() {
      return null;
    }
    getSshCommandString() {
      return null;
    }
  }
  return { ...actual, RunPodClient: FakeRunPodClient };
});

process.env.RUNPOD_API_KEY = "rp_test_key";

const { server } = await import("../index.js");
const { loadReadinessState, saveReadinessState, recordGate } = await import("../readiness.js");

const CREATED_POD: Pod = {
  id: "pod-new",
  name: "fresh",
  desiredStatus: "RUNNING",
  publicIp: undefined,
  portMappings: undefined,
  costPerHr: 0.22,
};

const GPU_TYPE = {
  id: "NVIDIA GeForce RTX 3090",
  displayName: "RTX 3090",
  memoryInGb: 24,
  secureCloud: false,
  communityCloud: true,
  securePrice: 0.44,
  communityPrice: 0.22,
  oneMonthPrice: null,
  threeMonthPrice: null,
  lowestPrice: { minimumBidPrice: 0.11, uninterruptablePrice: 0.22, stockStatus: "High" },
};

async function callCreatePodAuto(args: Record<string, unknown> = {}) {
  const tool = (server as any)._registeredTools["create_pod_auto"];
  const withDefaults = {
    name: "test-pod",
    imageName: "runpod/pytorch:2.1.0",
    gpuPreference: ["NVIDIA GeForce RTX 3090"],
    minVram: 12,
    gpuCount: 1,
    spot: false,
    maxBidPerGpu: 0.3,
    containerDiskInGb: 50,
    volumeInGb: 20,
    optimizePytorch: false,
    cloudType: "COMMUNITY",
    dryRun: false,
    cpuOnly: false,
    vcpuCount: 2,
    skipReadinessGate: false,
    ...args,
  };
  const result = await tool.handler(withDefaults, {});
  return result.content[0].text as string;
}

describe("create_pod_auto import-readiness gate", () => {
  let dir: string;
  const originalCwd = process.cwd();

  beforeEach(async () => {
    listPods.mockReset();
    listGpuTypes.mockReset().mockResolvedValue([GPU_TYPE]);
    createPod.mockReset().mockResolvedValue(CREATED_POD);
    dir = await mkdtemp(join(tmpdir(), "readiness-gate-"));
    process.chdir(dir);
  });

  afterEach(async () => {
    process.chdir(originalCwd);
    await rm(dir, { recursive: true, force: true });
  });

  it("creates the pod and opens a gate when imports are given", async () => {
    listPods.mockResolvedValue([]);
    const output = await callCreatePodAuto({ imports: ["torch", "kornia"] });

    expect(output).toContain("pod-new");
    expect(output).toContain("NOT READY");
    expect(output).toContain("run_preflight");
    expect(output).toContain("kornia");

    const state = await loadReadinessState();
    expect(state["pod-new"]).toMatchObject({ imports: ["torch", "kornia"], verifiedAt: null });
  });

  it("refuses the NEXT pod while an earlier gate is unverified", async () => {
    await saveReadinessState(recordGate({}, "pod-first", ["torch"], new Date()));
    listPods.mockResolvedValue([{ ...CREATED_POD, id: "pod-first" }]);

    const output = await callCreatePodAuto({ imports: ["torch"] });

    expect(output).toContain("팟 생성 거부");
    expect(output).toContain("pod-first");
    expect(createPod).not.toHaveBeenCalled();
  });

  it("allows creation once the gate is verified", async () => {
    const { markVerified } = await import("../readiness.js");
    await saveReadinessState(markVerified(recordGate({}, "pod-first", ["torch"], new Date()), "pod-first", new Date()));
    listPods.mockResolvedValue([{ ...CREATED_POD, id: "pod-first" }]);

    const output = await callCreatePodAuto();

    expect(output).toContain("pod-new");
    expect(createPod).toHaveBeenCalled();
  });

  it("drops gates for pods that no longer exist, instead of blocking forever", async () => {
    await saveReadinessState(recordGate({}, "pod-deleted", ["torch"], new Date()));
    listPods.mockResolvedValue([]); // the gated pod is gone

    const output = await callCreatePodAuto();

    expect(createPod).toHaveBeenCalled();
    expect(output).toContain("pod-new");
    expect(await loadReadinessState()).toEqual({});
  });

  it("honors skipReadinessGate", async () => {
    await saveReadinessState(recordGate({}, "pod-first", ["torch"], new Date()));
    listPods.mockResolvedValue([{ ...CREATED_POD, id: "pod-first" }]);

    await callCreatePodAuto({ skipReadinessGate: true });

    expect(createPod).toHaveBeenCalled();
    // Bypassing the refusal must not silently release the other pod's gate.
    expect(await loadReadinessState()).toHaveProperty("pod-first");
  });

  it("does not gate a CPU staging pod (Staging Pod Pattern stays usable)", async () => {
    await saveReadinessState(recordGate({}, "pod-first", ["torch"], new Date()));
    listPods.mockResolvedValue([{ ...CREATED_POD, id: "pod-first" }]);

    const output = await callCreatePodAuto({ cpuOnly: true, cpuFlavorIds: undefined });

    expect(output).not.toContain("팟 생성 거부");
    expect(await loadReadinessState()).toHaveProperty("pod-first");
  });

  it("does not gate a dryRun", async () => {
    await saveReadinessState(recordGate({}, "pod-first", ["torch"], new Date()));
    listPods.mockResolvedValue([{ ...CREATED_POD, id: "pod-first" }]);

    const output = await callCreatePodAuto({ dryRun: true });

    expect(output).not.toContain("팟 생성 거부");
    expect(createPod).not.toHaveBeenCalled();
  });

  it("stays fail-soft when the live-pod lookup fails mid-gate-check", async () => {
    // A pending gate exists, so the gate WOULD refuse — but listPods throwing must not
    // turn gate bookkeeping into a pod-creation outage.
    await saveReadinessState(recordGate({}, "pod-first", ["torch"], new Date()));
    listPods.mockRejectedValue(new Error("EIO: runpod api unreachable"));

    const output = await callCreatePodAuto({ imports: ["torch"] });

    expect(createPod).toHaveBeenCalled();
    expect(output).toContain("pod-new");
    expect(output).not.toContain("팟 생성 거부");
  });

  it("creates without a gate when imports are omitted", async () => {
    listPods.mockResolvedValue([]);
    const output = await callCreatePodAuto();

    expect(output).not.toContain("NOT READY");
    expect(await loadReadinessState()).toEqual({});
  });

  it("serializes concurrent calls so the second one sees the first one's gate", async () => {
    // Without the lock both calls pass the check before either records a gate — the exact
    // double-pod shape this feature exists to prevent.
    const created: string[] = [];
    createPod.mockImplementation(async () => {
      const id = `pod-${created.length + 1}`;
      created.push(id);
      return { ...CREATED_POD, id };
    });
    listPods.mockImplementation(async () => created.map((id) => ({ ...CREATED_POD, id })));

    const [first, second] = await Promise.all([
      callCreatePodAuto({ imports: ["torch"] }),
      callCreatePodAuto({ imports: ["torch"] }),
    ]);

    const refusals = [first, second].filter((o) => o.includes("팟 생성 거부"));
    expect(refusals).toHaveLength(1);
    expect(createPod).toHaveBeenCalledTimes(1);
  });

  it("does not poison the lock when a call fails — later calls still work", async () => {
    // withReadinessLock shares one promise chain across calls. If a rejection were left on it,
    // every later create_pod_auto would be permanently blocked.
    listPods.mockResolvedValue([]);
    // listGpuTypes rejects outside the per-DC try/catch, so the rejection reaches the lock.
    listGpuTypes.mockRejectedValueOnce(new Error("runpod api exploded"));

    await callCreatePodAuto();

    listGpuTypes.mockResolvedValue([GPU_TYPE]);
    const output = await callCreatePodAuto();

    expect(output).toContain("pod-new");
  });
});
