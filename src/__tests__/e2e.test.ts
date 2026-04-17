/**
 * E2E integration tests — HTTP-level mocking of RunPod API.
 * Tests full tool flows: REST lifecycle, error handling, cleanup orchestration.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { RunPodClient } from "../api.js";
import { filterStalePods, deletePodWithStop } from "../pod-ops.js";
import type { Pod } from "../types.js";

// ── HTTP-level mock ──

function mockFetch(responses: Array<{ status: number; body: unknown }>) {
  let callIdx = 0;
  return vi.fn(async () => {
    const resp = responses[callIdx] ?? responses[responses.length - 1];
    callIdx++;
    return {
      ok: resp.status >= 200 && resp.status < 300,
      status: resp.status,
      text: async () => JSON.stringify(resp.body),
      json: async () => resp.body,
    };
  });
}

// ── Fixtures ──

const POD_RUNNING: Pod = {
  id: "pod-abc123",
  name: "e2e-test-pod",
  desiredStatus: "RUNNING",
  publicIp: "1.2.3.4",
  portMappings: { "22/tcp": 22222 },
  lastStatusChange: new Date().toISOString(),
};

const POD_EXITED: Pod = {
  ...POD_RUNNING,
  desiredStatus: "EXITED",
  publicIp: undefined,
  portMappings: undefined,
};

const POD_STALE: Pod = {
  id: "pod-stale1",
  name: "old-training",
  desiredStatus: "EXITED",
  lastStatusChange: new Date(Date.now() - 48 * 3600_000).toISOString(),
};

// ── Scenario 1: Happy path — pod lifecycle ──

describe("E2E: pod lifecycle (create → get → delete)", () => {
  let client: RunPodClient;
  let fetchMock: ReturnType<typeof mockFetch>;

  beforeEach(() => {
    client = new RunPodClient({
      apiKey: "rp_test_key",
      restBaseUrl: "https://rest.runpod.io/v1",
      graphqlUrl: "https://api.runpod.io/graphql",
    });
  });

  it("creates a pod via REST and retrieves it", async () => {
    const createdPod = { ...POD_RUNNING, id: "pod-new123" };
    fetchMock = mockFetch([
      { status: 200, body: createdPod },  // createPod
      { status: 200, body: createdPod },  // getPod
    ]);
    vi.stubGlobal("fetch", fetchMock);

    const pod = await client.createPod({
      name: "test-pod",
      imageName: "pytorch/pytorch:latest",
      gpuTypeIds: ["NVIDIA GeForce RTX 3090"],
      cloudType: "COMMUNITY",
    });
    expect(pod.id).toBe("pod-new123");
    expect(pod.desiredStatus).toBe("RUNNING");

    const fetched = await client.getPod("pod-new123");
    expect(fetched.id).toBe("pod-new123");

    // Verify REST calls made
    expect(fetchMock).toHaveBeenCalledTimes(2);

    vi.unstubAllGlobals();
  });

  it("deletes a stopped pod directly", async () => {
    fetchMock = mockFetch([
      { status: 200, body: POD_EXITED },  // getPod (check status)
      { status: 200, body: "" },           // deletePod
    ]);
    vi.stubGlobal("fetch", fetchMock);

    const { wasRunning } = await deletePodWithStop(client, "pod-abc123");
    expect(wasRunning).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    vi.unstubAllGlobals();
  });

  it("stops a running pod before deletion", async () => {
    let callCount = 0;
    fetchMock = vi.fn(async (url: string, opts: any) => {
      callCount++;
      if (callCount === 1) {
        // getPod → RUNNING
        return { ok: true, status: 200, text: async () => JSON.stringify(POD_RUNNING), json: async () => POD_RUNNING };
      }
      if (opts?.method === "POST" && String(url).includes("/stop")) {
        // stopPod
        return { ok: true, status: 200, text: async () => JSON.stringify({}), json: async () => ({}) };
      }
      if (callCount <= 3) {
        // getPod poll → EXITED
        return { ok: true, status: 200, text: async () => JSON.stringify(POD_EXITED), json: async () => POD_EXITED };
      }
      // deletePod
      return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
    }) as any;
    vi.stubGlobal("fetch", fetchMock);

    const { wasRunning } = await deletePodWithStop(client, "pod-abc123", 10_000);
    expect(wasRunning).toBe(true);

    vi.unstubAllGlobals();
  });
});

// ── Scenario 2: Error paths ──

describe("E2E: API error handling", () => {
  let client: RunPodClient;

  beforeEach(() => {
    client = new RunPodClient({
      apiKey: "rp_invalid_key",
      restBaseUrl: "https://rest.runpod.io/v1",
      graphqlUrl: "https://api.runpod.io/graphql",
    });
  });

  it("throws on 401 Unauthorized", async () => {
    const fetchMock = mockFetch([
      { status: 401, body: "Unauthorized" },
    ]);
    vi.stubGlobal("fetch", fetchMock);

    await expect(client.listPods()).rejects.toThrow("401");

    vi.unstubAllGlobals();
  });

  it("throws on 500 Internal Server Error", async () => {
    const fetchMock = mockFetch([
      { status: 500, body: "Internal Server Error" },
    ]);
    vi.stubGlobal("fetch", fetchMock);

    await expect(client.getPod("pod-123")).rejects.toThrow("500");

    vi.unstubAllGlobals();
  });

  it("GraphQL error with no data throws", async () => {
    const fetchMock = mockFetch([
      { status: 200, body: { errors: [{ message: "Rate limited" }], data: null } },
    ]);
    vi.stubGlobal("fetch", fetchMock);

    await expect(client.listGpuTypes()).rejects.toThrow("Rate limited");

    vi.unstubAllGlobals();
  });

  it("GraphQL partial error returns data with warning", async () => {
    const stderrSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const fetchMock = mockFetch([
      {
        status: 200,
        body: {
          errors: [{ message: "lowestPrice unavailable" }],
          data: { gpuTypes: [{ id: "GPU-A", displayName: "A", memoryInGb: 24 }] },
        },
      },
    ]);
    vi.stubGlobal("fetch", fetchMock);

    const gpus = await client.listGpuTypes();
    expect(gpus).toHaveLength(1);
    expect(gpus[0].id).toBe("GPU-A");
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("partial error"));

    stderrSpy.mockRestore();
    vi.unstubAllGlobals();
  });
});

// ── Scenario 3: Cleanup orchestration flow ──

describe("E2E: cleanup flow (listPods → filter → delete)", () => {
  it("full cleanup pipeline: list → filterStalePods → deletePodWithStop", async () => {
    const pods: Pod[] = [
      POD_STALE,
      { ...POD_RUNNING, id: "pod-running1", name: "active-training" },
      { id: "pod-keep1", name: "keep-permanent", desiredStatus: "EXITED", lastStatusChange: new Date(Date.now() - 100 * 3600_000).toISOString() },
    ];

    // Step 1: filter stale pods (pure function)
    const { stale, skipped } = filterStalePods(pods, 24); // 24h grace
    expect(stale).toHaveLength(1);
    expect(stale[0].pod.id).toBe("pod-stale1");
    expect(stale[0].idleHours).toBe(48);
    expect(skipped).toHaveLength(2);
    expect(skipped.find(s => s.pod.id === "pod-running1")?.reason).toContain("RUNNING");
    expect(skipped.find(s => s.pod.id === "pod-keep1")?.reason).toContain("keep/persist");

    // Step 2: delete each stale pod via client
    const client = new RunPodClient({
      apiKey: "rp_test",
      restBaseUrl: "https://rest.runpod.io/v1",
      graphqlUrl: "https://api.runpod.io/graphql",
    });

    const fetchMock = mockFetch([
      { status: 200, body: { ...POD_STALE, desiredStatus: "EXITED" } }, // getPod
      { status: 200, body: "" },                                          // deletePod
    ]);
    vi.stubGlobal("fetch", fetchMock);

    const deleted: string[] = [];
    const failed: string[] = [];

    for (const { pod } of stale) {
      try {
        await deletePodWithStop(client, pod.id);
        deleted.push(pod.id);
      } catch (e) {
        failed.push(pod.id);
      }
    }

    expect(deleted).toEqual(["pod-stale1"]);
    expect(failed).toHaveLength(0);

    vi.unstubAllGlobals();
  });

  it("cleanup handles delete failure correctly (pushes to failed, not deleted)", async () => {
    const pods: Pod[] = [
      { id: "pod-fail1", name: "will-fail", desiredStatus: "EXITED", lastStatusChange: new Date(Date.now() - 72 * 3600_000).toISOString() },
    ];

    const { stale } = filterStalePods(pods, 24);
    expect(stale).toHaveLength(1);

    const client = new RunPodClient({
      apiKey: "rp_test",
      restBaseUrl: "https://rest.runpod.io/v1",
      graphqlUrl: "https://api.runpod.io/graphql",
    });

    const fetchMock = mockFetch([
      { status: 200, body: { ...pods[0], desiredStatus: "EXITED" } }, // getPod
      { status: 500, body: "Internal Server Error" },                  // deletePod fails
    ]);
    vi.stubGlobal("fetch", fetchMock);

    const deleted: string[] = [];
    const failed: string[] = [];

    for (const { pod } of stale) {
      try {
        await deletePodWithStop(client, pod.id);
        deleted.push(pod.id);
      } catch (e) {
        failed.push(pod.id); // T1 bug fix: must go to failed, not deleted
      }
    }

    expect(deleted).toHaveLength(0);
    expect(failed).toEqual(["pod-fail1"]);

    vi.unstubAllGlobals();
  });
});
