import { describe, it, expect, vi, beforeEach } from "vitest";
import { RunPodClient } from "../api.js";
// Mock global fetch
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);
function makeClient() {
    return new RunPodClient({
        apiKey: "rp_test123",
        restBaseUrl: "https://rest.runpod.io/v1",
        graphqlUrl: "https://api.runpod.io/graphql",
        sshKeyPath: "/tmp/test_key",
    });
}
function jsonResponse(data, status = 200) {
    return new Response(JSON.stringify(data), {
        status,
        headers: { "Content-Type": "application/json" },
    });
}
function errorResponse(status, body) {
    return new Response(body, { status });
}
beforeEach(() => {
    mockFetch.mockReset();
});
// ── REST API ──
describe("restRequest", () => {
    it("listPods returns normalized pods", async () => {
        const pods = [
            { id: "pod1", name: "test", desiredStatus: "RUNNING", portMappings: { "22/tcp": 10022 } },
        ];
        mockFetch.mockResolvedValueOnce(jsonResponse(pods));
        const client = makeClient();
        const result = await client.listPods();
        expect(result).toHaveLength(1);
        expect(result[0].portMappings).toEqual({ "22": 10022 });
        expect(mockFetch).toHaveBeenCalledWith("https://rest.runpod.io/v1/pods", expect.objectContaining({
            method: "GET",
            headers: expect.objectContaining({
                Authorization: "Bearer rp_test123",
            }),
        }));
    });
    it("getPod returns a single normalized pod", async () => {
        const pod = { id: "pod1", name: "test", desiredStatus: "RUNNING", portMappings: { "22/tcp": 10022 } };
        mockFetch.mockResolvedValueOnce(jsonResponse(pod));
        const client = makeClient();
        const result = await client.getPod("pod1");
        expect(result.id).toBe("pod1");
        expect(result.portMappings).toEqual({ "22": 10022 });
    });
    it("throws on HTTP error with status and body", async () => {
        mockFetch.mockResolvedValueOnce(errorResponse(401, "Unauthorized"));
        const client = makeClient();
        await expect(client.listPods()).rejects.toThrow("RunPod REST API GET /pods: 401 Unauthorized");
    });
    it("throws on 500 server error", async () => {
        mockFetch.mockResolvedValueOnce(errorResponse(500, "Internal Server Error"));
        const client = makeClient();
        await expect(client.getPod("pod1")).rejects.toThrow("RunPod REST API GET /pods/pod1: 500 Internal Server Error");
    });
    it("createPod sends correct body", async () => {
        const createdPod = { id: "new-pod", name: "my-pod", desiredStatus: "CREATED" };
        mockFetch.mockResolvedValueOnce(jsonResponse(createdPod));
        const client = makeClient();
        const result = await client.createPod({
            name: "my-pod",
            imageName: "runpod/pytorch:2.1.0",
            gpuTypeIds: ["NVIDIA GeForce RTX 3090"],
            gpuCount: 1,
        });
        expect(result.id).toBe("new-pod");
        const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
        expect(callBody.name).toBe("my-pod");
        expect(callBody.gpuTypeIds).toEqual(["NVIDIA GeForce RTX 3090"]);
    });
    it("deletePod sends DELETE request", async () => {
        mockFetch.mockResolvedValueOnce(jsonResponse({}));
        const client = makeClient();
        await client.deletePod("pod1");
        expect(mockFetch).toHaveBeenCalledWith("https://rest.runpod.io/v1/pods/pod1", expect.objectContaining({ method: "DELETE" }));
    });
    it("stopPod sends POST to stop endpoint", async () => {
        mockFetch.mockResolvedValueOnce(jsonResponse({}));
        const client = makeClient();
        await client.stopPod("pod1");
        expect(mockFetch).toHaveBeenCalledWith("https://rest.runpod.io/v1/pods/pod1/stop", expect.objectContaining({ method: "POST" }));
    });
});
// ── GraphQL API ──
describe("graphqlRequest", () => {
    it("listGpuTypes parses GPU data", async () => {
        const data = {
            data: {
                gpuTypes: [
                    {
                        id: "NVIDIA GeForce RTX 3090",
                        displayName: "RTX 3090",
                        memoryInGb: 24,
                        communityCloud: true,
                        secureCloud: false,
                        communityPrice: 0.44,
                        communitySpotPrice: 0.2,
                        securePrice: null,
                        secureSpotPrice: null,
                        lowestPrice: { minimumBidPrice: 0.2, uninterruptablePrice: 0.44, stockStatus: "High" },
                    },
                ],
            },
        };
        mockFetch.mockResolvedValueOnce(jsonResponse(data));
        const client = makeClient();
        const gpus = await client.listGpuTypes();
        expect(gpus).toHaveLength(1);
        expect(gpus[0].displayName).toBe("RTX 3090");
        expect(gpus[0].memoryInGb).toBe(24);
        expect(gpus[0].lowestPrice?.stockStatus).toBe("High");
    });
    it("throws on GraphQL errors", async () => {
        const data = {
            data: null,
            errors: [{ message: "Invalid query" }, { message: "Field not found" }],
        };
        mockFetch.mockResolvedValueOnce(jsonResponse(data));
        const client = makeClient();
        await expect(client.listGpuTypes()).rejects.toThrow("RunPod GraphQL: Invalid query, Field not found");
    });
    it("throws on HTTP error for GraphQL", async () => {
        mockFetch.mockResolvedValueOnce(errorResponse(403, "Forbidden"));
        const client = makeClient();
        await expect(client.listGpuTypes()).rejects.toThrow("RunPod GraphQL API: 403 Forbidden");
    });
    it("createSpotPod sends mutation with variables", async () => {
        const data = { data: { podRentInterruptable: { id: "spot-pod-1", imageName: "test", machineId: "m1" } } };
        mockFetch.mockResolvedValueOnce(jsonResponse(data));
        const client = makeClient();
        const result = await client.createSpotPod({
            name: "spot-test",
            imageName: "runpod/pytorch:2.1.0",
            gpuTypeIds: ["NVIDIA GeForce RTX 3090"],
            gpuCount: 1,
            bidPerGpu: 0.25,
        });
        expect(result.id).toBe("spot-pod-1");
        const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
        expect(callBody.variables.input.bidPerGpu).toBe(0.25);
        expect(callBody.variables.input.name).toBe("spot-test");
    });
});
// ── SSH Helpers ──
describe("getSshArgs", () => {
    it("returns SSH args with key path", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: "1.2.3.4", portMappings: { "22": 10022 } };
        const args = client.getSshArgs(pod);
        expect(args).toEqual([
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=15",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-p", "10022",
            "-i", "/tmp/test_key",
            "root@1.2.3.4",
        ]);
    });
    it("returns null when pod has no public IP", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: null, portMappings: { "22": 10022 } };
        expect(client.getSshArgs(pod)).toBeNull();
    });
    it("returns null when pod has no SSH port", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: "1.2.3.4", portMappings: {} };
        expect(client.getSshArgs(pod)).toBeNull();
    });
    it("getSshCommandString returns joined string", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: "1.2.3.4", portMappings: { "22": 10022 } };
        const cmd = client.getSshCommandString(pod);
        expect(cmd).toBe("ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -p 10022 -i /tmp/test_key root@1.2.3.4");
    });
});
// ── Rsync Helpers ──
describe("getRsyncArgs", () => {
    it("returns upload rsync args", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: "1.2.3.4", portMappings: { "22": 10022 } };
        const args = client.getRsyncArgs(pod, "/local/data", "/workspace/data", "upload");
        expect(args).not.toBeNull();
        expect(args[0]).toBe("rsync");
        expect(args).toContain("/local/data");
        expect(args.at(-1)).toBe("root@1.2.3.4:/workspace/data");
    });
    it("returns download rsync args", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: "1.2.3.4", portMappings: { "22": 10022 } };
        const args = client.getRsyncArgs(pod, "/local/out", "/workspace/results", "download");
        expect(args).not.toBeNull();
        expect(args.at(-1)).toBe("/local/out");
        expect(args).toContain("root@1.2.3.4:/workspace/results");
    });
    it("returns null when pod not ready", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: null };
        expect(client.getRsyncArgs(pod, "/a", "/b", "upload")).toBeNull();
    });
    it("excludes --no-same-owner (EXP-046: unsupported in rsync 3.1.3) but keeps --no-same-group", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: "1.2.3.4", portMappings: { "22": 10022 } };
        const args = client.getRsyncArgs(pod, "/local/data", "/workspace/data", "upload");
        expect(args).not.toContain("--no-same-owner");
        expect(args).toContain("--no-same-group");
    });
    it("includes --stats flag and excludes -v (verbose)", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: "1.2.3.4", portMappings: { "22": 10022 } };
        const args = client.getRsyncArgs(pod, "/local/data", "/workspace/data", "upload");
        expect(args).toContain("--stats");
        expect(args.join(" ")).not.toMatch(/\b-v\b/);
        expect(args.join(" ")).not.toMatch(/-avzP/);
    });
    it("includes --skip-compress for ML binary formats", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: "1.2.3.4", portMappings: { "22": 10022 } };
        const args = client.getRsyncArgs(pod, "/local/data", "/workspace/data", "upload");
        const skipCompress = args.find((a) => a.startsWith("--skip-compress="));
        expect(skipCompress).toBeDefined();
        expect(skipCompress).toContain("pt");
        expect(skipCompress).toContain("safetensors");
        expect(skipCompress).toContain("gguf");
    });
    it("includes --timeout=120 for rsync transfer timeout", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: "1.2.3.4", portMappings: { "22": 10022 } };
        const args = client.getRsyncArgs(pod, "/local/data", "/workspace/data", "upload");
        expect(args).toContain("--timeout=120");
    });
    it("includes ConnectTimeout and BatchMode in rsync ssh command", () => {
        const client = makeClient();
        const pod = { id: "p1", name: "test", desiredStatus: "RUNNING", publicIp: "1.2.3.4", portMappings: { "22": 10022 } };
        const args = client.getRsyncArgs(pod, "/local/data", "/workspace/data", "upload");
        const sshArg = args.find((a) => a.startsWith("ssh "));
        expect(sshArg).toContain("ConnectTimeout=15");
        expect(sshArg).toContain("BatchMode=yes");
        // ServerAlive excluded from rsync ssh (false positives during idle compression)
        expect(sshArg).not.toContain("ServerAliveInterval");
    });
});
// ── waitForPod ──
describe("waitForPod", () => {
    it("resolves when pod is running with SSH ready", async () => {
        const client = makeClient();
        const readyPod = {
            id: "p1", name: "test", desiredStatus: "RUNNING",
            publicIp: "1.2.3.4", portMappings: { "22": 10022 },
        };
        vi.spyOn(client, "getPod").mockResolvedValue(readyPod);
        vi.spyOn(client, "tcpProbe").mockResolvedValue(true);
        const pod = await client.waitForPod("p1", 5000, 100);
        expect(pod.id).toBe("p1");
    });
    it("throws on terminal state (EXITED)", async () => {
        const client = makeClient();
        const exitedPod = { id: "p1", name: "test", desiredStatus: "EXITED" };
        vi.spyOn(client, "getPod").mockResolvedValue(exitedPod);
        await expect(client.waitForPod("p1", 5000, 100)).rejects.toThrow("terminal state: EXITED");
    });
    it("throws on timeout when pod never becomes ready", async () => {
        const client = makeClient();
        const pendingPod = { id: "p1", name: "test", desiredStatus: "CREATED" };
        vi.spyOn(client, "getPod").mockResolvedValue(pendingPod);
        await expect(client.waitForPod("p1", 300, 100)).rejects.toThrow("did not become ready");
    });
    it("calls onProgress callback with status updates", async () => {
        const client = makeClient();
        const pendingPod = { id: "p1", name: "test", desiredStatus: "CREATED" };
        const readyPod = {
            id: "p1", name: "test", desiredStatus: "RUNNING",
            publicIp: "1.2.3.4", portMappings: { "22": 10022 },
        };
        let callCount = 0;
        vi.spyOn(client, "getPod").mockImplementation(async () => {
            callCount++;
            return (callCount >= 2 ? readyPod : pendingPod);
        });
        vi.spyOn(client, "tcpProbe").mockResolvedValue(true);
        const messages = [];
        await client.waitForPod("p1", 5000, 50, (msg) => messages.push(msg));
        expect(messages.length).toBeGreaterThan(0);
        expect(messages.some((m) => m.includes("CREATED") || m.includes("RUNNING"))).toBe(true);
    });
});
// ── Network Volume API ──
describe("network volume methods", () => {
    it("listNetworkVolumes returns volumes from GraphQL", async () => {
        const volumes = [
            { id: "vol1", name: "train-data", size: 20, dataCenterId: "US-TX-3" },
            { id: "vol2", name: "checkpoints", size: 50, dataCenterId: "EU-RO-1" },
        ];
        mockFetch.mockResolvedValueOnce(new Response(JSON.stringify({ data: { myself: { networkVolumes: volumes } } })));
        const client = makeClient();
        const result = await client.listNetworkVolumes();
        expect(result).toHaveLength(2);
        expect(result[0].name).toBe("train-data");
        expect(result[1].dataCenterId).toBe("EU-RO-1");
    });
    it("getNetworkVolume returns matching volume", async () => {
        const volumes = [
            { id: "vol1", name: "train-data", size: 20, dataCenterId: "US-TX-3" },
        ];
        mockFetch.mockResolvedValueOnce(new Response(JSON.stringify({ data: { myself: { networkVolumes: volumes } } })));
        const client = makeClient();
        const result = await client.getNetworkVolume("vol1");
        expect(result).not.toBeNull();
        expect(result.name).toBe("train-data");
    });
    it("getNetworkVolume returns null for missing volume", async () => {
        mockFetch.mockResolvedValueOnce(new Response(JSON.stringify({ data: { myself: { networkVolumes: [] } } })));
        const client = makeClient();
        const result = await client.getNetworkVolume("nonexistent");
        expect(result).toBeNull();
    });
    it("createNetworkVolume sends correct GraphQL mutation", async () => {
        const created = { id: "vol-new", name: "my-vol", size: 10, dataCenterId: "US-TX-3" };
        mockFetch.mockResolvedValueOnce(new Response(JSON.stringify({ data: { createNetworkVolume: created } })));
        const client = makeClient();
        const result = await client.createNetworkVolume("my-vol", 10, "US-TX-3");
        expect(result.id).toBe("vol-new");
        expect(result.size).toBe(10);
        // Verify the request body
        const call = mockFetch.mock.calls[0];
        const body = JSON.parse(call[1].body);
        expect(body.query).toContain("createNetworkVolume");
        expect(body.variables.input).toEqual({ name: "my-vol", size: 10, dataCenterId: "US-TX-3" });
    });
    it("deleteNetworkVolume calls REST DELETE", async () => {
        mockFetch.mockResolvedValueOnce(new Response(JSON.stringify({})));
        const client = makeClient();
        await client.deleteNetworkVolume("vol-123");
        const call = mockFetch.mock.calls[0];
        expect(call[0]).toContain("/networkvolumes/vol-123");
        expect(call[1].method).toBe("DELETE");
    });
});
// ── Network error simulation ──
describe("network errors", () => {
    it("handles fetch rejection (network timeout)", async () => {
        mockFetch.mockRejectedValueOnce(new TypeError("fetch failed"));
        const client = makeClient();
        await expect(client.listPods()).rejects.toThrow("fetch failed");
    });
});
// ── spawnAsync ──
import { spawnAsync } from "../api.js";
describe("spawnAsync", () => {
    it("captures stdout from a simple command", async () => {
        const result = await spawnAsync("echo", ["hello"]);
        expect(result.stdout.trim()).toBe("hello");
        expect(result.stderr).toBe("");
        expect(result.status).toBe(0);
        expect(result.error).toBeUndefined();
    });
    it("captures stderr and non-zero exit code", async () => {
        const result = await spawnAsync("bash", ["-c", "echo err >&2; exit 42"]);
        expect(result.stderr.trim()).toBe("err");
        expect(result.status).toBe(42);
    });
    it("returns error for non-existent command", async () => {
        const result = await spawnAsync("__nonexistent_cmd__", []);
        expect(result.error).toBeDefined();
    });
    it("respects timeout", async () => {
        const result = await spawnAsync("sleep", ["10"], { timeout: 500 });
        // Node kills the process on timeout, status is null (signaled, not exited)
        expect(result.status).toBeNull();
    });
});
//# sourceMappingURL=api.test.js.map