import { describe, it, expect, vi } from "vitest";
import { filterStalePods, selectGpuCandidates, deletePodWithStop, DEFAULT_DC_PRIORITY, formatDcGpuFailureMatrix, sanitizePodName, isoToDateStamp, buildPodMetadataPath, toYaml, buildPodMetadataStub, parseDuBytes, parseDfAvailBytes, checkFreeSpace, checkSizeMatch, looksLikeSetupCommand, estimatePodCost } from "../pod-ops.js";
// ── Helper factories ──
function makePod(overrides = {}) {
    return {
        id: "pod-1",
        name: "test-pod",
        desiredStatus: "EXITED",
        lastStatusChange: new Date(Date.now() - 5 * 3600_000).toISOString(),
        ...overrides,
    };
}
function makeGpu(overrides = {}) {
    return {
        id: "NVIDIA GeForce RTX 3090",
        displayName: "RTX 3090",
        memoryInGb: 24,
        secureCloud: true,
        communityCloud: true,
        lowestPrice: {
            minimumBidPrice: 0.2,
            uninterruptablePrice: 0.44,
            stockStatus: "High",
        },
        ...overrides,
    };
}
// ── Integration: filterStalePods → deletePodWithStop pipeline ──
describe("orchestration: stale detection → deletion pipeline", () => {
    it("filters stale pods then deletes them via deletePodWithStop", async () => {
        const now = Date.now();
        const pods = [
            makePod({ id: "stale-1", name: "old-pod", lastStatusChange: new Date(now - 10 * 3600_000).toISOString() }),
            makePod({ id: "keep-1", name: "keep-important", desiredStatus: "EXITED" }),
            makePod({ id: "running-1", desiredStatus: "RUNNING" }),
        ];
        // Step 1: filter
        const { stale, skipped } = filterStalePods(pods, 2, now);
        expect(stale).toHaveLength(1);
        expect(stale[0].pod.id).toBe("stale-1");
        expect(skipped).toHaveLength(2);
        // Step 2: delete each stale pod
        const client = {
            getPod: vi.fn().mockResolvedValue({ id: "stale-1", desiredStatus: "EXITED" }),
            stopPod: vi.fn().mockResolvedValue(undefined),
            deletePod: vi.fn().mockResolvedValue(undefined),
        };
        const { wasRunning } = await deletePodWithStop(client, stale[0].pod.id);
        expect(wasRunning).toBe(false);
        expect(client.deletePod).toHaveBeenCalledWith("stale-1");
    });
    it("handles stale pod that is still RUNNING (auto-stop)", async () => {
        const now = Date.now();
        // Edge case: pod has EXITED in desiredStatus but is actually RUNNING when we try to delete
        const pods = [
            makePod({ id: "zombie-1", name: "zombie", lastStatusChange: new Date(now - 8 * 3600_000).toISOString() }),
        ];
        const { stale } = filterStalePods(pods, 2, now);
        expect(stale).toHaveLength(1);
        let callCount = 0;
        const client = {
            getPod: vi.fn().mockImplementation(async () => {
                callCount++;
                if (callCount === 1)
                    return { id: "zombie-1", desiredStatus: "RUNNING" };
                return { id: "zombie-1", desiredStatus: "EXITED" };
            }),
            stopPod: vi.fn().mockResolvedValue(undefined),
            deletePod: vi.fn().mockResolvedValue(undefined),
        };
        const { wasRunning } = await deletePodWithStop(client, "zombie-1", 5_000);
        expect(wasRunning).toBe(true);
        expect(client.stopPod).toHaveBeenCalledWith("zombie-1");
        expect(client.deletePod).toHaveBeenCalledWith("zombie-1");
    });
});
// ── Integration: selectGpuCandidates → create_pod_auto flow ──
describe("orchestration: GPU selection → pod creation flow", () => {
    it("selects cheapest viable GPU from preference list", () => {
        const gpus = [
            makeGpu({ id: "GPU-EXPENSIVE", displayName: "A100", memoryInGb: 80, lowestPrice: { minimumBidPrice: 1.0, uninterruptablePrice: 3.0, stockStatus: "High" } }),
            makeGpu({ id: "GPU-CHEAP", displayName: "RTX 3090", memoryInGb: 24, lowestPrice: { minimumBidPrice: 0.2, uninterruptablePrice: 0.44, stockStatus: "High" } }),
            makeGpu({ id: "GPU-TINY", displayName: "RTX 3060", memoryInGb: 12, lowestPrice: { minimumBidPrice: 0.1, uninterruptablePrice: 0.2, stockStatus: "High" } }),
        ];
        // User prefers cheap → expensive
        const { candidates } = selectGpuCandidates(gpus, {
            gpuPreference: ["GPU-CHEAP", "GPU-TINY", "GPU-EXPENSIVE"],
            minVram: 20,
            gpuCount: 1,
            spot: false,
            maxBidPerGpu: 0.5,
        });
        // GPU-TINY (12GB) is below minVram 20GB, so should be filtered
        expect(candidates).toHaveLength(2);
        expect(candidates[0].gpuId).toBe("GPU-CHEAP");
        expect(candidates[1].gpuId).toBe("GPU-EXPENSIVE");
    });
    it("spot bidding: skips GPUs where bid is below minimum", () => {
        const gpus = [
            makeGpu({ id: "GPU-A", displayName: "A", memoryInGb: 24, lowestPrice: { minimumBidPrice: 0.5, uninterruptablePrice: 1.0, stockStatus: "High" } }),
            makeGpu({ id: "GPU-B", displayName: "B", memoryInGb: 24, lowestPrice: { minimumBidPrice: 0.1, uninterruptablePrice: 0.3, stockStatus: "High" } }),
        ];
        const { candidates, errors } = selectGpuCandidates(gpus, {
            gpuPreference: ["GPU-A", "GPU-B"],
            minVram: 12,
            gpuCount: 1,
            spot: true,
            maxBidPerGpu: 0.2, // Below GPU-A's min bid of 0.5
        });
        // GPU-A should be skipped (bid 0.2 < min 0.5), GPU-B should work (bid 0.2 > min 0.1)
        expect(candidates).toHaveLength(1);
        expect(candidates[0].gpuId).toBe("GPU-B");
        expect(errors).toHaveLength(1);
        expect(errors[0]).toContain("A:"); // uses displayName
    });
    it("all GPUs out of stock returns empty candidates", () => {
        const gpus = [
            makeGpu({ id: "GPU-A", lowestPrice: { minimumBidPrice: 0, uninterruptablePrice: 0, stockStatus: "Out of Stock" } }),
            makeGpu({ id: "GPU-B", lowestPrice: { minimumBidPrice: 0, uninterruptablePrice: 0, stockStatus: "Out of Stock" } }),
        ];
        const { candidates, errors } = selectGpuCandidates(gpus, {
            gpuPreference: ["GPU-A", "GPU-B"],
            minVram: 12,
            gpuCount: 1,
            spot: false,
            maxBidPerGpu: 0.5,
        });
        expect(candidates).toHaveLength(0);
        expect(errors).toHaveLength(0);
    });
    it("warns on overprovisioned GPU selection", () => {
        const gpus = [
            makeGpu({ id: "GPU-BIG", displayName: "A100 80GB", memoryInGb: 80 }),
        ];
        const { candidates } = selectGpuCandidates(gpus, {
            gpuPreference: ["GPU-BIG"],
            minVram: 16, // 80GB >> 16GB needed
            gpuCount: 1,
            spot: false,
            maxBidPerGpu: 5.0,
        });
        expect(candidates).toHaveLength(1);
        expect(candidates[0].overprovisionWarning).toContain("Overprovisioned");
        expect(candidates[0].overprovisionWarning).toContain("80GB");
    });
});
// ── DC priority fallback (create_pod_auto helpers) ──
describe("DEFAULT_DC_PRIORITY", () => {
    it("contains expected DCs in priority order (largest pools first)", () => {
        expect(DEFAULT_DC_PRIORITY[0]).toBe("US-GA-1");
        expect(DEFAULT_DC_PRIORITY[1]).toBe("US-CA-2");
        expect(DEFAULT_DC_PRIORITY).toContain("EU-SE-1");
        expect(DEFAULT_DC_PRIORITY).toContain("AP-JP-1");
        // US-TX-3 and EU-RO-1 are smaller pools — must be near the end
        const txIdx = DEFAULT_DC_PRIORITY.indexOf("US-TX-3");
        const gaIdx = DEFAULT_DC_PRIORITY.indexOf("US-GA-1");
        expect(txIdx).toBeGreaterThan(gaIdx);
    });
    it("has no duplicates", () => {
        const set = new Set(DEFAULT_DC_PRIORITY);
        expect(set.size).toBe(DEFAULT_DC_PRIORITY.length);
    });
});
describe("formatDcGpuFailureMatrix", () => {
    it("returns empty string for empty input", () => {
        expect(formatDcGpuFailureMatrix([])).toBe("");
    });
    it("groups attempts by DC and shows GPU + error per row", () => {
        const out = formatDcGpuFailureMatrix([
            { dc: "US-TX-3", gpu: "RTX 4090", error: "no stock" },
            { dc: "US-TX-3", gpu: "A40", error: "could not find any pods" },
            { dc: "AP-JP-1", gpu: "RTX 4090", error: "500 server error" },
        ]);
        expect(out).toContain("[US-TX-3]");
        expect(out).toContain("[AP-JP-1]");
        expect(out).toContain("RTX 4090");
        expect(out).toContain("no stock");
        // US-TX-3 group should appear before AP-JP-1 (insertion order)
        expect(out.indexOf("[US-TX-3]")).toBeLessThan(out.indexOf("[AP-JP-1]"));
    });
    it("truncates very long error messages", () => {
        const longErr = "x".repeat(500);
        const out = formatDcGpuFailureMatrix([{ dc: "EU-RO-1", gpu: "A100", error: longErr }]);
        expect(out).toContain("...");
        expect(out.length).toBeLessThan(500);
    });
});
// ── save_pod_metadata helpers ──
describe("sanitizePodName", () => {
    it("strips filesystem-unsafe characters", () => {
        expect(sanitizePodName("piu/v2:t14*test")).toBe("piu-v2-t14-test");
        expect(sanitizePodName('a"b<c>d|e?')).toBe("a-b-c-d-e");
    });
    it("collapses whitespace and dashes", () => {
        expect(sanitizePodName("  foo   bar  ")).toBe("foo-bar");
        expect(sanitizePodName("foo---bar")).toBe("foo-bar");
    });
    it("trims leading/trailing dashes", () => {
        expect(sanitizePodName("---foo---")).toBe("foo");
    });
    it("falls back to 'unnamed-pod' for empty input", () => {
        expect(sanitizePodName("")).toBe("unnamed-pod");
        expect(sanitizePodName("   ")).toBe("unnamed-pod");
        expect(sanitizePodName("///")).toBe("unnamed-pod");
    });
    it("truncates very long names to 80 chars", () => {
        const longName = "a".repeat(200);
        const out = sanitizePodName(longName);
        expect(out.length).toBe(80);
    });
});
describe("isoToDateStamp", () => {
    it("extracts YYYY-MM-DD from a valid ISO timestamp", () => {
        expect(isoToDateStamp("2026-04-07T08:06:39Z")).toBe("2026-04-07");
        expect(isoToDateStamp("2025-12-31T23:59:59.999Z")).toBe("2025-12-31");
    });
    it("falls back to provided 'now' for missing or invalid input", () => {
        const now = new Date("2026-04-07T12:00:00Z");
        expect(isoToDateStamp(undefined, now)).toBe("2026-04-07");
        expect(isoToDateStamp("not-a-date", now)).toBe("2026-04-07");
    });
});
describe("buildPodMetadataPath", () => {
    it("uses default base path '.omc/pods' and .yaml extension", () => {
        const out = buildPodMetadataPath({ name: "piu-v2-t14", created_at: "2026-04-07T08:00:00Z" });
        expect(out).toBe(".omc/pods/2026-04-07_piu-v2-t14.yaml");
    });
    it("respects custom base path", () => {
        const out = buildPodMetadataPath({ name: "test", created_at: "2026-04-07T00:00:00Z" }, ".runpod/pods");
        expect(out).toBe(".runpod/pods/2026-04-07_test.yaml");
    });
    it("sanitizes unsafe pod names in the filename", () => {
        const out = buildPodMetadataPath({ name: "bad/name:1*", created_at: "2026-04-07T00:00:00Z" });
        expect(out).toBe(".omc/pods/2026-04-07_bad-name-1.yaml");
    });
    it("falls back to 'unnamed-pod' when name is missing", () => {
        const out = buildPodMetadataPath({ created_at: "2026-04-07T00:00:00Z" });
        expect(out).toBe(".omc/pods/2026-04-07_unnamed-pod.yaml");
    });
});
describe("toYaml", () => {
    it("emits scalars correctly", () => {
        expect(toYaml(null)).toBe("null\n");
        expect(toYaml(true)).toBe("true\n");
        expect(toYaml(42)).toBe("42\n");
        expect(toYaml("hello")).toBe("hello\n");
    });
    it("quotes strings that look like numbers or booleans", () => {
        expect(toYaml("123").trim()).toBe('"123"');
        expect(toYaml("true").trim()).toBe('"true"');
        expect(toYaml("null").trim()).toBe('"null"');
    });
    it("quotes strings with special chars", () => {
        expect(toYaml("a: b").trim()).toBe('"a: b"');
        expect(toYaml("{inline}").trim()).toBe('"{inline}"');
    });
    it("emits a flat object", () => {
        const out = toYaml({ pod_id: "abc", gpu_count: 1, deleted_at: null });
        expect(out).toBe("pod_id: abc\ngpu_count: 1\ndeleted_at: null\n");
    });
    it("emits a nested object with indentation", () => {
        const out = toYaml({ network_volume: { id: "v1", size_gb: 50 } });
        expect(out).toBe("network_volume:\n  id: v1\n  size_gb: 50\n");
    });
    it("emits arrays of strings", () => {
        const out = toYaml({ post_create_steps: ["pip install foo", "apt-get install bar"] });
        expect(out).toBe("post_create_steps:\n  - pip install foo\n  - apt-get install bar\n");
    });
    it("emits empty arrays and objects inline", () => {
        expect(toYaml({ incidents: [] }).trim()).toBe("incidents: []");
        expect(toYaml({ extras: {} }).trim()).toBe("extras: {}");
    });
    it("handles multiline strings as block scalars", () => {
        const out = toYaml({ note: "line1\nline2" });
        expect(out).toContain("note: |");
        expect(out).toContain("line1");
        expect(out).toContain("line2");
    });
    it("round-trips a realistic pod metadata document", () => {
        const meta = {
            pod_id: "oukau8f6ezy0kj",
            name: "piu-v2-t14",
            purpose: "T14 EMA+SWA ablation",
            created_at: "2026-04-07T08:06:39Z",
            datacenter: "EU-RO-1",
            network_volume: { id: "6bi4k0e6or", name: "piu-v2-data-eu", size_gb: 50 },
            post_create_steps: ["apt-get install -y rsync", "pip install torch"],
            incidents: [],
        };
        const out = toYaml(meta);
        // Spot-check key features
        expect(out).toContain("pod_id: oukau8f6ezy0kj");
        expect(out).toContain("network_volume:\n  id: 6bi4k0e6or");
        expect(out).toContain("post_create_steps:\n  - apt-get install -y rsync");
        expect(out).toContain("incidents: []");
    });
});
describe("buildPodMetadataStub", () => {
    it("includes all known fields and placeholder for unknowns", () => {
        const stub = buildPodMetadataStub({
            pod_id: "abc123",
            name: "test-pod",
            created_at: "2026-04-07T00:00:00Z",
            datacenter: "US-GA-1",
            gpu: "RTX 4090 (24GB)",
            gpu_count: 1,
            cost_per_hr: 0.44,
            image: "runpod/pytorch:2.4.0",
            container_disk_gb: 50,
        });
        const parsed = JSON.parse(stub);
        expect(parsed.pod_id).toBe("abc123");
        expect(parsed.purpose).toBe("<fill in: what this pod is for>");
        expect(parsed.deleted_at).toBeNull();
        expect(parsed.network_volume).toBeNull();
        expect(parsed.ssh).toBeNull();
        expect(parsed.post_create_steps).toEqual([]);
        expect(parsed.incidents).toEqual([]);
        expect(parsed.gpu).toBe("RTX 4090 (24GB)");
        expect(parsed.datacenter).toBe("US-GA-1");
    });
    it("includes network_volume when provided", () => {
        const stub = buildPodMetadataStub({
            pod_id: "abc",
            name: "test",
            created_at: "2026-04-07T00:00:00Z",
            network_volume: { id: "v1", name: "data", size_gb: 50, datacenter: "EU-RO-1" },
        });
        const parsed = JSON.parse(stub);
        expect(parsed.network_volume.id).toBe("v1");
        expect(parsed.network_volume.size_gb).toBe(50);
    });
    it("defaults gpu_count to 1", () => {
        const stub = buildPodMetadataStub({
            pod_id: "abc",
            name: "test",
            created_at: "2026-04-07T00:00:00Z",
        });
        expect(JSON.parse(stub).gpu_count).toBe(1);
    });
});
// ── Patch D: upload integrity helpers ──
describe("parseDuBytes", () => {
    it("parses 'du -sb' output", () => {
        expect(parseDuBytes("123456\t/path/to/dir")).toBe(123456);
        expect(parseDuBytes("0\t/empty")).toBe(0);
    });
    it("returns null on garbage input", () => {
        expect(parseDuBytes("not a number")).toBeNull();
        expect(parseDuBytes("")).toBeNull();
    });
});
describe("parseDfAvailBytes", () => {
    it("parses 'df -B1 --output=avail' output (header on line 1, value on line 2)", () => {
        expect(parseDfAvailBytes("Avail\n5368709120")).toBe(5368709120);
        expect(parseDfAvailBytes("Available\n  1024  ")).toBe(1024);
    });
    it("returns null when header is missing or value is unparseable", () => {
        expect(parseDfAvailBytes("Avail")).toBeNull();
        expect(parseDfAvailBytes("Avail\nnope")).toBeNull();
        expect(parseDfAvailBytes("")).toBeNull();
    });
});
describe("checkFreeSpace", () => {
    it("passes when avail >= local * margin", () => {
        const r = checkFreeSpace(1_000_000, 2_000_000);
        expect(r.status).toBe("OK");
        expect(r.message).toContain("Free space OK");
    });
    it("fails with the silent-truncation warning when avail is short", () => {
        const r = checkFreeSpace(22_000_000_000, 20_000_000_000); // piu-v2 scenario: 22GB into 20GB
        expect(r.status).toBe("FREE_SPACE_LOW");
        expect(r.message).toContain("silently truncate");
        expect(r.message).toContain("verifySize=false");
    });
    it("respects custom safety margin", () => {
        // local=100, avail=105, margin=1.0 → OK; margin=1.1 → FAIL
        expect(checkFreeSpace(100, 105, 1.0).status).toBe("OK");
        expect(checkFreeSpace(100, 105, 1.1).status).toBe("FREE_SPACE_LOW");
    });
});
describe("checkSizeMatch", () => {
    it("passes on full match", () => {
        expect(checkSizeMatch(1000, 1000).status).toBe("OK");
    });
    it("passes within tolerance (95% default)", () => {
        expect(checkSizeMatch(1000, 960).status).toBe("OK");
        expect(checkSizeMatch(1000, 950).status).toBe("OK");
    });
    it("flags the silent-truncation pattern", () => {
        // piu-v2: 22GB local, but only ~5GB landed (16996 of 75264 files were 0-byte)
        const r = checkSizeMatch(22_000_000_000, 5_000_000_000);
        expect(r.status).toBe("SIZE_MISMATCH");
        expect(r.message).toContain("INCOMPLETE");
        expect(r.message).toContain("silent-truncation");
    });
    it("treats empty source as trivially OK", () => {
        expect(checkSizeMatch(0, 0).status).toBe("OK");
    });
});
describe("looksLikeSetupCommand", () => {
    it("matches common provisioning commands", () => {
        expect(looksLikeSetupCommand("apt-get install -y rsync tmux")).toBe(true);
        expect(looksLikeSetupCommand("pip install transformers peft")).toBe(true);
        expect(looksLikeSetupCommand("git clone https://github.com/foo/bar")).toBe(true);
        expect(looksLikeSetupCommand("rsync -av /src /dst")).toBe(true);
        expect(looksLikeSetupCommand("tar -xf data.tar")).toBe(true);
        expect(looksLikeSetupCommand("conda install numpy")).toBe(true);
    });
    it("does not match runtime commands", () => {
        expect(looksLikeSetupCommand("python train.py")).toBe(false);
        expect(looksLikeSetupCommand("nvidia-smi")).toBe(false);
        expect(looksLikeSetupCommand("ls /workspace")).toBe(false);
        expect(looksLikeSetupCommand("tail -f log")).toBe(false);
    });
});
describe("estimatePodCost", () => {
    it("computes uptime hours and cost", () => {
        const now = new Date("2026-04-08T12:00:00Z");
        const r = estimatePodCost(0.59, "2026-04-07T12:00:00Z", now);
        expect(r).not.toBeNull();
        expect(r.hours).toBe(24);
        expect(r.cost).toBeCloseTo(0.59 * 24, 5);
    });
    it("returns null when costPerHr is missing", () => {
        expect(estimatePodCost(undefined, "2026-04-07T12:00:00Z", new Date())).toBeNull();
    });
    it("returns null when startedAt is missing", () => {
        expect(estimatePodCost(0.5, undefined, new Date())).toBeNull();
    });
    it("returns null when startedAt is invalid", () => {
        expect(estimatePodCost(0.5, "not-a-date", new Date())).toBeNull();
    });
    it("returns null for negative durations (clock skew)", () => {
        const now = new Date("2026-04-07T00:00:00Z");
        const future = "2026-04-08T00:00:00Z";
        expect(estimatePodCost(0.5, future, now)).toBeNull();
    });
});
// ── Edge cases: deletePodWithStop timeout ──
describe("deletePodWithStop edge cases", () => {
    it("proceeds with delete even if pod never transitions from RUNNING (timeout)", async () => {
        const client = {
            getPod: vi.fn().mockResolvedValue({ id: "stuck-pod", desiredStatus: "RUNNING" }),
            stopPod: vi.fn().mockResolvedValue(undefined),
            deletePod: vi.fn().mockResolvedValue(undefined),
        };
        // Very short timeout to trigger timeout path
        const { wasRunning } = await deletePodWithStop(client, "stuck-pod", 100);
        expect(wasRunning).toBe(true);
        expect(client.stopPod).toHaveBeenCalled();
        expect(client.deletePod).toHaveBeenCalled();
    });
    it("handles delete failure gracefully", async () => {
        const client = {
            getPod: vi.fn().mockResolvedValue({ id: "pod-1", desiredStatus: "EXITED" }),
            stopPod: vi.fn(),
            deletePod: vi.fn().mockRejectedValue(new Error("API error: pod not found")),
        };
        await expect(deletePodWithStop(client, "pod-1")).rejects.toThrow("API error: pod not found");
    });
});
//# sourceMappingURL=orchestration.test.js.map