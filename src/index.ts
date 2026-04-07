#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";
import { RunPodClient, spawnAsync } from "./api.js";
import type { Pod } from "./types.js";
import { safeTool, text, errorResult } from "./tool-helpers.js";
import type { ToolResult } from "./tool-helpers.js";
import { parseNvidiaSmiOutput, calcSuggestedBatchSize, isOverprovisioned, injectPytorchEnv, summarizeTrend, getStockStatus, isInStock, getSpotPrice, getOnDemandPrice } from "./gpu-utils.js";
import { filterStalePods, selectGpuCandidates, deletePodWithStop, DEFAULT_DC_PRIORITY, formatDcGpuFailureMatrix, buildPodMetadataPath } from "./pod-ops.js";

const API_KEY = process.env.RUNPOD_API_KEY;

const SETUP_MSG =
  "RUNPOD_API_KEY is not configured. To set up:\n\n" +
  "1. Get your API key from https://www.runpod.io/console/user/settings\n" +
  "2. Add to your shell profile (~/.bashrc or ~/.zshrc):\n" +
  "   export RUNPOD_API_KEY=rp_xxxxxx\n" +
  "3. (Optional) For SSH/rsync features:\n" +
  "   export SSH_KEY_PATH=~/.ssh/id_ed25519\n" +
  "4. Restart Claude Code for changes to take effect.";

let client: RunPodClient | null = null;
if (API_KEY) {
  client = new RunPodClient({
    apiKey: API_KEY,
    restBaseUrl: "https://rest.runpod.io/v1",
    graphqlUrl: "https://api.runpod.io/graphql",
    sshKeyPath: process.env.SSH_KEY_PATH,
  });
}

function requireClient(): RunPodClient {
  if (!client) throw new Error(SETUP_MSG);
  return client;
}

const server = new McpServer({
  name: "runpod-tools",
  version: "0.2.0",
});

// ── Helpers ──

function podSummary(pod: Pod): string {
  const c = requireClient();
  const ssh = c.getSshCommandString(pod);
  return [
    `ID: ${pod.id}`,
    `Name: ${pod.name}`,
    `Status: ${pod.desiredStatus}`,
    pod.gpu ? `GPU: ${pod.gpu.displayName} x${pod.gpu.count}` : null,
    pod.publicIp ? `IP: ${pod.publicIp}` : "IP: (not yet assigned)",
    pod.portMappings?.["22"] ? `SSH Port: ${pod.portMappings["22"]}` : null,
    pod.networkVolumeId ? `Network Volume: ${pod.networkVolumeId}` : null,
    pod.costPerHr != null ? `Cost: $${pod.costPerHr}/hr` : null,
    ssh ? `SSH: ${ssh}` : null,
  ]
    .filter(Boolean)
    .join("\n");
}

function isAuthError(e: unknown): boolean {
  const msg = String((e as { message?: string })?.message ?? "");
  return /\b(401|403|unauthorized|forbidden|authentication)\b/i.test(msg);
}


// ══════════════════════════════════════════
//  TOOLS
// ══════════════════════════════════════════

// ── list_pods ──
server.tool("list_pods", "List all RunPod pods with status and SSH info", {}, safeTool(async () => {
  const pods = await requireClient().listPods();
  if (!pods.length) return text("No pods found.");
  return text(pods.map((p) => podSummary(p)).join("\n\n---\n\n"));
}));

// ── get_pod ──
server.tool(
  "get_pod",
  "Get detailed info about a specific pod",
  { podId: z.string().describe("Pod ID") },
  safeTool(async ({ podId }) => text(podSummary(await requireClient().getPod(podId))))
);

// ── create_pod ──
server.tool(
  "create_pod",
  "Create a new RunPod GPU pod. Uses REST API for on-demand, GraphQL for spot instances.",
  {
    name: z.string().describe("Pod name"),
    imageName: z.string().default("runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04").describe("Docker image"),
    gpuTypeId: z.string().describe('GPU type, e.g. "NVIDIA GeForce RTX 3090"'),
    gpuCount: z.number().default(1),
    spot: z.boolean().default(false).describe("Use spot (interruptible) instance — cheaper but can be preempted at any time"),
    bidPerGpu: z.number().optional().describe("Max bid per GPU for spot instances"),
    containerDiskInGb: z.number().default(50),
    volumeInGb: z.number().default(20),
    volumeMountPath: z.string().default("/workspace"),
    networkVolumeId: z.string().optional().describe("Attach existing network volume"),
    sshPublicKey: z.string().optional().describe("SSH public key to inject (overrides account default)"),
    ports: z.array(z.string()).default(["22/tcp"]),
    env: z.record(z.string()).optional().describe("Environment variables"),
    dockerArgs: z.string().optional(),
    cloudType: z
      .enum(["ALL", "SECURE", "COMMUNITY"])
      .default("ALL")
      .describe("Cloud type filter: ALL (default), SECURE (dedicated), or COMMUNITY (cheaper, shared)"),
    optimizePytorch: z
      .boolean()
      .default(false)
      .describe("Inject PyTorch CUDA optimization env vars (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True). Requires PyTorch >= 2.0."),
  },
  safeTool(async (args) => {
    const podEnv = injectPytorchEnv(args.env, args.optimizePytorch);

    const opts = {
      name: args.name,
      imageName: args.imageName,
      gpuTypeIds: [args.gpuTypeId],
      gpuCount: args.gpuCount,
      interruptible: args.spot,
      containerDiskInGb: args.containerDiskInGb,
      volumeInGb: args.volumeInGb,
      volumeMountPath: args.volumeMountPath,
      networkVolumeId: args.networkVolumeId,
      sshPublicKey: args.sshPublicKey,
      ports: args.ports,
      env: podEnv,
      dockerArgs: args.dockerArgs,
      cloudType: args.cloudType,
    };

    if (args.spot && args.bidPerGpu) {
      const result = await requireClient().createSpotPod({ ...opts, bidPerGpu: args.bidPerGpu });
      return text(`Spot pod created!\nID: ${result.id}\n\n## Next Steps\n→ wait_for_pod(podId: "${result.id}")`);
    }

    const pod = await requireClient().createPod(opts);
    return text(`Pod created!\n${podSummary(pod)}\n\n## Next Steps\n→ wait_for_pod(podId: "${pod.id}")`);
  })
);

// ── create_pod_auto ──
server.tool(
  "create_pod_auto",
  "Create a pod with automatic GPU selection based on stock availability. Tries GPUs in order of preference, including Low stock (worth trying). Use dryRun=true to preview GPU selection and cost estimate without creating a pod.",
  {
    name: z.string().describe("Pod name"),
    imageName: z.string().default("runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"),
    gpuPreference: z
      .array(z.string())
      .default(["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA A40", "NVIDIA RTX A5000"])
      .describe("GPU types in order of preference"),
    minVram: z.number().default(12).describe("Minimum VRAM in GB"),
    gpuCount: z.number().default(1).describe("Number of GPUs per pod"),
    spot: z.boolean().default(false).describe("Use spot (interruptible) instance — cheaper but can be preempted"),
    maxBidPerGpu: z.number().default(0.3).describe("Max spot bid per GPU"),
    containerDiskInGb: z.number().default(50),
    volumeInGb: z.number().default(20),
    sshPublicKey: z.string().optional(),
    env: z.record(z.string()).optional(),
    networkVolumeId: z.string().optional().describe("Attach existing network volume. When provided, the pod is automatically created in the volume's datacenter (dcPriority is ignored)."),
    dcPriority: z
      .array(z.string())
      .optional()
      .describe(
        "Datacenter priority list for fallback when stock is tight. Tries each DC in order with each GPU type until a pod is created. Ignored when networkVolumeId is set (NV constrains the DC). Defaults to a built-in priority based on observed RunPod stock pool sizes (largest first): US-GA-1, US-CA-2, EU-SE-1, EU-CZ-1, AP-JP-1, US-TX-3, EU-RO-1."
      ),
    optimizePytorch: z
      .boolean()
      .default(false)
      .describe("Inject PyTorch CUDA optimization env vars (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True). Requires PyTorch >= 2.0."),
    cloudType: z
      .enum(["ALL", "SECURE", "COMMUNITY"])
      .default("ALL")
      .describe("Cloud type filter: ALL (default), SECURE (dedicated), or COMMUNITY (cheaper, shared)"),
    dryRun: z
      .boolean()
      .default(false)
      .describe("Preview GPU selection and cost estimate without creating a pod. Note: GPU availability may change between dry run and actual creation."),
  },
  safeTool(async (args) => {
    const c = requireClient();
    const gpuTypes = await c.listGpuTypes();

    // Resolve datacenter affinity from network volume.
    // If NV is provided, the pod MUST run in NV's DC — dcPriority is ignored.
    let nvDataCenterId: string | undefined;
    let volumeNote = "";
    if (args.networkVolumeId) {
      const vol = await c.getNetworkVolume(args.networkVolumeId);
      if (!vol) return text(`Network volume ${args.networkVolumeId} not found.`);
      nvDataCenterId = vol.dataCenterId;
      volumeNote = `\nNetwork Volume: ${vol.name} (${vol.id}) in ${vol.dataCenterId}`;
    }

    // DC iteration list:
    //   - NV present  → only NV's DC (single)
    //   - NV absent   → user-supplied dcPriority OR built-in default
    const dcsToTry: string[] = nvDataCenterId
      ? [nvDataCenterId]
      : (args.dcPriority && args.dcPriority.length > 0 ? args.dcPriority : DEFAULT_DC_PRIORITY);

    const { candidates, errors } = selectGpuCandidates(gpuTypes, {
      gpuPreference: args.gpuPreference,
      minVram: args.minVram,
      gpuCount: args.gpuCount,
      spot: args.spot,
      maxBidPerGpu: args.maxBidPerGpu,
    });

    // Dry run: return preview from the first viable candidate WITHOUT iterating DCs.
    // (We can't probe per-DC stock without actually creating, so dry run is best-effort.)
    if (args.dryRun && candidates.length > 0) {
      const { gpu, stock, ondemandPrice, bidPrice, minBid, overprovisionWarning } = candidates[0];
      const priceInfo = args.spot && bidPrice
        ? `Spot bid: $${bidPrice}/hr (min: $${minBid}/hr)`
        : `On-demand: $${ondemandPrice}/hr`;
      const monthlyCost = (args.spot && bidPrice ? bidPrice : ondemandPrice) * 24 * 30;
      const dcNote = nvDataCenterId
        ? `\nDatacenter: ${nvDataCenterId} (forced by network volume)`
        : `\nDC fallback order: ${dcsToTry.join(" → ")}`;
      return text(
        `## Dry Run — Preview Only (no pod created)\n\n` +
          `GPU: ${gpu.displayName} (${gpu.memoryInGb}GB VRAM, stock: ${stock ?? "unknown"})\n` +
          `${priceInfo}\n` +
          `Estimated monthly: $${monthlyCost.toFixed(0)}\n` +
          `Image: ${args.imageName}\n` +
          `GPU count: ${args.gpuCount}${dcNote}${overprovisionWarning}${volumeNote}\n\n` +
          `Note: per-DC stock cannot be probed without creating a pod. Real run will iterate DCs in the order shown.\n\n` +
          `## Next Steps\n→ create_pod_auto with same parameters and dryRun: false`
      );
    }

    // DC × GPU fallback loop. Outer = DC priority, inner = GPU preference.
    // Per-attempt failures are recorded into a matrix for diagnostic output.
    const failureMatrix: Array<{ dc: string; gpu: string; error: string }> = [];

    for (const dc of dcsToTry) {
      for (const { gpu, gpuId, stock, ondemandPrice, bidPrice, overprovisionWarning } of candidates) {
        try {
          const podEnv = injectPytorchEnv(args.env, args.optimizePytorch);

          const opts = {
            name: args.name,
            imageName: args.imageName,
            gpuTypeIds: [gpuId],
            gpuCount: args.gpuCount,
            interruptible: args.spot,
            containerDiskInGb: args.containerDiskInGb,
            volumeInGb: args.volumeInGb,
            volumeMountPath: "/workspace",
            sshPublicKey: args.sshPublicKey,
            ports: ["22/tcp"] as string[],
            env: podEnv,
            networkVolumeId: args.networkVolumeId,
            dataCenterIds: [dc],
            cloudType: args.cloudType,
          };

          if (args.spot && bidPrice) {
            const result = await c.createSpotPod({ ...opts, bidPerGpu: bidPrice });
            return text(
              `Auto-selected: ${gpu.displayName} in ${dc} (stock: ${stock ?? "unknown"})\n` +
                `Spot bid: $${bidPrice}/hr\n` +
                `Pod ID: ${result.id}${overprovisionWarning}${volumeNote}\n\n` +
                `## Next Steps\n→ wait_for_pod(podId: "${result.id}")`
            );
          }

          const pod = await c.createPod(opts);
          const dcLabel = nvDataCenterId ? dc : `${dc} (price: $${ondemandPrice}/hr)`;
          return text(
            `Auto-selected: ${gpu.displayName} in ${dcLabel} (stock: ${stock ?? "unknown"})${overprovisionWarning}${volumeNote}\n${podSummary(pod)}\n\n` +
              `## Next Steps\n→ wait_for_pod(podId: "${pod.id}")`
          );
        } catch (e) {
          if (isAuthError(e)) return errorResult(e);
          failureMatrix.push({ dc, gpu: gpu.displayName, error: (e as Error).message });
          continue;
        }
      }
    }

    // Exhausted: build diagnostic output.
    const available = gpuTypes
      .filter((g) => g.memoryInGb >= args.minVram && getStockStatus(g) !== "Out of Stock")
      .sort((a, b) => {
        const ap = getSpotPrice(a) ?? Infinity;
        const bp = getSpotPrice(b) ?? Infinity;
        return ap - bp;
      })
      .slice(0, 10);

    const matrixText = formatDcGpuFailureMatrix(failureMatrix);
    const matrixBlock = matrixText
      ? `\n\nFailure matrix (${failureMatrix.length} attempts across ${dcsToTry.length} DC × ${candidates.length} GPU):\n${matrixText}`
      : "";
    const selectionErrors = errors.length ? `\n\nSelection errors:\n${errors.join("\n")}` : "";
    const nvHint = nvDataCenterId
      ? `\n\n⚠ Network volume ${args.networkVolumeId} constrains pods to ${nvDataCenterId}.${volumeNote}\n` +
        `If this DC is dry, options:\n` +
        `  1. Wait and retry — RunPod stock fluctuates.\n` +
        `  2. Create a new network volume in a different DC (create_network_volume), upload data again, and retry.\n` +
        `  3. Run without networkVolumeId to use dcPriority fallback (${DEFAULT_DC_PRIORITY.slice(0, 3).join(", ")}, ...).`
      : `\n\nDC fallback order tried: ${dcsToTry.join(" → ")}\n` +
        `All combinations exhausted. Try again later or override dcPriority with a different list.`;

    return text(
      `No pod could be created.${nvHint}\n\nCheapest alternatives (global stock — NOT guaranteed in any specific DC):\n\n` +
        available.map((g) => {
          const price = getSpotPrice(g);
          const st = getStockStatus(g);
          return `${g.displayName} (${g.memoryInGb}GB) - ${price != null ? `$${price}/hr` : "n/a"} [${st}]`;
        }).join("\n") +
        matrixBlock +
        selectionErrors
    );
  })
);

// ── stop_pod ──
server.tool(
  "stop_pod",
  "Stop a running pod (preserves volume data, stops billing for compute)",
  { podId: z.string() },
  safeTool(async ({ podId }) => {
    await requireClient().stopPod(podId);
    return text(`Pod ${podId} stop requested.`);
  })
);

// ── start_pod ──
server.tool(
  "start_pod",
  "Start a stopped pod",
  { podId: z.string() },
  safeTool(async ({ podId }) => {
    await requireClient().startPod(podId);
    return text(`Pod ${podId} start requested.\n\n## Next Steps\n→ wait_for_pod(podId: "${podId}")`);
  })
);

// ── restart_pod ──
server.tool(
  "restart_pod",
  "Restart a running pod",
  { podId: z.string() },
  safeTool(async ({ podId }) => {
    await requireClient().restartPod(podId);
    return text(`Pod ${podId} restart requested.`);
  })
);

// ── delete_pod ──
server.tool(
  "delete_pod",
  "Permanently delete a pod (auto-stops if running). WARNING: destroys all data not on network volumes.",
  { podId: z.string() },
  safeTool(async ({ podId }) => {
    const { wasRunning } = await deletePodWithStop(requireClient(), podId);
    return text(`Pod ${podId} deleted.${wasRunning ? ` (was running → auto-stopped first)` : ""}`);
  })
);

// ── cleanup_stale_pods ──
server.tool(
  "cleanup_stale_pods",
  "Find and delete EXITED pods that have been idle longer than graceHours. Pods with 'keep' or 'persist' in their name are skipped. Use dryRun=true (default) to preview what would be deleted.",
  {
    graceHours: z.number().default(2).describe("Hours since last status change before a pod is considered stale"),
    dryRun: z.boolean().default(true).describe("If true, only list stale pods without deleting"),
  },
  safeTool(async ({ graceHours, dryRun }) => {
    const c = requireClient();
    const pods = await c.listPods();
    const { stale, skipped } = filterStalePods(pods, graceHours);

    if (!stale.length) {
      return text(`No stale pods found.\n\nSkipped: ${skipped.length} pod(s)${skipped.length ? "\n" + skipped.map(s => `  - ${s.pod.name}: ${s.reason}`).join("\n") : ""}`);
    }

    if (dryRun) {
      const lines = stale.map(s =>
        `  - ${s.pod.name} (${s.pod.id}) — idle ${s.idleHours}h, ${s.pod.gpu?.displayName ?? "unknown GPU"}, $${s.pod.costPerHr ?? "?"}/hr`
      );
      return text(`[DRY RUN] Would delete ${stale.length} stale pod(s):\n${lines.join("\n")}\n\nRe-run with dryRun=false to delete.`);
    }

    const deleted: string[] = [];
    const failed: string[] = [];
    for (const s of stale) {
      try {
        await c.deletePod(s.pod.id);
        deleted.push(`${s.pod.name} (idle ${s.idleHours}h)`);
      } catch (e) {
        failed.push(`${s.pod.name} (idle ${s.idleHours}h): ${(e as Error).message}`);
      }
    }

    return text(`Deleted ${deleted.length} stale pod(s):\n${deleted.map(d => `  - ${d}`).join("\n")}${failed.length ? `\n\nFailed: ${failed.join(", ")}` : ""}`);
  })
);

// ── save_pod_metadata ──
//
// Persists a pod's full provisioning recipe to disk so debugging is possible
// after the pod is deleted. Without this, the DC, image tag, installed packages,
// data layout, launch command, and incident history vanish with the pod.
//
// The file lives in the user's project repo (NOT in runpod-mcp), default path
// `.runpod/pods/{YYYY-MM-DD}_{podName}.json`. The caller is expected to git
// commit it. See CLAUDE.md "Pod Metadata Persistence" for the workflow.
const podMetadataSchema = z
  .object({
    pod_id: z.string().describe("RunPod pod ID"),
    name: z.string().describe("Pod name (used in the filename)"),
    purpose: z.string().optional().describe("One-line description of what this pod is for"),
    created_at: z.string().optional().describe("ISO timestamp when the pod was created (drives the filename date stamp; defaults to today)"),
    deleted_at: z.string().nullable().optional().describe("ISO timestamp when the pod was deleted (null while still alive)"),
    datacenter: z.string().optional(),
    gpu: z.string().optional().describe("e.g. 'NVIDIA GeForce RTX 4090 (24GB)'"),
    gpu_count: z.number().optional(),
    cost_per_hr: z.number().optional(),
    cost_actual_usd: z.number().optional().describe("Final cost after deletion (set when closing the record)"),
    container_disk_gb: z.number().optional(),
    image: z.string().optional().describe("Docker image tag"),
    network_volume: z
      .object({
        id: z.string(),
        name: z.string(),
        size_gb: z.number(),
        datacenter: z.string().optional(),
      })
      .nullable()
      .optional(),
    ssh: z.object({ host: z.string(), port: z.number() }).optional(),
    post_create_steps: z.array(z.string()).optional().describe("Shell commands run after pod creation (apt-get, pip install, etc.)"),
    data: z
      .object({
        source: z.string().optional(),
        dest: z.string().optional(),
        transfer_method: z.string().optional(),
        size_gb: z.number().optional(),
      })
      .optional(),
    code: z.object({ source: z.string().optional(), commit: z.string().optional() }).optional(),
    execution: z
      .object({
        script: z.string().optional(),
        log: z.string().optional(),
        output_dir: z.string().optional(),
        expected_runs: z.number().optional(),
        expected_gpu_hours: z.number().optional(),
        expected_cost_usd: z.number().optional(),
      })
      .optional(),
    monitor: z.object({ cron_id: z.string().optional() }).optional(),
    incidents: z.array(z.string()).optional().describe("Free-form incident log — append entries as they happen, then re-save"),
  })
  .passthrough();

server.tool(
  "save_pod_metadata",
  "Persist a pod's provisioning recipe to disk so debugging is possible after the pod is deleted. Writes JSON to `{path}/{YYYY-MM-DD}_{podName}.json`. Default path: `.runpod/pods/` relative to the caller's CWD. The file is meant to be git-committed in the user's project repo. Call after pod setup completes (post-create installs done, training launched), again on incidents (append to incidents[] and re-save), and once more before deletion (set deleted_at + cost_actual_usd). See CLAUDE.md 'Pod Metadata Persistence' for the full workflow.",
  {
    metadata: podMetadataSchema,
    path: z
      .string()
      .optional()
      .describe("Base directory for the metadata file (default: '.runpod/pods'). Relative paths resolve against the current working directory. Will be created if it does not exist."),
  },
  safeTool(async ({ metadata, path }) => {
    const basePath = path ?? ".runpod/pods";
    const relPath = buildPodMetadataPath(metadata, basePath);
    const absPath = isAbsolute(relPath) ? relPath : resolve(process.cwd(), relPath);

    try {
      await mkdir(dirname(absPath), { recursive: true });
      await writeFile(absPath, JSON.stringify(metadata, null, 2) + "\n", "utf8");
    } catch (e) {
      return text(`Failed to save pod metadata to ${absPath}: ${(e as Error).message}`);
    }

    const incidentCount = metadata.incidents?.length ?? 0;
    const stepCount = metadata.post_create_steps?.length ?? 0;
    const closed = metadata.deleted_at ? " [CLOSED]" : "";
    return text(
      `Pod metadata saved${closed}\n` +
        `Path: ${absPath}\n` +
        `Pod: ${metadata.name} (${metadata.pod_id})\n` +
        `Steps recorded: ${stepCount} | Incidents: ${incidentCount}\n\n` +
        `## Next Steps\n→ git add ${relPath} && git commit -m "docs: record pod ${metadata.name}"`
    );
  })
);

// ── wait_for_pod ──
server.tool(
  "wait_for_pod",
  "Poll until a pod is RUNNING with a public IP and SSH port available (includes TCP probe). Returns SSH command when ready. Always call after create_pod/create_pod_auto before any SSH operations.",
  {
    podId: z.string(),
    timeoutSeconds: z.number().default(300).describe("Max wait time in seconds"),
    intervalSeconds: z.number().default(10).describe("Poll interval in seconds"),
  },
  safeTool(async ({ podId, timeoutSeconds, intervalSeconds }, extra?: any) => {
    const onProgress = extra?.sendNotification
      ? (message: string) => {
          extra.sendNotification({
            method: "notifications/message",
            params: { level: "info", logger: "wait_for_pod", data: message },
          }).catch(() => {});
        }
      : undefined;
    const pod = await requireClient().waitForPod(podId, timeoutSeconds * 1000, intervalSeconds * 1000, onProgress);
    return text(`Pod is ready!\n\n${podSummary(pod)}`);
  })
);

// ── list_gpu_types ──
server.tool(
  "list_gpu_types",
  "List available GPU types with pricing and stock status (via GraphQL)",
  {
    minVram: z.number().default(0).describe("Filter by minimum VRAM in GB"),
    inStockOnly: z.boolean().default(false).describe("Only show GPUs with High/Medium stock"),
  },
  safeTool(async ({ minVram, inStockOnly }) => {
    let gpus = await requireClient().listGpuTypes();
    if (minVram > 0) gpus = gpus.filter((g) => g.memoryInGb >= minVram);
    if (inStockOnly) gpus = gpus.filter((g) => {
      const status = getStockStatus(g);
      return status === "High" || status === "Medium" || status === "available";
    });
    gpus.sort((a, b) => {
      const aPrice = getSpotPrice(a) ?? Infinity;
      const bPrice = getSpotPrice(b) ?? Infinity;
      return aPrice - bPrice;
    });

    if (!gpus.length) return text("No GPUs match the criteria.");

    const header = "GPU Type | VRAM | Spot Price | On-Demand | Stock";
    const sep = "---|---|---|---|---";
    const rows = gpus.map((g) => {
      const spot = getSpotPrice(g);
      const ondemand = getOnDemandPrice(g);
      const stock = getStockStatus(g);
      return `${g.displayName} | ${g.memoryInGb}GB | ${spot != null ? `$${spot}/hr` : "n/a"} | ${ondemand != null ? `$${ondemand}/hr` : "n/a"} | ${stock}`;
    });
    return text([header, sep, ...rows].join("\n"));
  })
);

// ── get_ssh_command ──
server.tool(
  "get_ssh_command",
  "Get the SSH command for connecting to a running pod",
  { podId: z.string() },
  safeTool(async ({ podId }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    const cmd = c.getSshCommandString(pod);
    if (!cmd) return text("Pod is not ready (no public IP or SSH port). Try wait_for_pod first.");
    return text(cmd);
  })
);

// ── execute_ssh_command (uses async spawn with args array — no shell injection) ──
server.tool(
  "execute_ssh_command",
  "Execute a command on a running pod via SSH. Returns stdout/stderr. Requires wait_for_pod first; background long jobs with nohup.",
  {
    podId: z.string(),
    command: z.string().describe("Shell command to execute on the pod"),
    timeoutSeconds: z.number().default(120).describe("Command timeout"),
  },
  safeTool(async ({ podId, command, timeoutSeconds }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    const sshArgs = c.getSshArgs(pod);
    if (!sshArgs) return text("Pod is not ready for SSH.");

    const result = await spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", command], {
      timeout: timeoutSeconds * 1000,
    });

    if (result.error) return text(`SSH error: ${result.error.message}`);
    if (result.status !== 0) {
      return text(`Exit code: ${result.status}\n\nStderr:\n${result.stderr}\n\nStdout:\n${result.stdout}`);
    }
    return text(result.stdout || "(no output)");
  })
);

// ── upload_files (uses async spawn with args array — no shell injection) ──
server.tool(
  "upload_files",
  "Upload local files/directories to a pod via rsync",
  {
    podId: z.string(),
    localPath: z.string().describe("Local file or directory path"),
    remotePath: z.string().default("/workspace").describe("Destination path on pod"),
    dryRun: z.boolean().default(false).describe("Show command without executing"),
  },
  safeTool(async ({ podId, localPath, remotePath, dryRun }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    const args = c.getRsyncArgs(pod, localPath, remotePath, "upload");
    if (!args) return text("Pod is not ready for file transfer.");
    if (dryRun) return text(`Command (dry run):\n${args.join(" ")}`);

    const result = await spawnAsync(args[0], args.slice(1), { timeout: 600_000 });

    if (result.error) return text(`Upload error: ${result.error.message}`);
    if (result.status !== 0) return text(`Upload failed (exit ${result.status}):\n${result.stderr}`);
    return text(`Upload complete.\n\n${result.stdout}`);
  })
);

// ── download_files (uses async spawn with args array — no shell injection) ──
server.tool(
  "download_files",
  "Download files from a pod to local filesystem via rsync",
  {
    podId: z.string(),
    remotePath: z.string().describe("Path on pod to download"),
    localPath: z.string().describe("Local destination path"),
    dryRun: z.boolean().default(false),
  },
  safeTool(async ({ podId, remotePath, localPath, dryRun }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    const args = c.getRsyncArgs(pod, localPath, remotePath, "download");
    if (!args) return text("Pod is not ready for file transfer.");
    if (dryRun) return text(`Command (dry run):\n${args.join(" ")}`);

    const result = await spawnAsync(args[0], args.slice(1), { timeout: 600_000 });

    if (result.error) return text(`Download error: ${result.error.message}`);
    if (result.status !== 0) return text(`Download failed (exit ${result.status}):\n${result.stderr}`);
    return text(`Download complete.\n\n${result.stdout}`);
  })
);

// ── gpu_health_check ──
server.tool(
  "gpu_health_check",
  "Check GPU memory utilization on a running pod via nvidia-smi. Returns per-GPU metrics with utilization labels and optional batch size recommendation. Best called 1-2 min after training starts to measure actual GPU utilization.",
  {
    podId: z.string().describe("Pod ID"),
    perSampleMb: z
      .number()
      .optional()
      .describe(
        "Memory per sample in MiB (measure with: peak_memory / batch_size after a few batches). When provided, calculates recommended batch size for ~82% VRAM utilization."
      ),
    timeoutSeconds: z.number().default(30).describe("SSH timeout"),
  },
  safeTool(async ({ podId, perSampleMb, timeoutSeconds }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    const sshArgs = c.getSshArgs(pod);
    if (!sshArgs) return text("Pod is not ready for SSH. Use wait_for_pod first.");

    // Query per-GPU metrics and per-process memory in parallel
    const gpuCmd =
      "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits 2>&1";
    const procCmd =
      "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true";

    const [gpuResult, procResult] = await Promise.all([
      spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", gpuCmd], { timeout: timeoutSeconds * 1000 }),
      spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", procCmd], { timeout: timeoutSeconds * 1000 }),
    ]);

    if (gpuResult.error) return text(`SSH error: ${gpuResult.error.message}`);

    const output = (gpuResult.stdout ?? "").trim();
    if (!output || output.includes("command not found") || output.includes("not found")) {
      return text(
        "nvidia-smi not found on this pod. GPU health check requires an NVIDIA GPU with drivers installed.\n" +
          "Most RunPod GPU images include nvidia-smi by default."
      );
    }
    if (gpuResult.status !== 0) {
      return text(`nvidia-smi failed (exit ${gpuResult.status}):\n${output}\n${gpuResult.stderr ?? ""}`);
    }

    // Parse GPU metrics using extracted utility
    const gpus = parseNvidiaSmiOutput(output);

    if (!gpus.length) return text("No GPU data returned from nvidia-smi.");

    // Format output
    const sections: string[] = ["## GPU Health Check\n"];

    for (const gpu of gpus) {
      sections.push(
        `### GPU ${gpu.index}: ${gpu.name}`,
        `- VRAM: ${gpu.usedMb} / ${gpu.totalMb} MiB (${gpu.usedPct}%) — **${gpu.label}**`,
        `- GPU Utilization: ${gpu.gpuUtil}%`,
        `- Memory Bandwidth: ${gpu.memUtil}%`,
        `- Temperature: ${gpu.temp}°C`,
        ""
      );
    }

    // Aggregate for multi-GPU
    if (gpus.length > 1) {
      const totalVram = gpus.reduce((s, g) => s + g.totalMb, 0);
      const usedVram = gpus.reduce((s, g) => s + g.usedMb, 0);
      const avgUtil = Math.round(gpus.reduce((s, g) => s + g.gpuUtil, 0) / gpus.length);
      sections.push(
        `### Aggregate (${gpus.length} GPUs)`,
        `- Total VRAM: ${usedVram} / ${totalVram} MiB (${Math.round((usedVram / totalVram) * 100)}%)`,
        `- Avg GPU Utilization: ${avgUtil}%`,
        ""
      );
    }

    // Per-process info
    const procOutput = (procResult?.stdout ?? "").trim();
    if (procOutput) {
      const procLines = procOutput.split("\n").filter((l) => l.trim() && !l.includes("No running"));
      if (procLines.length > 0) {
        sections.push("### Active GPU Processes");
        for (const pl of procLines) {
          const [, pid, pname, pmem] = pl.split(",").map((s) => s.trim());
          sections.push(`- PID ${pid}: ${pname} (${pmem} MiB)`);
        }
        sections.push("");
      }
    }

    // Recommendations
    const recs: string[] = [];
    const primaryGpu = gpus[0];

    if (primaryGpu.label === "IDLE" || primaryGpu.label === "UNDERUTILIZED") {
      recs.push(
        `- **Low VRAM usage (${primaryGpu.usedPct}%)**: You are using ${primaryGpu.usedMb} MiB of ${primaryGpu.totalMb} MiB.`,
        "  Consider increasing batch size, using larger model variants, or switching to a cheaper GPU."
      );
    }
    if (primaryGpu.gpuUtil < 30 && primaryGpu.usedPct > 10) {
      recs.push(
        `- **Low GPU compute utilization (${primaryGpu.gpuUtil}%)**: GPU may be waiting for data.`,
        "  Check data loading pipeline: increase num_workers, enable pin_memory, or use prefetching."
      );
    }
    if (primaryGpu.label === "NEAR_OOM") {
      recs.push(
        `- **Near OOM (${primaryGpu.usedPct}%)**: Consider reducing batch size, enabling gradient checkpointing, or using mixed precision (fp16/bf16).`
      );
    }

    // Batch size advisor (e004)
    if (perSampleMb != null && perSampleMb > 0) {
      const suggestedBs = calcSuggestedBatchSize(primaryGpu.totalMb, perSampleMb);
      const currentEstBs = primaryGpu.usedMb > 0 ? Math.round(primaryGpu.usedMb / perSampleMb) : null;

      sections.push("### Batch Size Advisor");
      sections.push(`- Per-sample memory: ${perSampleMb} MiB`);
      sections.push(`- Target VRAM utilization: 82%`);
      if (suggestedBs <= 0) {
        sections.push("- **Per-sample memory exceeds available VRAM target.** Reduce sequence length, enable gradient checkpointing, or use mixed precision.");
      } else {
        sections.push(`- **Recommended batch size: ${suggestedBs}**`);
      }
      if (currentEstBs != null && currentEstBs > 0) {
        const ratio = suggestedBs / currentEstBs;
        sections.push(`- Current estimated batch size: ~${currentEstBs} (${ratio > 1 ? `${ratio.toFixed(1)}x increase possible` : "already near optimal"})`);
      }
      if (gpus.length > 1) {
        sections.push(`- **Multi-GPU note**: Recommendation is per-GPU. For DataParallel, effective batch = ${suggestedBs} x ${gpus.length} = ${suggestedBs * gpus.length}.`);
      }
      sections.push(
        "",
        "> **Note**: For transformer/attention models, memory scales O(n^2) with sequence length.",
        "> Increase batch size gradually and monitor for OOM errors."
      );
    }

    if (recs.length > 0) {
      sections.push("### Recommendations", ...recs);
    }

    // Cost context
    if (pod.costPerHr != null) {
      sections.push("", `**Current cost**: $${pod.costPerHr}/hr`);
      if (primaryGpu.label === "IDLE") {
        sections.push(`\n## Next Steps\n→ gpu_cost_compare(podId: "${podId}")`);
      }
    }

    return text(sections.join("\n"));
  })
);

// ── gpu_cost_compare ──
server.tool(
  "gpu_cost_compare",
  "Compare current pod GPU cost against catalog alternatives. Finds cheaper GPUs with similar or sufficient VRAM. Call after gpu_health_check reveals underutilization to find cheaper alternatives.",
  {
    podId: z.string().describe("Pod ID to compare"),
    requiredVramGb: z.number().optional().describe("Minimum VRAM needed in GB (defaults to pod's current GPU VRAM)"),
  },
  safeTool(async ({ podId, requiredVramGb }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);

    if (!pod.gpu) return text("This pod has no GPU information. Cannot compare costs.");

    const currentCost = pod.costPerHr ?? pod.adjustedCostPerHr;
    const gpuTypes = await c.listGpuTypes();
    const currentGpu = gpuTypes.find((g) => g.id === pod.gpu!.id || g.displayName === pod.gpu!.displayName);
    const currentVram = currentGpu?.memoryInGb ?? 0;
    const minVram = requiredVramGb ?? (currentVram > 0 ? currentVram : null);

    if (minVram == null) {
      return text(
        `Could not determine VRAM for ${pod.gpu.displayName} from catalog.\n` +
          `Please specify requiredVramGb explicitly.`
      );
    }

    if (!currentCost && !currentGpu) {
      return text(`Could not determine current GPU cost for ${pod.gpu.displayName}.`);
    }

    // Determine pricing mode: compare like-for-like
    const isSpot = pod.adjustedCostPerHr != null && pod.adjustedCostPerHr !== pod.costPerHr;
    const currentPrice = currentCost ?? (currentGpu ? getSpotPrice(currentGpu) : null) ?? 0;

    // Find alternatives: same or higher VRAM, in stock
    const alternatives = gpuTypes
      .filter((g) => {
        if (g.id === currentGpu?.id) return false;
        if (g.memoryInGb < minVram) return false;
        if (getStockStatus(g) === "Out of Stock") return false;
        return true;
      })
      .map((g) => {
        const spotPrice = getSpotPrice(g);
        const ondemandPrice = getOnDemandPrice(g);
        // Compare like-for-like: use spot price if current pod is spot, else on-demand
        const comparePrice = isSpot ? (spotPrice ?? ondemandPrice) : (ondemandPrice ?? spotPrice);
        return { ...g, spotPrice, ondemandPrice, comparePrice: comparePrice ?? Infinity };
      })
      .filter((g) => g.comparePrice < Infinity)
      .sort((a, b) => a.comparePrice - b.comparePrice)
      .slice(0, 10);

    const pricingLabel = isSpot ? "spot" : "on-demand";
    const sections: string[] = [
      "## GPU Cost Comparison\n",
      `### Current GPU: ${pod.gpu.displayName} x${pod.gpu.count}`,
      `- VRAM: ${currentVram > 0 ? `${currentVram}GB` : "unknown"}`,
      `- Cost: $${currentPrice}/hr ${pricingLabel} ($${(currentPrice * 24 * 30).toFixed(0)}/month estimated)`,
      "",
    ];

    const cheaper = alternatives.filter((a) => a.comparePrice < currentPrice);
    const similar = alternatives.filter((a) => a.comparePrice >= currentPrice);

    if (cheaper.length === 0) {
      sections.push(`**Already on the cheapest available GPU for your VRAM requirement (${pricingLabel} pricing).**\n`);
    } else {
      sections.push(`### Cheaper Alternatives (${pricingLabel} pricing)\n`);
      sections.push("| GPU | VRAM | Spot | On-Demand | Stock | Monthly Savings |");
      sections.push("|-----|------|------|-----------|-------|-----------------|");
      for (const g of cheaper) {
        const savings = (currentPrice - g.comparePrice) * 24 * 30;
        const stock = getStockStatus(g);
        sections.push(
          `| ${g.displayName} | ${g.memoryInGb}GB | ${g.spotPrice != null ? `$${g.spotPrice}/hr` : "n/a"} | ${g.ondemandPrice != null ? `$${g.ondemandPrice}/hr` : "n/a"} | ${stock} | ~$${savings.toFixed(0)} |`
        );
      }
      sections.push("");
    }

    if (similar.length > 0 && cheaper.length < 5) {
      sections.push("### Other Options (same or higher price)\n");
      for (const g of similar.slice(0, 5)) {
        const stock = getStockStatus(g);
        sections.push(`- ${g.displayName} (${g.memoryInGb}GB) - $${g.comparePrice}/hr [${stock}]`);
      }
    }

    return text(sections.join("\n"));
  })
);

// ── gpu_sample_burst ──
server.tool(
  "gpu_sample_burst",
  "Take multiple rapid GPU utilization snapshots (3-5 samples, 3-5s apart) to detect trends. Returns per-sample metrics plus a trend verdict: STABLE_OPTIMAL, IMPROVING, DEGRADING, CONSISTENTLY_IDLE, or VOLATILE. Use during training to verify GPU stays utilized over time.",
  {
    podId: z.string().describe("Pod ID"),
    samples: z.number().min(2).max(10).default(5).describe("Number of samples to take"),
    intervalSeconds: z.number().min(2).max(10).default(3).describe("Seconds between samples"),
    timeoutSeconds: z.number().default(120).describe("Total SSH timeout"),
  },
  safeTool(async ({ podId, samples, intervalSeconds, timeoutSeconds }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    const sshArgs = c.getSshArgs(pod);
    if (!sshArgs) return text("Pod is not ready for SSH. Use wait_for_pod first.");

    // Build a single SSH command that takes N samples with sleep between them
    const smiCmd = "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits";
    const loopParts: string[] = [];
    for (let i = 0; i < samples; i++) {
      if (i > 0) loopParts.push(`sleep ${intervalSeconds}`);
      loopParts.push(`echo "---SAMPLE_${i}---"`);
      loopParts.push(smiCmd);
    }
    const fullCmd = loopParts.join(" && ");

    const result = await spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", fullCmd], {
      timeout: timeoutSeconds * 1000,
    });

    if (result.error) return text(`SSH error: ${result.error.message}`);
    if (result.status !== 0) return text(`nvidia-smi failed (exit ${result.status}):\n${result.stderr}\n${result.stdout}`);

    const output = result.stdout ?? "";
    const sampleBlocks = output.split(/---SAMPLE_\d+---/).filter((b) => b.trim());

    const allSamples = sampleBlocks.map((block) => parseNvidiaSmiOutput(block.trim()));
    const gpuCount = allSamples[0]?.length ?? 0;
    const primarySamples = allSamples.map((gpus) => gpus[0]).filter(Boolean);

    if (!primarySamples.length) return text("No GPU data returned from nvidia-smi.");

    const trend = summarizeTrend(primarySamples);

    const sections: string[] = [
      `## GPU Sample Burst (${primarySamples.length} samples, ${intervalSeconds}s apart)\n`,
      `### Trend: **${trend.verdict}**`,
      `- Avg VRAM: ${trend.avgVramPct}% | Avg GPU Util: ${trend.avgGpuUtil}%`,
      `- VRAM Range: ${trend.minVramPct}% — ${trend.maxVramPct}%`,
      "",
      "### Samples",
      "| # | VRAM Used | VRAM % | GPU Util | Label |",
      "|---|-----------|--------|----------|-------|",
    ];

    for (let i = 0; i < primarySamples.length; i++) {
      const s = primarySamples[i];
      sections.push(`| ${i + 1} | ${s.usedMb}/${s.totalMb} MiB | ${s.usedPct}% | ${s.gpuUtil}% | ${s.label} |`);
    }

    // Recommendations based on trend
    sections.push("");
    switch (trend.verdict) {
      case "CONSISTENTLY_IDLE":
        sections.push("**Action needed**: GPU is consistently idle. Check if training actually started. Consider stopping the pod to avoid cost waste.");
        break;
      case "DEGRADING":
        sections.push("**Warning**: GPU utilization is declining. Possible causes: data pipeline exhausted, training finished, or memory leak causing swapping.");
        break;
      case "VOLATILE":
        sections.push("**Note**: Large VRAM fluctuations detected. This may indicate dynamic batching, gradient accumulation, or periodic evaluation phases.");
        break;
      case "IMPROVING":
        sections.push("**Good**: GPU utilization is ramping up. Training is warming up — re-check in a few minutes to confirm stabilization.");
        break;
      case "STABLE_UNDERUTILIZED":
        sections.push("**Action needed**: GPU utilization is stable but low. Consider increasing batch size or switching to a cheaper GPU. Use `gpu_cost_compare` to find alternatives.");
        break;
      case "STABLE_OPTIMAL":
        sections.push("**Excellent**: GPU utilization is stable. No action needed.");
        break;
    }

    if (gpuCount > 1) {
      sections.push("", `**Multi-GPU detected (${gpuCount} GPUs)**: Trend analysis is based on GPU 0 only. Check individual GPU utilization with \`gpu_health_check\` for a full per-GPU breakdown.`);
    }

    if (pod.costPerHr != null) {
      sections.push("", `**Current cost**: $${pod.costPerHr}/hr`);
    }

    return text(sections.join("\n"));
  })
);

// ══════════════════════════════════════════
//  NETWORK VOLUME TOOLS
// ══════════════════════════════════════════

// ── list_network_volumes ──
server.tool(
  "list_network_volumes",
  "List all network volumes in your RunPod account. Network volumes persist data across pod restarts and can be shared between pods in the same datacenter.",
  {},
  safeTool(async () => {
    const volumes = await requireClient().listNetworkVolumes();
    if (!volumes.length) return text("No network volumes found.");
    const header = "ID | Name | Size | Datacenter";
    const sep = "---|------|------|----------";
    const rows = volumes.map((v) => `${v.id} | ${v.name} | ${v.size}GB | ${v.dataCenterId}`);
    return text([header, sep, ...rows].join("\n"));
  })
);

// ── get_network_volume ──
server.tool(
  "get_network_volume",
  "Get details of a specific network volume",
  { volumeId: z.string().describe("Network volume ID") },
  safeTool(async ({ volumeId }) => {
    const vol = await requireClient().getNetworkVolume(volumeId);
    if (!vol) return text(`Network volume ${volumeId} not found.`);
    return text(`ID: ${vol.id}\nName: ${vol.name}\nSize: ${vol.size}GB\nDatacenter: ${vol.dataCenterId}`);
  })
);

// ── create_network_volume ──
server.tool(
  "create_network_volume",
  "Create a new network volume for persistent storage. Volumes persist across pod lifecycles and can be pre-loaded with data via a staging pod. Minimum size is 10GB but 50GB is the practical floor — undersized volumes silently truncate files when full (rsync/tar produce 0-byte files at quota).",
  {
    name: z.string().describe("Volume name"),
    size: z
      .number()
      .min(10)
      .describe(
        "Size in GB. Sizing formula: ceil((dataset_gb + outputs_gb) * 1.3) with 30% headroom for checkpoints/logs/tmp. Practical minimum: 50GB. Cost is ~$0.07/GB/month so 50GB ≈ $3.50/mo, 100GB ≈ $7/mo — the cost of an undersized volume (re-upload, debug, truncated training data) vastly exceeds the storage cost. NEVER use the 10GB minimum unless you've calculated and confirmed the dataset fits."
      ),
    dataCenterId: z.string().describe('Datacenter ID, e.g. "US-GA-1". Must match the datacenter of pods that will use this volume.'),
  },
  safeTool(async ({ name, size, dataCenterId }) => {
    const vol = await requireClient().createNetworkVolume(name, size, dataCenterId);
    const undersized = size < 50
      ? `\n\n⚠ ${size}GB is below the recommended 50GB floor. If your dataset + outputs exceed ${Math.floor(size / 1.3)}GB, files will be silently truncated when the volume fills up.`
      : "";
    return text(
      `Network volume created!\nID: ${vol.id}\nName: ${vol.name}\nSize: ${vol.size}GB\nDatacenter: ${vol.dataCenterId}${undersized}\n\n` +
        `## Next Steps\n→ create_pod_auto(networkVolumeId: "${vol.id}")`
    );
  })
);

// ── delete_network_volume ──
server.tool(
  "delete_network_volume",
  "Permanently delete a network volume. WARNING: This is irreversible and destroys all data on the volume. Ensure no pods are using this volume before deletion.",
  {
    volumeId: z.string().describe("Network volume ID to delete"),
    confirmName: z.string().describe("Type the volume name to confirm deletion (safety check)"),
  },
  safeTool(async ({ volumeId, confirmName }) => {
    const c = requireClient();
    const vol = await c.getNetworkVolume(volumeId);
    if (!vol) return text(`Network volume ${volumeId} not found.`);
    if (vol.name !== confirmName) {
      return text(
        `Safety check failed: you typed "${confirmName}" but the volume name is "${vol.name}".\n` +
          `Please provide the exact volume name in confirmName to proceed with deletion.`
      );
    }

    // Check for pods using this volume
    const pods = await c.listPods();
    const attachedPods = pods.filter((p) => p.networkVolumeId === volumeId);
    if (attachedPods.length > 0) {
      const podList = attachedPods.map((p) => `  - ${p.name} (${p.id}, status: ${p.desiredStatus})`).join("\n");
      return text(
        `Cannot delete volume "${vol.name}": ${attachedPods.length} pod(s) still attached:\n${podList}\n\n` +
          `Stop and delete these pods first, then retry.`
      );
    }

    await c.deleteNetworkVolume(volumeId);
    return text(`Network volume "${vol.name}" (${volumeId}) has been permanently deleted.`);
  })
);

// ══════════════════════════════════════════
//  START
// ══════════════════════════════════════════

const transport = new StdioServerTransport();
await server.connect(transport);
