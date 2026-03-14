#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { RunPodClient } from "./api.js";
import type { Pod } from "./types.js";

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
  version: "0.1.0",
});

// ── Helpers ──

function text(s: string) {
  return { content: [{ type: "text" as const, text: s }] };
}

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
server.tool("list_pods", "List all RunPod pods with status and SSH info", {}, async () => {
  const pods = await requireClient().listPods();
  if (!pods.length) return text("No pods found.");
  return text(pods.map((p) => podSummary(p)).join("\n\n---\n\n"));
});

// ── get_pod ──
server.tool(
  "get_pod",
  "Get detailed info about a specific pod",
  { podId: z.string().describe("Pod ID") },
  async ({ podId }) => text(podSummary(await requireClient().getPod(podId)))
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
    spot: z.boolean().default(true).describe("Use spot (interruptible) instance"),
    bidPerGpu: z.number().optional().describe("Max bid per GPU for spot instances"),
    containerDiskInGb: z.number().default(50),
    volumeInGb: z.number().default(20),
    volumeMountPath: z.string().default("/workspace"),
    networkVolumeId: z.string().optional().describe("Attach existing network volume"),
    sshPublicKey: z.string().optional().describe("SSH public key to inject (overrides account default)"),
    ports: z.array(z.string()).default(["22/tcp"]),
    env: z.record(z.string()).optional().describe("Environment variables"),
    dockerArgs: z.string().optional(),
  },
  async (args) => {
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
      env: args.env,
      dockerArgs: args.dockerArgs,
    };

    if (args.spot && args.bidPerGpu) {
      const result = await requireClient().createSpotPod({ ...opts, bidPerGpu: args.bidPerGpu });
      return text(`Spot pod created!\nID: ${result.id}\n\nUse wait_for_pod to monitor until ready.`);
    }

    const pod = await requireClient().createPod(opts);
    return text(`Pod created!\n${podSummary(pod)}\n\nUse wait_for_pod to monitor until ready.`);
  }
);

// ── create_pod_auto ──
server.tool(
  "create_pod_auto",
  "Create a pod with automatic GPU selection based on stock availability. Tries GPUs in order of preference, skipping those with Low/no stock.",
  {
    name: z.string().describe("Pod name"),
    imageName: z.string().default("runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"),
    gpuPreference: z
      .array(z.string())
      .default(["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA A40", "NVIDIA RTX A5000"])
      .describe("GPU types in order of preference"),
    minVram: z.number().default(12).describe("Minimum VRAM in GB"),
    spot: z.boolean().default(true),
    maxBidPerGpu: z.number().default(0.3).describe("Max spot bid per GPU"),
    containerDiskInGb: z.number().default(50),
    volumeInGb: z.number().default(20),
    sshPublicKey: z.string().optional(),
    env: z.record(z.string()).optional(),
  },
  async (args) => {
    const gpuTypes = await requireClient().listGpuTypes();
    const gpuMap = new Map(gpuTypes.map((g) => [g.id, g]));
    const errors: string[] = [];

    for (const gpuId of args.gpuPreference) {
      const gpu = gpuMap.get(gpuId);
      if (!gpu) continue;
      if (gpu.memoryInGb < args.minVram) continue;
      const stock = gpu.lowestPrice?.stockStatus;
      if (stock === "Low" || stock === "Out of Stock") continue;

      const ondemandPrice = gpu.lowestPrice?.uninterruptablePrice ?? gpu.communityPrice ?? 1.0;
      const bidPrice = args.spot
        ? Math.min(args.maxBidPerGpu, ondemandPrice * 0.8)
        : undefined;

      try {
        const opts = {
          name: args.name,
          imageName: args.imageName,
          gpuTypeIds: [gpuId],
          gpuCount: 1,
          interruptible: args.spot,
          containerDiskInGb: args.containerDiskInGb,
          volumeInGb: args.volumeInGb,
          volumeMountPath: "/workspace",
          sshPublicKey: args.sshPublicKey,
          ports: ["22/tcp"] as string[],
          env: args.env,
        };

        if (args.spot && bidPrice) {
          const result = await requireClient().createSpotPod({ ...opts, bidPerGpu: bidPrice });
          return text(
            `Auto-selected GPU: ${gpu.displayName} (stock: ${stock ?? "unknown"})\n` +
              `Spot bid: $${bidPrice}/hr\n` +
              `Pod ID: ${result.id}\n\nUse wait_for_pod to monitor.`
          );
        }

        const pod = await requireClient().createPod(opts);
        return text(`Auto-selected GPU: ${gpu.displayName} (stock: ${stock ?? "unknown"})\n${podSummary(pod)}`);
      } catch (e) {
        if (isAuthError(e)) throw e; // Re-throw auth/quota errors immediately
        errors.push(`${gpu.displayName}: ${(e as Error).message}`);
        continue;
      }
    }

    const available = gpuTypes
      .filter((g) => g.memoryInGb >= args.minVram && g.lowestPrice?.stockStatus !== "Out of Stock")
      .sort((a, b) => {
        const ap = a.lowestPrice?.minimumBidPrice ?? a.communitySpotPrice ?? Infinity;
        const bp = b.lowestPrice?.minimumBidPrice ?? b.communitySpotPrice ?? Infinity;
        return ap - bp;
      })
      .slice(0, 10);

    const errMsg = errors.length ? `\n\nErrors encountered:\n${errors.join("\n")}` : "";
    return text(
      "No preferred GPU available. Cheapest alternatives:\n\n" +
        available.map((g) => {
          const price = g.lowestPrice?.minimumBidPrice ?? g.communitySpotPrice ?? null;
          const st = g.lowestPrice?.stockStatus ?? "unknown";
          return `${g.displayName} (${g.memoryInGb}GB) - ${price != null ? `$${price}/hr` : "n/a"} [${st}]`;
        }).join("\n") +
        errMsg
    );
  }
);

// ── stop_pod ──
server.tool(
  "stop_pod",
  "Stop a running pod (preserves volume data, stops billing for compute)",
  { podId: z.string() },
  async ({ podId }) => {
    await requireClient().stopPod(podId);
    return text(`Pod ${podId} stop requested.`);
  }
);

// ── start_pod ──
server.tool(
  "start_pod",
  "Start a stopped pod",
  { podId: z.string() },
  async ({ podId }) => {
    await requireClient().startPod(podId);
    return text(`Pod ${podId} start requested. Use wait_for_pod to monitor.`);
  }
);

// ── restart_pod ──
server.tool(
  "restart_pod",
  "Restart a running pod",
  { podId: z.string() },
  async ({ podId }) => {
    await requireClient().restartPod(podId);
    return text(`Pod ${podId} restart requested.`);
  }
);

// ── delete_pod ──
server.tool(
  "delete_pod",
  "Permanently delete a pod (WARNING: destroys all data not on network volumes)",
  { podId: z.string() },
  async ({ podId }) => {
    await requireClient().deletePod(podId);
    return text(`Pod ${podId} deleted.`);
  }
);

// ── wait_for_pod ──
server.tool(
  "wait_for_pod",
  "Poll until a pod is RUNNING with a public IP and SSH port available (includes TCP probe). Returns SSH command when ready.",
  {
    podId: z.string(),
    timeoutSeconds: z.number().default(300).describe("Max wait time in seconds"),
    intervalSeconds: z.number().default(10).describe("Poll interval in seconds"),
  },
  async ({ podId, timeoutSeconds, intervalSeconds }) => {
    const pod = await requireClient().waitForPod(podId, timeoutSeconds * 1000, intervalSeconds * 1000);
    return text(`Pod is ready!\n\n${podSummary(pod)}`);
  }
);

// ── list_gpu_types ──
server.tool(
  "list_gpu_types",
  "List available GPU types with pricing and stock status (via GraphQL)",
  {
    minVram: z.number().default(0).describe("Filter by minimum VRAM in GB"),
    inStockOnly: z.boolean().default(false).describe("Only show GPUs with High/Medium stock"),
  },
  async ({ minVram, inStockOnly }) => {
    let gpus = await requireClient().listGpuTypes();
    if (minVram > 0) gpus = gpus.filter((g) => g.memoryInGb >= minVram);
    if (inStockOnly) gpus = gpus.filter((g) => {
      const status = g.lowestPrice?.stockStatus;
      return status === "High" || status === "Medium";
    });
    gpus.sort((a, b) => {
      const aPrice = a.lowestPrice?.minimumBidPrice ?? a.communitySpotPrice ?? Infinity;
      const bPrice = b.lowestPrice?.minimumBidPrice ?? b.communitySpotPrice ?? Infinity;
      return aPrice - bPrice;
    });

    if (!gpus.length) return text("No GPUs match the criteria.");

    const header = "GPU Type | VRAM | Spot Price | On-Demand | Stock";
    const sep = "---|---|---|---|---";
    const rows = gpus.map((g) => {
      const spot = g.lowestPrice?.minimumBidPrice ?? g.communitySpotPrice ?? null;
      const ondemand = g.lowestPrice?.uninterruptablePrice ?? g.communityPrice ?? null;
      const stock = g.lowestPrice?.stockStatus ?? (g.communityCloud ? "available" : "n/a");
      return `${g.displayName} | ${g.memoryInGb}GB | ${spot != null ? `$${spot}/hr` : "n/a"} | ${ondemand != null ? `$${ondemand}/hr` : "n/a"} | ${stock}`;
    });
    return text([header, sep, ...rows].join("\n"));
  }
);

// ── get_ssh_command ──
server.tool(
  "get_ssh_command",
  "Get the SSH command for connecting to a running pod",
  { podId: z.string() },
  async ({ podId }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    const cmd = c.getSshCommandString(pod);
    if (!cmd) return text("Pod is not ready (no public IP or SSH port). Try wait_for_pod first.");
    return text(cmd);
  }
);

// ── execute_ssh_command (uses spawnSync with args array — no shell injection) ──
server.tool(
  "execute_ssh_command",
  "Execute a command on a running pod via SSH. Returns stdout/stderr.",
  {
    podId: z.string(),
    command: z.string().describe("Shell command to execute on the pod"),
    timeoutSeconds: z.number().default(120).describe("Command timeout"),
  },
  async ({ podId, command, timeoutSeconds }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    const sshArgs = c.getSshArgs(pod);
    if (!sshArgs) return text("Pod is not ready for SSH.");

    const { spawnSync } = await import("node:child_process");
    const result = spawnSync(sshArgs[0], [...sshArgs.slice(1), "--", command], {
      timeout: timeoutSeconds * 1000,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
    });

    if (result.error) return text(`SSH error: ${result.error.message}`);
    if (result.status !== 0) {
      return text(`Exit code: ${result.status}\n\nStderr:\n${result.stderr}\n\nStdout:\n${result.stdout}`);
    }
    return text(result.stdout || "(no output)");
  }
);

// ── upload_files (uses spawnSync with args array — no shell injection) ──
server.tool(
  "upload_files",
  "Upload local files/directories to a pod via rsync",
  {
    podId: z.string(),
    localPath: z.string().describe("Local file or directory path"),
    remotePath: z.string().default("/workspace").describe("Destination path on pod"),
    dryRun: z.boolean().default(false).describe("Show command without executing"),
  },
  async ({ podId, localPath, remotePath, dryRun }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    const args = c.getRsyncArgs(pod, localPath, remotePath, "upload");
    if (!args) return text("Pod is not ready for file transfer.");
    if (dryRun) return text(`Command (dry run):\n${args.join(" ")}`);

    const { spawnSync } = await import("node:child_process");
    const result = spawnSync(args[0], args.slice(1), {
      encoding: "utf-8",
      timeout: 600_000,
      maxBuffer: 10 * 1024 * 1024,
    });

    if (result.error) return text(`Upload error: ${result.error.message}`);
    if (result.status !== 0) return text(`Upload failed (exit ${result.status}):\n${result.stderr}`);
    return text(`Upload complete.\n\n${result.stdout}`);
  }
);

// ── download_files (uses spawnSync with args array — no shell injection) ──
server.tool(
  "download_files",
  "Download files from a pod to local filesystem via rsync",
  {
    podId: z.string(),
    remotePath: z.string().describe("Path on pod to download"),
    localPath: z.string().describe("Local destination path"),
    dryRun: z.boolean().default(false),
  },
  async ({ podId, remotePath, localPath, dryRun }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    const args = c.getRsyncArgs(pod, localPath, remotePath, "download");
    if (!args) return text("Pod is not ready for file transfer.");
    if (dryRun) return text(`Command (dry run):\n${args.join(" ")}`);

    const { spawnSync } = await import("node:child_process");
    const result = spawnSync(args[0], args.slice(1), {
      encoding: "utf-8",
      timeout: 600_000,
      maxBuffer: 10 * 1024 * 1024,
    });

    if (result.error) return text(`Download error: ${result.error.message}`);
    if (result.status !== 0) return text(`Download failed (exit ${result.status}):\n${result.stderr}`);
    return text(`Download complete.\n\n${result.stdout}`);
  }
);

// ══════════════════════════════════════════
//  START
// ══════════════════════════════════════════

const transport = new StdioServerTransport();
await server.connect(transport);
