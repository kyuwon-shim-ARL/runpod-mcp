#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { RunPodClient } from "./api.js";
import type { Pod } from "./types.js";
import { parseNvidiaSmiOutput, calcSuggestedBatchSize, isOverprovisioned, injectPytorchEnv } from "./gpu-utils.js";

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

type ToolResult = { content: Array<{ type: "text"; text: string }>; isError?: boolean };

function text(s: string): ToolResult {
  return { content: [{ type: "text" as const, text: s }] };
}

function errorResult(e: unknown): ToolResult {
  const msg = e instanceof Error ? e.message : String(e);
  return { content: [{ type: "text" as const, text: `Error: ${msg}` }], isError: true };
}

/** Wrap a tool handler with error catching that returns MCP-friendly error text instead of throwing */
function safeTool<T extends Record<string, unknown>>(
  handler: (args: T) => Promise<ToolResult>
): (args: T) => Promise<ToolResult> {
  return async (args: T) => {
    try {
      return await handler(args);
    } catch (e) {
      return errorResult(e);
    }
  };
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
    };

    if (args.spot && args.bidPerGpu) {
      const result = await requireClient().createSpotPod({ ...opts, bidPerGpu: args.bidPerGpu });
      return text(`Spot pod created!\nID: ${result.id}\n\nUse wait_for_pod to monitor until ready.`);
    }

    const pod = await requireClient().createPod(opts);
    return text(`Pod created!\n${podSummary(pod)}\n\nUse wait_for_pod to monitor until ready.`);
  })
);

// ── create_pod_auto ──
server.tool(
  "create_pod_auto",
  "Create a pod with automatic GPU selection based on stock availability. Tries GPUs in order of preference, skipping those with Low/no stock. Prefer over create_pod when no specific GPU model is required.",
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
    optimizePytorch: z
      .boolean()
      .default(false)
      .describe("Inject PyTorch CUDA optimization env vars (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True). Requires PyTorch >= 2.0."),
  },
  safeTool(async (args) => {
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
      const minBid = gpu.lowestPrice?.minimumBidPrice ?? 0;
      const bidPrice = args.spot
        ? Math.min(args.maxBidPerGpu, ondemandPrice * 0.8)
        : undefined;

      // Skip if bid price is below minimum bid
      if (args.spot && bidPrice != null && minBid > 0 && bidPrice < minBid) {
        errors.push(`${gpu.displayName}: Bid $${bidPrice.toFixed(3)}/hr below minimum $${minBid}/hr, skipped`);
        continue;
      }

      const overprovisionWarning = isOverprovisioned(gpu.memoryInGb, args.minVram)
        ? `\nOverprovisioned: ${gpu.displayName} has ${gpu.memoryInGb}GB VRAM but you requested ${args.minVram}GB minimum.\n` +
          `  Consider a smaller GPU to save cost, or increase your workload to utilize the extra VRAM.\n`
        : "";

      try {
        const podEnv = injectPytorchEnv(args.env, args.optimizePytorch);

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
          env: podEnv,
        };

        if (args.spot && bidPrice) {
          const result = await requireClient().createSpotPod({ ...opts, bidPerGpu: bidPrice });
          return text(
            `Auto-selected GPU: ${gpu.displayName} (stock: ${stock ?? "unknown"})\n` +
              `Spot bid: $${bidPrice}/hr\n` +
              `Pod ID: ${result.id}${overprovisionWarning}\n\nUse wait_for_pod to monitor.`
          );
        }

        const pod = await requireClient().createPod(opts);
        return text(`Auto-selected GPU: ${gpu.displayName} (stock: ${stock ?? "unknown"})${overprovisionWarning}\n${podSummary(pod)}`);
      } catch (e) {
        if (isAuthError(e)) return errorResult(e); // Auth/quota errors stop GPU iteration immediately
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
    return text(`Pod ${podId} start requested. Use wait_for_pod to monitor.`);
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
  "Permanently delete a pod (WARNING: destroys all data not on network volumes)",
  { podId: z.string() },
  safeTool(async ({ podId }) => {
    await requireClient().deletePod(podId);
    return text(`Pod ${podId} deleted.`);
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
  safeTool(async ({ podId, timeoutSeconds, intervalSeconds }) => {
    const pod = await requireClient().waitForPod(podId, timeoutSeconds * 1000, intervalSeconds * 1000);
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

// ── execute_ssh_command (uses spawnSync with args array — no shell injection) ──
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
  })
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
  safeTool(async ({ podId, localPath, remotePath, dryRun }) => {
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
  })
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
  safeTool(async ({ podId, remotePath, localPath, dryRun }) => {
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

    const { spawnSync } = await import("node:child_process");

    // Query per-GPU metrics
    const gpuCmd =
      "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits 2>&1";
    const gpuResult = spawnSync(sshArgs[0], [...sshArgs.slice(1), "--", gpuCmd], {
      timeout: timeoutSeconds * 1000,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
    });

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

    // Query per-process GPU memory
    const procCmd =
      "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true";
    const procResult = spawnSync(sshArgs[0], [...sshArgs.slice(1), "--", procCmd], {
      timeout: timeoutSeconds * 1000,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
    });

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
        sections.push("Consider using `gpu_cost_compare` to find a cheaper GPU that matches your actual usage.");
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
    const currentPrice = currentCost ?? currentGpu?.lowestPrice?.minimumBidPrice ?? currentGpu?.communitySpotPrice ?? 0;

    // Find alternatives: same or higher VRAM, in stock
    const alternatives = gpuTypes
      .filter((g) => {
        if (g.id === currentGpu?.id) return false;
        if (g.memoryInGb < minVram) return false;
        const stock = g.lowestPrice?.stockStatus;
        if (stock === "Out of Stock") return false;
        return true;
      })
      .map((g) => {
        const spotPrice = g.lowestPrice?.minimumBidPrice ?? g.communitySpotPrice ?? null;
        const ondemandPrice = g.lowestPrice?.uninterruptablePrice ?? g.communityPrice ?? null;
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
        const stock = g.lowestPrice?.stockStatus ?? "unknown";
        sections.push(
          `| ${g.displayName} | ${g.memoryInGb}GB | ${g.spotPrice != null ? `$${g.spotPrice}/hr` : "n/a"} | ${g.ondemandPrice != null ? `$${g.ondemandPrice}/hr` : "n/a"} | ${stock} | ~$${savings.toFixed(0)} |`
        );
      }
      sections.push("");
    }

    if (similar.length > 0 && cheaper.length < 5) {
      sections.push("### Other Options (same or higher price)\n");
      for (const g of similar.slice(0, 5)) {
        const stock = g.lowestPrice?.stockStatus ?? "unknown";
        sections.push(`- ${g.displayName} (${g.memoryInGb}GB) - $${g.comparePrice}/hr [${stock}]`);
      }
    }

    return text(sections.join("\n"));
  })
);

// ══════════════════════════════════════════
//  START
// ══════════════════════════════════════════

const transport = new StdioServerTransport();
await server.connect(transport);
