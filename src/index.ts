#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { dirname, isAbsolute, resolve } from "node:path";
import { RunPodClient, spawnAsync } from "./api.js";
import type { Pod } from "./types.js";
import { safeTool, text, errorResult } from "./tool-helpers.js";
import type { ToolResult } from "./tool-helpers.js";
import { parseNvidiaSmiOutput, calcSuggestedBatchSize, isOverprovisioned, injectPytorchEnv, summarizeTrend, getStockStatus, isInStock, getSpotPrice, getOnDemandPrice } from "./gpu-utils.js";
import { filterStalePods, selectGpuCandidates, deletePodWithStop, DEFAULT_DC_PRIORITY, formatDcGpuFailureMatrix, buildPodMetadataPath, toYaml, buildPodMetadataStub, parseDuBytes, parseDfAvailBytes, checkFreeSpace, checkSizeMatch, looksLikeSetupCommand, estimatePodCost } from "./pod-ops.js";

const COST_GATE_GPU_COUNT = 2;       // gpuCount >= this triggers gate
const COST_GATE_HOURLY_USD = 1.0;    // ondemandPrice * gpuCount >= this triggers gate

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

/**
 * Reads the SSH public key from SSH_KEY_PATH env var.
 * Handles .pub extension deduplication and silently returns undefined on any error.
 */
async function readSshPubKey(): Promise<string | undefined> {
  const keyPath = process.env.SSH_KEY_PATH;
  if (!keyPath) return undefined;
  const pubPath = keyPath.endsWith(".pub") ? keyPath : keyPath + ".pub";
  try {
    const content = await readFile(pubPath, "utf8");
    return content.trim();
  } catch {
    return undefined; // ENOENT, EACCES: silent fallback
  }
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
      .default("COMMUNITY")
      .describe("Cloud type filter: COMMUNITY (default, cheaper/shared), SECURE (dedicated), or ALL"),
    optimizePytorch: z
      .boolean()
      .default(false)
      .describe("Inject PyTorch CUDA optimization env vars (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True). Requires PyTorch >= 2.0."),
  },
  safeTool(async (args) => {
    const autoSshKey = await readSshPubKey();
    const sshWarnText = (process.env.SSH_KEY_PATH && !autoSshKey)
      ? "\n\n⚠️ SSH_KEY_PATH 설정됨 but 공개키 읽기 실패 — 직접 SSH/SCP 불가, execute_ssh_command(프록시) 사용"
      : "";
    const resolvedSshPublicKey = args.sshPublicKey ?? autoSshKey;
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
      ...(resolvedSshPublicKey ? { sshPublicKey: resolvedSshPublicKey } : {}),
      ports: args.ports,
      env: podEnv,
      dockerArgs: args.dockerArgs,
      cloudType: args.cloudType,
    };

    const buildStub = (podId: string, costPerHr: number | null) =>
      buildPodMetadataStub({
        pod_id: podId,
        name: args.name,
        created_at: new Date().toISOString(),
        gpu: args.gpuTypeId,
        gpu_count: args.gpuCount,
        cost_per_hr: costPerHr ?? undefined,
        image: args.imageName,
        container_disk_gb: args.containerDiskInGb,
        network_volume: args.networkVolumeId
          ? { id: args.networkVolumeId, name: "<lookup with get_network_volume>", size_gb: 0 }
          : null,
      });

    if (args.spot && args.bidPerGpu) {
      const result = await requireClient().createSpotPod({ ...opts, bidPerGpu: args.bidPerGpu });
      const stub = buildStub(result.id, args.bidPerGpu);
      return text(
        `Spot pod created!\nID: ${result.id}${sshWarnText}\n\n` +
          `## Pod Metadata Stub (pass to save_pod_metadata after enriching)\n\`\`\`json\n${stub}\n\`\`\`\n\n` +
          `## Next Steps\n→ wait_for_pod(podId: "${result.id}")\n→ save_pod_metadata({metadata: <stub above with purpose filled in>})`
      );
    }

    const pod = await requireClient().createPod(opts);
    const stub = buildStub(pod.id, null);
    return text(
      `Pod created!\n${podSummary(pod)}${sshWarnText}\n\n` +
        `## Pod Metadata Stub (pass to save_pod_metadata after enriching)\n\`\`\`json\n${stub}\n\`\`\`\n\n` +
        `## Next Steps\n→ wait_for_pod(podId: "${pod.id}")\n→ save_pod_metadata({metadata: <stub above with purpose filled in>})`
    );
  })
);

// ── create_pod_auto ──
server.tool(
  "create_pod_auto",
  "Create a pod with automatic GPU selection based on stock availability. Tries GPUs in order of preference, including Low stock (worth trying). Use dryRun=true to preview GPU selection and cost estimate without creating a pod.\n⚠️ costSafetyConfirmed는 사용자가 직접 확인한 경우에만 true로 설정하세요. Claude가 자동으로 true를 설정하는 것은 엄격히 금지됩니다.",
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
      .default("COMMUNITY")
      .describe("Cloud type filter: COMMUNITY (default, cheaper/shared), SECURE (dedicated), or ALL"),
    dryRun: z
      .boolean()
      .default(false)
      .describe("Preview GPU selection and cost estimate without creating a pod. Note: GPU availability may change between dry run and actual creation."),
    costSafetyConfirmed: z
      .boolean()
      .optional()
      .describe("Set to true only after the user has reviewed the cost safety checklist. Required for gpuCount >= 2 with dryRun: false."),
    nvReadinessToken: z
      .string()
      .optional()
      .describe("Token from verify_data_on_nv. Required when gpuCount >= 2 AND networkVolumeId is set (ensures data was verified on NV before launching expensive multi-GPU pod)."),
  },
  safeTool(async (args) => {
    // NV readiness token check: multi-GPU + NV requires prior verify_data_on_nv call
    if (args.gpuCount >= COST_GATE_GPU_COUNT && args.networkVolumeId && !args.dryRun) {
      if (!args.nvReadinessToken) {
        return text(
          `⚠️ NV READINESS TOKEN REQUIRED (gpuCount=${args.gpuCount}, networkVolumeId=${args.networkVolumeId})\n` +
          `고비용 다중-GPU 팟 생성 전 데이터 검증이 필요합니다:\n` +
          `1. 스테이징 팟에서 데이터 전송 완료\n` +
          `2. verify_data_on_nv(podId, requiredPaths) 호출 → 토큰 발급\n` +
          `3. 발급된 토큰을 nvReadinessToken 파라미터에 전달해 재호출하세요.`
        );
      }
      // Validate token
      try {
        const tokenPath = `${NV_READY_DIR}/nv_ready_${args.networkVolumeId}.json`;
        const tokenRaw = await readFile(tokenPath, "utf-8");
        const tokenData = JSON.parse(tokenRaw) as { token: string; nvId: string; verifiedAt: string };
        if (tokenData.token !== args.nvReadinessToken) {
          return text(`❌ NV readiness token mismatch for volume ${args.networkVolumeId}. Re-run verify_data_on_nv to get a fresh token.`);
        }
        const ageHours = (Date.now() - new Date(tokenData.verifiedAt).getTime()) / 3_600_000;
        if (ageHours > TOKEN_TTL_HOURS) {
          return text(`❌ NV readiness token expired (${ageHours.toFixed(1)}h old, TTL=${TOKEN_TTL_HOURS}h). Re-run verify_data_on_nv.`);
        }
      } catch {
        return text(`❌ NV readiness token file not found for volume ${args.networkVolumeId}. Run verify_data_on_nv first.`);
      }
    }

    // Cost safety gate: multi-GPU pods require explicit user confirmation.
    // Tries MCP Elicitation first (Claude Code >= v2.1.76); falls back to boolean gate for older clients.
    if (args.gpuCount >= COST_GATE_GPU_COUNT && !args.dryRun) {
      const mcpServer = (server as any).server;
      const hasElicitation = mcpServer?._clientCapabilities?.elicitation !== undefined;
      const booleanFallback = () => !args.costSafetyConfirmed
        ? text(
            `⚠️ COST SAFETY CHECK (gpuCount=${args.gpuCount})\n` +
            `고비용 팟 생성 전 확인하세요:\n` +
            `[ ] 1. 데이터/코드가 이미 준비됨 (로컬 전처리 or 전송 팟 완료)\n` +
            `[ ] 2. 1-GPU로 검증 테스트 완료됨 (VRAM·속도·코드 정상 동작)\n\n` +
            `확인 완료 후 동일 파라미터에 costSafetyConfirmed: true를 추가해 재호출하세요.`
          )
        : null;

      if (hasElicitation) {
        try {
          const elicitResult = await mcpServer.elicitInput({
            message: `⚠️ COST SAFETY CHECK (gpuCount=${args.gpuCount})\n고비용 팟 생성 전 확인하세요:\n[ ] 1. 데이터/코드가 이미 준비됨 (로컬 전처리 or 전송 팟 완료)\n[ ] 2. 1-GPU로 검증 테스트 완료됨 (VRAM·속도·코드 정상 동작)\n\n위 항목을 확인했으면 승인하세요.`,
            requestedSchema: {
              type: 'object' as const,
              properties: {
                confirmed: {
                  type: 'boolean' as const,
                  title: '비용 안전 체크리스트 확인 완료',
                  description: '위 항목을 모두 확인했습니다',
                  default: false
                }
              },
              required: ['confirmed']
            }
          });
          const approved = elicitResult?.action === 'accept' && elicitResult?.content?.confirmed === true;
          if (!approved) {
            return text(`🚫 취소됨. 체크리스트 확인 후 재시도하세요.\n(elicitation action: ${elicitResult?.action ?? 'null'})`);
          }
        } catch {
          const blocked = booleanFallback();
          if (blocked) return blocked;
        }
      } else {
        const blocked = booleanFallback();
        if (blocked) return blocked;
      }
    }

    const c = requireClient();

    // Auto-inject SSH public key from SSH_KEY_PATH env if not explicitly provided
    const autoSshKey = await readSshPubKey();
    const sshWarnText = (process.env.SSH_KEY_PATH && !autoSshKey)
      ? "\n⚠️ SSH_KEY_PATH 설정됨 but 공개키 읽기 실패 — 직접 SSH/SCP 불가, execute_ssh_command(프록시) 사용"
      : "";
    const resolvedSshPublicKey = args.sshPublicKey ?? autoSshKey;

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
        (args.gpuCount >= 2 ? `⚠️ COST SAFETY REMINDER: 실제 생성(dryRun: false) 시 사전 차단이 발동됩니다.\n\n` : ``) +
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
            ...(resolvedSshPublicKey ? { sshPublicKey: resolvedSshPublicKey } : {}),
            ports: ["22/tcp"] as string[],
            env: podEnv,
            networkVolumeId: args.networkVolumeId,
            dataCenterIds: [dc],
            cloudType: args.cloudType,
          };

          if (args.spot && bidPrice) {
            const result = await c.createSpotPod({ ...opts, bidPerGpu: bidPrice });
            const stub = buildPodMetadataStub({
              pod_id: result.id,
              name: args.name,
              created_at: new Date().toISOString(),
              datacenter: dc,
              gpu: `${gpu.displayName} (${gpu.memoryInGb}GB)`,
              gpu_count: args.gpuCount,
              cost_per_hr: bidPrice,
              image: args.imageName,
              container_disk_gb: args.containerDiskInGb,
              network_volume: args.networkVolumeId
                ? { id: args.networkVolumeId, name: "<lookup with get_network_volume>", size_gb: 0, datacenter: dc }
                : null,
            });
            return text(
              `Auto-selected: ${gpu.displayName} in ${dc} (stock: ${stock ?? "unknown"})\n` +
                `Spot bid: $${bidPrice}/hr\n` +
                `Pod ID: ${result.id}${overprovisionWarning}${volumeNote}${sshWarnText}\n\n` +
                `## Pod Metadata Stub (pass to save_pod_metadata after enriching)\n\`\`\`json\n${stub}\n\`\`\`\n\n` +
                `## Next Steps\n→ wait_for_pod(podId: "${result.id}")\n→ save_pod_metadata({metadata: <stub above with purpose filled in>})`
            );
          }

          const pod = await c.createPod(opts);
          const dcLabel = nvDataCenterId ? dc : `${dc} (price: $${ondemandPrice}/hr)`;
          const stub = buildPodMetadataStub({
            pod_id: pod.id,
            name: args.name,
            created_at: new Date().toISOString(),
            datacenter: dc,
            gpu: `${gpu.displayName} (${gpu.memoryInGb}GB)`,
            gpu_count: args.gpuCount,
            cost_per_hr: ondemandPrice,
            image: args.imageName,
            container_disk_gb: args.containerDiskInGb,
            network_volume: args.networkVolumeId
              ? { id: args.networkVolumeId, name: "<lookup with get_network_volume>", size_gb: 0, datacenter: dc }
              : null,
          });
          return text(
            `Auto-selected: ${gpu.displayName} in ${dcLabel} (stock: ${stock ?? "unknown"})${overprovisionWarning}${volumeNote}${sshWarnText}\n${podSummary(pod)}\n\n` +
              `## Pod Metadata Stub (pass to save_pod_metadata after enriching)\n\`\`\`json\n${stub}\n\`\`\`\n\n` +
              `## Next Steps\n→ wait_for_pod(podId: "${pod.id}")\n→ save_pod_metadata({metadata: <stub above with purpose filled in>})`
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
    return text(`⚠️ 주의: stop은 과금이 계속됩니다. 훈련이 완료되었으면 delete_pod를 사용하세요.\n\nPod ${podId} stop requested.`);
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
  "Permanently delete a pod (auto-stops if running). WARNING: destroys all data not on network volumes. Returns an estimated total cost (uptime × cost_per_hr) for closing the pod metadata record.",
  {
    podId: z.string(),
    artifactsSavedConfirmed: z.boolean().optional().describe(
      "Required when pod has no network volume. Set true only after confirming model weights/results are downloaded locally. " +
      "Claude must NOT set this automatically without explicit user confirmation. (Same rule as costSafetyConfirmed.)"
    ),
  },
  safeTool(async ({ podId, artifactsSavedConfirmed }) => {
    const c = requireClient();
    // Capture cost-relevant info BEFORE deletion (the pod is gone after).
    let costEstimate: ReturnType<typeof estimatePodCost> = null;
    let podName: string | undefined;
    let hasNv = false;
    try {
      const pod = await c.getPod(podId);
      podName = pod.name;
      costEstimate = estimatePodCost(pod.costPerHr, pod.lastStartedAt);
      hasNv = !!(pod as { networkVolumeId?: string }).networkVolumeId;
    } catch {
      // Pod may already be unreachable; proceed with deletion attempt anyway.
    }

    // Artifact gate: block deletion when NV is absent and user hasn't confirmed download.
    if (!hasNv && artifactsSavedConfirmed !== true) {
      return text(
        `⛔ ARTIFACT GATE: Pod "${podName ?? podId}" has no network volume — container disk data will be permanently lost.\n\n` +
        `Before deleting:\n` +
        `  1. download_files(podId: "${podId}", remotePath: "/workspace", localPath: "./outputs/")\n` +
        `     OR confirm outputs are already saved elsewhere.\n` +
        `  2. Re-call: delete_pod(podId: "${podId}", artifactsSavedConfirmed: true)\n\n` +
        `⚠️ Claude must NOT set artifactsSavedConfirmed:true automatically. Requires explicit user confirmation. (Same rule as costSafetyConfirmed.)`
      );
    }

    const { wasRunning } = await deletePodWithStop(c, podId);

    const stoppedNote = wasRunning ? " (was running → auto-stopped first)" : "";
    const nvNote = hasNv
      ? `\n💡 NV pod: /workspace outputs persist after deletion. Data outside /workspace (container disk) is gone.`
      : "";
    const costNote = costEstimate
      ? `\n\n[Cost estimate] Uptime ${costEstimate.hours.toFixed(2)}h × rate → $${costEstimate.cost.toFixed(2)}` +
        `\nUpdate the pod metadata: read .omc/pods/<file>.yaml → set deleted_at and cost_actual_usd → save_pod_metadata → git commit "chore(pod): close ${podName ?? podId}"`
      : "\n\n[Cost estimate] Unavailable (no costPerHr or lastStartedAt). Set cost_actual_usd manually if you tracked it.";

    return text(`Pod ${podId} deleted.${stoppedNote}${nvNote}${costNote}`);
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

    const noNvPods = stale.filter(s => !(s.pod as { networkVolumeId?: string }).networkVolumeId);

    if (dryRun) {
      const lines = stale.map(s => {
        const hasNv = !!(s.pod as { networkVolumeId?: string }).networkVolumeId;
        const artifactWarn = hasNv ? "" : ` ⚠️ NO NV — container disk data will be lost`;
        return `  - ${s.pod.name} (${s.pod.id}) — idle ${s.idleHours}h, ${s.pod.gpu?.displayName ?? "unknown GPU"}, $${s.pod.costPerHr ?? "?"}/hr${artifactWarn}`;
      });
      const nvWarning = noNvPods.length
        ? `\n\n⚠️ ARTIFACT WARNING: ${noNvPods.length} pod(s) have no network volume — container disk data will be permanently lost on deletion:\n` +
          noNvPods.map(s => `  - ${s.pod.name}: use delete_pod(artifactsSavedConfirmed:true) after downloading outputs`).join("\n")
        : "";
      return text(`[DRY RUN] Would delete ${stale.length} stale pod(s):\n${lines.join("\n")}${nvWarning}\n\nRe-run with dryRun=false to delete.`);
    }

    // Warn about NV-less pods before batch deletion (no hard block to preserve automation).
    const preWarning = noNvPods.length
      ? `⚠️ ARTIFACT WARNING: ${noNvPods.length} pod(s) with no network volume will lose container disk data:\n` +
        noNvPods.map(s => `  - ${s.pod.name}`).join("\n") + "\nProceeding with deletion...\n\n"
      : "";

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

    return text(`${preWarning}Deleted ${deleted.length} stale pod(s):\n${deleted.map(d => `  - ${d}`).join("\n")}${failed.length ? `\n\nFailed: ${failed.join(", ")}` : ""}`);
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
  "Persist a pod's provisioning recipe to disk so debugging is possible after the pod is deleted. Writes YAML to `{path}/{YYYY-MM-DD}_{podName}.yaml`. Default path: `.omc/pods/` relative to the caller's CWD (aligned with the existing `.omc/*` convention). The file is meant to be git-committed in the user's project repo. Call after pod setup completes (post-create installs done, training launched), again on incidents (append to incidents[] and re-save), and once more before deletion (set deleted_at + cost_actual_usd). See CLAUDE.md 'Pod Metadata Persistence' for the full workflow.",
  {
    metadata: podMetadataSchema,
    path: z
      .string()
      .optional()
      .describe("Base directory for the metadata file (default: '.omc/pods'). Relative paths resolve against the current working directory. Will be created if it does not exist."),
  },
  safeTool(async ({ metadata, path }) => {
    const basePath = path ?? ".omc/pods";
    const relPath = buildPodMetadataPath(metadata, basePath);
    const absPath = isAbsolute(relPath) ? relPath : resolve(process.cwd(), relPath);

    try {
      await mkdir(dirname(absPath), { recursive: true });
      await writeFile(absPath, toYaml(metadata), "utf8");
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
        `## Next Steps\n→ git add ${relPath} && git commit -m "chore(pod): record ${metadata.name}"`
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
    // Setup-step heuristic: nudge Claude to record this in pod metadata if it
    // looks like an apt/pip/git/etc install. Avoids forgetting setup steps that
    // make the pod reproducible later. See CLAUDE.md "Pod Metadata Persistence".
    const setupHint = looksLikeSetupCommand(command)
      ? `\n\n[Setup step detected] This command looks like provisioning. Append it to post_create_steps in your pod metadata yaml (Read .omc/pods/<pod>.yaml → modify → save_pod_metadata).`
      : "";
    return text((result.stdout || "(no output)") + setupHint);
  })
);

// ── upload_files (uses async spawn with args array — no shell injection) ──
//
// Integrity checks (Patch D — silent truncation defense):
//   1. PRE: query pod's free space at the destination, abort if < local + 10%
//   2. POST: compare local `du -sb` vs remote `du -sb` of the destination,
//      flag if remote is < 95% of local (the silent-truncation pattern from
//      the piu-v2 incident where 16996 .npy files were 0-byte)
// Both can be skipped with verifySize=false for power-user cases.
server.tool(
  "upload_files",
  "Upload local files/directories to a pod via rsync. By default performs free-space precheck and post-upload size verification to catch silent truncation (the failure mode where rsync produces 0-byte files when the destination quota is full).",
  {
    podId: z.string(),
    localPath: z.string().describe("Local file or directory path"),
    remotePath: z.string().default("/workspace").describe("Destination path on pod"),
    dryRun: z.boolean().default(false).describe("Show command without executing"),
    verifySize: z
      .boolean()
      .default(true)
      .describe("Run pre-upload free-space precheck and post-upload du size match. Set false to skip (only for power users with a reason)."),
    verifyPath: z
      .string()
      .optional()
      .describe("Override the path used for the post-upload du verification on the pod. Defaults to `${remotePath}/${basename(localPath)}` for directory uploads, or `${remotePath}` for single files."),
  },
  safeTool(async ({ podId, localPath, remotePath, dryRun, verifySize, verifyPath }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    const args = c.getRsyncArgs(pod, localPath, remotePath, "upload");
    if (!args) return text("Pod is not ready for file transfer.");
    if (dryRun) return text(`Command (dry run):\n${args.join(" ")}`);

    const sshArgs = c.getSshArgs(pod);
    if (!sshArgs && verifySize) {
      return text("Pod has no SSH endpoint; cannot run integrity checks. Pass verifySize=false to skip them.");
    }

    // ── Step 1: local size measurement ──
    let localBytes: number | null = null;
    if (verifySize) {
      const duLocal = await spawnAsync("du", ["-sb", localPath], { timeout: 60_000 });
      if (duLocal.status !== 0) {
        return text(`Failed to measure local size of ${localPath}: ${duLocal.stderr || "du exited non-zero"}`);
      }
      localBytes = parseDuBytes(duLocal.stdout);
      if (localBytes == null) {
        return text(`Could not parse local du output: ${duLocal.stdout}`);
      }
    }

    // ── Step 2: pre-upload free-space precheck ──
    if (verifySize && sshArgs && localBytes != null) {
      // Run df on the parent of remotePath (the destination directory must exist before we can df it)
      const dfCmd = `df -B1 --output=avail "${remotePath}" 2>/dev/null || df -B1 --output=avail "$(dirname "${remotePath}")"`;
      const df = await spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", dfCmd], { timeout: 30_000 });
      if (df.status === 0) {
        const avail = parseDfAvailBytes(df.stdout);
        if (avail != null) {
          const check = checkFreeSpace(localBytes, avail);
          if (check.status === "FREE_SPACE_LOW") {
            return text(`[PRE-UPLOAD CHECK FAILED]\n${check.message}`);
          }
        }
        // If parse failed, fall through silently — better to upload than block on a transient parse miss
      }
    }

    // ── Step 3: actual rsync ──
    const result = await spawnAsync(args[0], args.slice(1), { timeout: 600_000 });

    if (result.error) return text(`Upload error: ${result.error.message}`);
    if (result.status !== 0) return text(`Upload failed (exit ${result.status}):\n${result.stderr}`);

    let postNote = "";

    // ── Step 4: post-upload size verification ──
    if (verifySize && sshArgs && localBytes != null) {
      // Infer the actual destination on the pod.
      // rsync semantics: if localPath ends with `/`, contents go INTO remotePath.
      // Otherwise the basename of localPath is appended to remotePath.
      const inferredDest = verifyPath
        ?? (localPath.endsWith("/")
          ? remotePath
          : `${remotePath.replace(/\/+$/, "")}/${localPath.replace(/\/+$/, "").split("/").pop()}`);

      const duRemote = await spawnAsync(
        sshArgs[0],
        [...sshArgs.slice(1), "--", `du -sb "${inferredDest}" 2>/dev/null`],
        { timeout: 60_000 }
      );

      if (duRemote.status === 0) {
        const remoteBytes = parseDuBytes(duRemote.stdout);
        if (remoteBytes != null) {
          const check = checkSizeMatch(localBytes, remoteBytes);
          if (check.status === "SIZE_MISMATCH") {
            return text(
              `[POST-UPLOAD INTEGRITY FAILED]\n${check.message}\n\n` +
                `Verified path on pod: ${inferredDest}\n` +
                `Rsync output:\n${result.stdout}`
            );
          }
          postNote = `\n\n[Integrity OK] ${check.message} at ${inferredDest}`;
        }
      } else {
        postNote = `\n\n[Integrity SKIPPED] Could not du remote path "${inferredDest}" (${duRemote.stderr.trim() || "no output"}). Pass verifyPath to override.`;
      }
    }

    return text(`Upload complete.\n\n${result.stdout}${postNote}`);
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

// ── plan_gpu_job ──
server.tool(
  "plan_gpu_job",
  "Pre-flight planning for a GPU job. Given a job description, recommends GPU, estimates total cost, determines if a staging pod + network volume are needed, and optionally generates a /gpu-exec pipeline_spec.json stub. Call this BEFORE create_pod_auto to avoid idle billing and wrong GPU selection.",
  {
    purpose: z.string().describe("What you're training/running (e.g. 'Fine-tune LLaMA 3 8B on 22GB dataset')"),
    datasetGb: z.number().optional().describe("Dataset size in GB to transfer to the pod"),
    modelSizeGb: z.number().optional().describe("Approximate model VRAM requirement in GB (used to filter GPUs)"),
    gpuCount: z.number().default(1).describe("Desired number of GPUs for the training pod"),
    expectedHours: z.number().optional().describe("Estimated training duration in hours"),
    gpuPreference: z.array(z.string()).optional().describe("Preferred GPU types in order. Defaults to RTX 3090 / 4090 / A40 / A5000"),
    seedCount: z.number().default(1).describe("Number of random seeds to run (e.g. 3 for seed 42/123/456). When > 1, shows parallel pod pattern instead of sequential — same cost, N× faster."),
    armCount: z.number().default(1).describe("Number of experimental arms/conditions (e.g. 2 for T14 vs T15). Total pods = armCount × seedCount."),
    randomAccessTrainingGb: z.number().optional().describe("Size (GB) of dataset that needs random access during training (image datasets, shuffled molecule sets, etc.). When provided, plan_gpu_job recommends containerDiskInGb large enough to copy data from NV to rootfs. NV random read is ~18× slower than rootfs (43 vs 775 files/sec), so random-access training MUST run on rootfs. rootfs CANNOT be expanded after pod creation."),
    checkpointBudgetGb: z.number().default(5).describe("Expected checkpoint storage during training (GB). Written to rootfs (/root/outputs/) during training, then rsync'd to NV on completion. Included in containerDiskInGb recommendation. Default 5GB covers typical model checkpoints."),
    outputSpecPath: z.string().optional().describe("If set, write a /gpu-exec pipeline_spec.json stub to this path"),
  },
  safeTool(async (args) => {
    const DEFAULT_MIN_VRAM_GB = 12;
    const NV_COST_PER_GB_MONTH = 0.07;

    const c = requireClient();
    const allGpuTypes = await c.listGpuTypes();

    const minVram = args.modelSizeGb ?? DEFAULT_MIN_VRAM_GB;
    const prefList = args.gpuPreference ?? ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA A40", "NVIDIA RTX A5000"];

    // Filter: on-demand price available + meets VRAM requirement
    const eligible = allGpuTypes.filter(g => getOnDemandPrice(g) != null && g.memoryInGb >= minVram);

    // Sort: prefer gpuPreference order, then ascending price
    eligible.sort((a, b) => {
      const ai = prefList.indexOf(a.displayName);
      const bi = prefList.indexOf(b.displayName);
      if (ai !== -1 && bi !== -1) return ai - bi;
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      return (getOnDemandPrice(a) ?? 999) - (getOnDemandPrice(b) ?? 999);
    });

    const recommended = eligible[0];
    const partialMode = !recommended;

    // NV sizing
    const datasetGb = args.datasetGb ?? 0;
    const estimatedOutputGb = args.modelSizeGb != null ? args.modelSizeGb * 2 : 5;
    const nvRaw = Math.ceil((datasetGb + estimatedOutputGb) * 1.3);
    const nvGb = Math.max(50, nvRaw);
    const stagingNeeded = datasetGb > 0.5;

    // Cost helpers
    const fmtCost = (n: number) => `~$${n.toFixed(2)}`;
    const gpuPrice = recommended ? (getOnDemandPrice(recommended) ?? 0) : 0;
    const cheapest1gpu = eligible.length > 0 ? (getOnDemandPrice(eligible[0]) ?? 0) : 0;

    const stagingHours = stagingNeeded ? Math.max(1, Math.ceil(datasetGb / 50)) : 0;
    const stagingCost = stagingNeeded ? stagingHours * cheapest1gpu : 0;
    const validationCost = gpuPrice * 1;
    const trainingCost = args.expectedHours != null ? gpuPrice * args.expectedHours * args.gpuCount : null;
    const nvCost = nvGb * NV_COST_PER_GB_MONTH;

    // Build output
    const lines: string[] = [];

    lines.push(`## GPU Job Plan: ${args.purpose}`);
    lines.push(``);

    if (partialMode) {
      lines.push(`⚠️ **PARTIAL PLAN** — No on-demand GPU meets VRAM requirement (${minVram}GB).`);
      lines.push(`Try reducing \`modelSizeGb\` or check \`list_gpu_types\` for available options.`);
    } else {
      lines.push(`### Recommended GPU`);
      lines.push(`**${recommended.displayName}** (${recommended.memoryInGb}GB VRAM) × ${args.gpuCount} — $${gpuPrice}/hr each`);
      lines.push(`Reason: cheapest on-demand GPU meeting ${minVram}GB VRAM requirement`);
    }

    lines.push(``);
    lines.push(`### Cost Estimate`);
    lines.push(`| Item | Detail | Cost |`);
    lines.push(`|------|--------|------|`);
    if (stagingNeeded) {
      lines.push(`| Staging pod (data transfer) | ${stagingHours}hr × $${cheapest1gpu.toFixed(2)}/hr × 1 GPU | ${fmtCost(stagingCost)} |`);
    }
    if (!partialMode) {
      lines.push(`| Validation pod (1-GPU test) | 1hr × $${gpuPrice.toFixed(2)}/hr | ${fmtCost(validationCost)} |`);
      lines.push(`| Training pod | ${args.expectedHours != null ? `${args.expectedHours}hr × $${gpuPrice.toFixed(2)}/hr × ${args.gpuCount} GPU` : "expectedHours not provided"} | ${trainingCost != null ? fmtCost(trainingCost) : "N/A"} |`);
    }
    lines.push(`| Network Volume (${nvGb}GB) | $${NV_COST_PER_GB_MONTH}/GB/mo | ${fmtCost(nvCost)}/mo |`);
    if (!partialMode && trainingCost != null) {
      const total = stagingCost + validationCost + trainingCost;
      lines.push(`| **Total (excl. NV)** | | **${fmtCost(total)}** |`);
    }

    lines.push(``);
    lines.push(`### Staging Pattern`);
    if (stagingNeeded) {
      lines.push(`⚠️ Dataset ${datasetGb}GB > 500MB — **staging pod required** to avoid paying GPU rates during upload.`);
      lines.push(`1. \`create_network_volume(${nvGb}GB)\` — ceil((${datasetGb} + ${estimatedOutputGb}) × 1.3) = ${nvRaw}GB → min 50GB`);
      lines.push(`2. \`create_pod_auto(cheapest 1-GPU, networkVolumeId)\` → upload data → delete pod`);
      lines.push(`3. \`create_pod_auto(${recommended?.displayName ?? "target GPU"} × ${args.gpuCount}, networkVolumeId)\` → train → delete pod`);
    } else {
      lines.push(`✅ Dataset ${datasetGb > 0 ? `${datasetGb}GB` : "not specified"} — direct upload on training pod is fine (< 500MB threshold).`);
    }

    lines.push(``);
    lines.push(`### NV Sizing`);
    lines.push(`\`ceil((datasetGb=${datasetGb} + estimatedOutputGb=${estimatedOutputGb}) × 1.3)\` = ${nvRaw}GB → **${nvGb}GB** (min 50GB)`);
    lines.push(`Cost: ~$${(nvGb * NV_COST_PER_GB_MONTH).toFixed(2)}/mo`);

    lines.push(``);
    lines.push(`### rootfs 사이징 (NV → rootfs 복사 필수)`);
    if (args.randomAccessTrainingGb != null && args.randomAccessTrainingGb > 0) {
      const ra = args.randomAccessTrainingGb;
      const recDisk = Math.ceil(ra * 1.3 + 30); // data*1.3 (copy + work) + 30GB system
      lines.push(`⚠️ **랜덤 액세스 훈련 ${ra}GB 감지** — NV는 rootfs보다 ~18× 느림 (43 vs 775 files/sec)`);
      lines.push(`훈련 데이터를 NV에서 rootfs로 복사 후 훈련해야 함. **rootfs는 팟 생성 후 못 늘림**.`);
      lines.push(``);
      lines.push(`**권장**: \`containerDiskInGb=${recDisk}\` (data ${ra}GB × 1.3 + 30GB system overhead)`);
      lines.push(`팟 생성 직후: \`mkdir -p /root/data && cp -r --reflink=auto /workspace/<dataset> /root/data/\``);
      lines.push(`훈련 스크립트는 \`/root/data/<dataset>\` 를 읽도록 설정.`);
      lines.push(`run_preflight 호출 시 \`trainDataPath="/root/data/<dataset>"\`, \`expectedRandomAccessGb=${ra}\` 전달.`);
    } else if (datasetGb >= 50) {
      lines.push(`⚠️ 대형 데이터셋 (${datasetGb}GB). 랜덤 액세스 훈련(이미지, 셔플된 데이터셋 등)이라면:`);
      lines.push(`- NV는 rootfs보다 ~18× 느림 — 데이터를 rootfs로 복사 후 훈련 필요`);
      lines.push(`- rootfs는 팟 생성 후 못 늘림 → \`containerDiskInGb\` 사전 계산 필수`);
      lines.push(`- **랜덤 액세스 훈련이라면** \`randomAccessTrainingGb\` 파라미터를 추가해 정확한 사이즈 권장값을 받으세요`);
      lines.push(`- 순수 sequential 읽기(rare)면 NV 직접 가능 — run_preflight에서 \`allowNvStreaming:true\``);
    } else {
      lines.push(`✅ 데이터셋 ${datasetGb}GB — rootfs 기본값으로 충분. 랜덤 액세스 훈련이라도 작은 데이터는 부담 적음.`);
    }

    lines.push(``);
    lines.push(`### Pre-flight Checklist`);
    lines.push(`[ ] 1. 로컬 전처리 완료 (tokenization, feature extraction 등 GPU 불필요한 작업)`);
    lines.push(`[ ] 2. 학습 코드 로컬 테스트 통과 (import, forward pass, 설정 파일 확인)`);
    lines.push(`[ ] 3. 체크포인트 저장 경로 설정 → **/root/outputs/** (rootfs, 훈련 중 write 빠름)`);
    lines.push(`[ ] 4. NV 크기 확인: ${nvGb}GB 준비`);
    lines.push(`[ ] 5. **rootfs 사이즈 확인** (랜덤 액세스 데이터 + 체크포인트 ${args.checkpointBudgetGb}GB + 시스템 오버헤드 — 생성 후 못 늘림)`);
    if (args.gpuCount >= COST_GATE_GPU_COUNT) {
      lines.push(`[ ] 6. **1-GPU 검증 테스트 완료** (gpuCount=${args.gpuCount} — 고비용 팟 전 필수)`);
    }
    lines.push(``);
    lines.push(`### Post-Training (아티팩트 보존)`);
    lines.push(`[ ] 훈련 완료 후 (NV 있음): \`rsync -a /root/outputs/ /workspace/outputs/ && echo RSYNC_OK\` → RSYNC_OK 확인 → \`delete_pod(artifactsSavedConfirmed: true)\``);
    lines.push(`[ ] 훈련 완료 후 (NV 없음): \`download_files\` → \`delete_pod(artifactsSavedConfirmed: true)\``);

    lines.push(``);
    // Seed parallelization section
    if (args.seedCount > 1) {
      const totalPods = args.seedCount * args.armCount;
      const defaultSeeds = [42, 123, 456, 789, 999];
      const seeds = defaultSeeds.slice(0, args.seedCount);

      lines.push(``);
      lines.push(`### ⚠️ Seed 병렬화 필수 (seedCount=${args.seedCount})`);
      lines.push(`**순차 실행 금지 — 같은 비용에 ${args.seedCount}배 느려짐.**`);
      lines.push(`arm당 seed ${args.seedCount}개는 ${args.seedCount}팟 동시 생성. DDP 불필요, 단순 병렬.`);
      lines.push(``);

      // Sequential vs parallel comparison
      lines.push(`| 방식 | 소요 시간 | 비용 |`);
      lines.push(`|------|----------|------|`);
      if (args.expectedHours != null) {
        const seqHours = args.expectedHours * args.seedCount;
        const parHours = args.expectedHours;
        const seqCost = gpuPrice * seqHours * args.gpuCount;
        const parCost = gpuPrice * parHours * args.gpuCount * args.seedCount;
        lines.push(`| 순차 실행 (${args.seedCount} seed × 1팟) | ${seqHours}hr | ${fmtCost(seqCost)} |`);
        lines.push(`| **병렬 실행 (${args.seedCount}팟 동시)** | **${parHours}hr** | **${fmtCost(parCost)}** |`);
      } else {
        lines.push(`| 순차 실행 | ${args.seedCount}× 소요 시간 | 동일 비용 |`);
        lines.push(`| **병렬 실행** | **1× 소요 시간** | **동일 비용** |`);
      }

      lines.push(``);
      lines.push(`**총 팟 수:** ${totalPods}개 (${args.armCount}arm × ${args.seedCount}seed)`);
      lines.push(``);

      // Per-seed pod creation pattern
      lines.push(`**팟 생성 패턴${args.armCount > 1 ? ` (arm 1개 기준, × ${args.armCount} 반복)` : ""}:**`);
      seeds.forEach((seed, i) => {
        lines.push(`\`create_pod_auto({ env: { SEED: "${seed}" }, ... })  # seed ${i + 1}/${args.seedCount}\``);
      });

      if (args.armCount > 1) {
        lines.push(``);
        lines.push(`> arm이 ${args.armCount}개이면 위 패턴을 arm별로 반복 → 총 **${totalPods}팟 동시** 실행`);
      }
    }

    // Container disk warning
    const DEFAULT_CONTAINER_DISK_GB = 30;
    const diskEstimateGb = ((args.modelSizeGb ?? 5) + datasetGb * 0.1 + args.checkpointBudgetGb + 2) * args.gpuCount;
    const diskThreshold = DEFAULT_CONTAINER_DISK_GB * 0.7;
    lines.push(``);
    lines.push(`### Container Disk`);
    if (diskEstimateGb > diskThreshold) {
      const recommendedDisk = Math.ceil(diskEstimateGb * 1.5);
      lines.push(`⚠️ **Container disk warning**: estimated experiment output ~${diskEstimateGb.toFixed(1)}GB (model + tmp + checkpoints + logs × gpuCount=${args.gpuCount}) exceeds 70% of RunPod default ${DEFAULT_CONTAINER_DISK_GB}GB disk.`);
      lines.push(`→ Set \`containerDiskInGb: ${recommendedDisk}\` in create_pod_auto`);
      lines.push(`→ Formula: (modelSizeGb=${args.modelSizeGb ?? 5} + datasetGb×0.1=${(datasetGb * 0.1).toFixed(1)} + checkpointBudgetGb=${args.checkpointBudgetGb} + 2) × gpuCount=${args.gpuCount} = ${diskEstimateGb.toFixed(1)}GB`);
    } else {
      lines.push(`✅ Estimated disk usage ~${diskEstimateGb.toFixed(1)}GB — within 70% of default ${DEFAULT_CONTAINER_DISK_GB}GB container disk.`);
    }

    lines.push(``);
    lines.push(`### Next Steps`);
    lines.push(`1. 체크리스트 완료 후: \`create_pod_auto(dryRun: true, gpuCount: ${args.gpuCount}, ...)\` 로 GPU 선택 재확인`);
    if (args.outputSpecPath) {
      lines.push(`2. /gpu-exec 파이프라인: \`pipeline_spec.json\` stub → \`${args.outputSpecPath}\``);
    } else {
      lines.push(`2. /gpu-exec 파이프라인이 필요하면 \`outputSpecPath\` 파라미터로 \`pipeline_spec.json\` stub 생성 가능`);
    }

    // Write pipeline_spec stub if requested
    if (args.outputSpecPath) {
      const specPath = isAbsolute(args.outputSpecPath) ? args.outputSpecPath : resolve(process.cwd(), args.outputSpecPath);
      const stub = {
        pipeline_id: `plan-${Date.now()}`,
        mode: "runpod",
        gpu: recommended?.displayName ?? "FILL_IN",
        gpu_count: args.gpuCount,
        network_volume_gb: nvGb,
        phases: [
          ...(stagingNeeded ? [{
            id: "upload",
            purpose: "Data transfer",
            gpu: "cheapest-1gpu",
            steps: [`Upload dataset (${datasetGb}GB) to /workspace/data/`]
          }] : []),
          {
            id: "train",
            purpose: args.purpose,
            gpu: recommended?.displayName ?? "FILL_IN",
            gpu_count: args.gpuCount,
            steps: ["Run training script", "Save checkpoint to /workspace/checkpoints/"],
            gate: "FILL_IN: e.g. {\"metric\": \"val_loss\", \"threshold\": 0.5, \"op\": \"<\"}"
          }
        ]
      };
      try {
        await mkdir(dirname(specPath), { recursive: true });
        await writeFile(specPath, JSON.stringify(stub, null, 2), "utf-8");
        lines.push(`\n✅ pipeline_spec.json stub 생성됨: \`${specPath}\``);
      } catch (e) {
        lines.push(`\n⚠️ pipeline_spec.json 저장 실패: ${e instanceof Error ? e.message : String(e)}`);
      }
    }

    return text(lines.join("\n"));
  })
);

// ══════════════════════════════════════════
//  COST SAFETY PHASE 1 TOOLS
// ══════════════════════════════════════════

const NV_READY_DIR = ".omc/gpu-exec";
const TOKEN_TTL_HOURS = 72;

// ── verify_data_on_nv ──
server.tool(
  "verify_data_on_nv",
  "Verify that a dataset has been successfully transferred to a Network Volume by " +
  "SSHing to a mounted staging pod and checking file existence + sizes. Returns a " +
  "readiness token (valid 72h) that create_pod_auto requires when gpuCount >= 2 + networkVolumeId. " +
  "Call this BEFORE deleting the staging pod — token requires pod to still be RUNNING.",
  {
    podId: z.string().describe("ID of the RUNNING staging pod with the NV mounted"),
    requiredPaths: z.array(z.string()).describe(
      "Paths to verify on /workspace/ (e.g. ['data/train.jsonl', 'data/val.jsonl'])"
    ),
    minTotalGb: z.number().optional().describe(
      "Minimum total size in GB across all paths. Fails if smaller (catches truncation)."
    ),
  },
  safeTool(async ({ podId, requiredPaths, minTotalGb }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    if (!pod) return text(`❌ Pod ${podId} not found. verify_data_on_nv requires the staging pod to still be RUNNING. Call this tool BEFORE deleting the staging pod.`);

    const nvId = pod.networkVolumeId;
    if (!nvId) return text(`❌ Pod ${podId} has no Network Volume attached. Cannot issue NV readiness token.`);

    const sshArgs = c.getSshArgs(pod);
    if (!sshArgs) return text(`❌ Pod ${podId} is not ready for SSH. Run wait_for_pod first.`);

    const lines: string[] = [`## verify_data_on_nv — ${podId}`];
    let totalBytes = 0;
    const pathResults: string[] = [];

    for (const p of requiredPaths) {
      const cmd = `ls -la /workspace/${p} 2>/dev/null && du -sb /workspace/${p} 2>/dev/null | awk '{print $1}'`;
      const result = await spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", cmd], { timeout: 30_000 });
      if (result.status !== 0 || !result.stdout.trim()) {
        pathResults.push(`❌ /workspace/${p} — not found or inaccessible`);
      } else {
        const sizeMatch = result.stdout.match(/(\d+)\s*$/m);
        const bytes = sizeMatch ? parseInt(sizeMatch[1], 10) : 0;
        totalBytes += bytes;
        const sizeGb = (bytes / 1073741824).toFixed(2);
        pathResults.push(`✅ /workspace/${p} — ${sizeGb}GB`);
      }
    }

    const failedPaths = pathResults.filter(r => r.startsWith("❌"));
    if (failedPaths.length > 0) {
      lines.push(...pathResults);
      return text(lines.join("\n") + "\n\n❌ Verification FAILED — missing files detected.");
    }

    const totalGb = totalBytes / 1073741824;
    if (minTotalGb != null && totalGb < minTotalGb) {
      lines.push(...pathResults);
      lines.push(`\n❌ Data truncation detected: expected ≥${minTotalGb}GB, found ${totalGb.toFixed(2)}GB`);
      return text(lines.join("\n"));
    }

    // Write token
    const token = randomUUID();
    const tokenData = { token, nvId, podId, verifiedAt: new Date().toISOString(), totalGb: parseFloat(totalGb.toFixed(3)), paths: requiredPaths };
    await mkdir(NV_READY_DIR, { recursive: true });
    await writeFile(`${NV_READY_DIR}/nv_ready_${nvId}.json`, JSON.stringify(tokenData, null, 2), "utf-8");

    lines.push(...pathResults);
    lines.push(`\n✅ NV ${nvId} verified: ${totalGb.toFixed(2)}GB across ${requiredPaths.length} paths.`);
    lines.push(`Token valid 72h. Pass to create_pod_auto as nvReadinessToken.`);
    lines.push(`Token: ${token}`);
    return text(lines.join("\n"));
  })
);

// ── run_preflight ──
server.tool(
  "run_preflight",
  "Run pre-flight checks on a RUNNING pod before starting an expensive experiment. " +
  "Checks disk space, requirements.txt pinning, file existence, system tools, and Python import smoke tests. " +
  "Based on real incident postmortem: 6/8 bugs catchable in <5 min with this tool. " +
  "Call after pod setup, before launching training.",
  {
    podId: z.string().describe("Pod ID to SSH into for checks"),
    requirementsPath: z.string().optional().describe("Local path to requirements.txt — checks ML-critical package pinning"),
    requiredFiles: z.array(z.string()).optional().describe("Paths on /workspace/ that must exist (model files, data, configs)"),
    requiredTools: z.array(z.string()).optional().describe("System tools that must be in PATH on pod (e.g. ['cpptraj', 'gmx_MMPBSA'])"),
    importSmokes: z.array(z.string()).optional().describe("Python import statements to test on pod (e.g. ['from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer'])"),
    minDiskFreeGb: z.number().default(10).describe("Minimum free disk space on /workspace/ in GB"),
    strict: z.boolean().default(false).describe("If true, treat any WARNING as FAIL"),
    trainDataPath: z.string().optional().describe("Absolute path on pod where the training script READS data from (e.g. '/workspace/dataset' or '/root/data'). REQUIRED for GPU pods with NV attached unless allowNvStreaming:true. NV random read is ~18x slower than rootfs (43 vs 775 files/sec). Verifies data has been copied from NV to rootfs."),
    expectedRandomAccessGb: z.number().optional().describe("Size (GB) of dataset that needs random access during training. Used to verify rootfs has enough free space. rootfs CANNOT be expanded after pod creation — must be sized at creation time via containerDiskInGb."),
    allowNvStreaming: z.boolean().default(false).describe("Opt-out: skip the NV→rootfs migration HALT. Only set true if training does PURELY sequential reads (rare for ML; image/molecule/text shuffled training all need random access)."),
  },
  safeTool(async ({ podId, requirementsPath, requiredFiles, requiredTools, importSmokes, minDiskFreeGb, strict, trainDataPath, expectedRandomAccessGb, allowNvStreaming }) => {
    const c = requireClient();
    const pod = await c.getPod(podId);
    if (!pod) return text(`❌ Pod ${podId} not found.`);
    const sshArgs = c.getSshArgs(pod);
    if (!sshArgs) return text(`❌ Pod ${podId} not ready for SSH. Run wait_for_pod first.`);

    const CRITICAL_ML = ["peft", "transformers", "torch", "torchaudio", "torchvision", "bitsandbytes", "accelerate", "datasets"];
    const results: Array<{ label: string; status: "✅" | "⚠️" | "❌"; detail: string }> = [];
    let hasFail = false;
    let hasWarn = false;

    // 0. CUDA availability check (always runs — bash -c safe, base64 encoded)
    {
      const cudaScript = [
        "import subprocess, torch",
        "ok = torch.cuda.is_available()",
        "if ok:",
        "  try:",
        "    drv = subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()",
        "  except FileNotFoundError:",
        "    drv = 'nvidia-smi-missing'",
        "  print(f'CUDA:OK cuda_build={torch.version.cuda} driver={drv} torch={torch.__version__}')",
        "else:",
        "  print(f'CUDA:FAIL torch={torch.__version__} cuda_build={torch.version.cuda}')",
      ].join("\n");
      const cudaB64 = Buffer.from(cudaScript).toString("base64");
      const cudaCheckCmd = `bash -c 'echo ${cudaB64} | base64 -d | python3'`;
      const cudaResult = await spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", cudaCheckCmd], { timeout: 60_000 });

      if (cudaResult.status === null) {
        results.push({ label: "CUDA", status: "⚠️", detail: "CUDA check timed out (60s) — pod may be cold-starting. Re-run run_preflight." });
        hasWarn = true;
      } else {
        const lines = (cudaResult.stdout ?? "").split("\n");
        const cudaLine = lines.find(l => l.startsWith("CUDA:OK") || l.startsWith("CUDA:FAIL")) ?? "";
        const stderrFull = cudaResult.stderr ?? "";
        const isModuleErr = stderrFull.includes("ModuleNotFoundError") || stderrFull.includes("No module named");
        const stderrDisplay = stderrFull.substring(0, 200);

        if (cudaLine.startsWith("CUDA:FAIL")) {
          results.push({ label: "CUDA", status: "❌", detail: cudaLine });
          hasFail = true;
        } else if (cudaResult.status !== 0 || !cudaLine) {
          const detail = isModuleErr
            ? `torch not installed: ${stderrDisplay}`
            : `check failed (exit ${cudaResult.status}): ${stderrDisplay || cudaLine}`;
          results.push({ label: "CUDA", status: "❌", detail });
          hasFail = true;
        } else {
          results.push({ label: "CUDA", status: "✅", detail: cudaLine.replace("CUDA:OK ", "") });
        }
      }
    }

    // 0.5. NV → rootfs migration check (HALT if data on NV)
    // Background: NV random read ~43 files/sec, rootfs (NVMe) ~775 files/sec — 18x slower.
    // For random-access training (image/molecule/shuffled datasets), data MUST be on rootfs.
    // rootfs CANNOT be expanded after pod creation — must be sized at creation via containerDiskInGb.
    {
      // Defense-in-depth path sanitization (base64 wrap below also prevents shell injection)
      const trainPathSafe = (trainDataPath ?? "").replace(/['\\\n\r]/g, "");
      const inspectShell = [
        `R=$(stat -c %m / 2>/dev/null || echo '/')`,
        `W=$(stat -c %m /workspace 2>/dev/null || echo 'MISSING')`,
        trainPathSafe ? `T=$(stat -c %m '${trainPathSafe}' 2>/dev/null || echo 'MISSING')` : `T=''`,
        `FREE=$(df -BG / 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}' || echo '0')`,
        `TOTAL=$(df -BG / 2>/dev/null | awk 'NR==2 {gsub("G","",$2); print $2}' || echo '0')`,
        trainPathSafe ? `DATA=$(du -sBG '${trainPathSafe}' 2>/dev/null | awk '{gsub("G","",$1); print $1}' || echo '0')` : `DATA=''`,
        `echo "R=$R|W=$W|T=$T|FREE=$FREE|TOTAL=$TOTAL|DATA=$DATA"`,
      ].join(";");
      const inspectB64 = Buffer.from(inspectShell).toString("base64");
      const inspectCmd = `bash -c 'echo ${inspectB64} | base64 -d | bash'`;
      const inspectResult = await spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", inspectCmd], { timeout: 30_000 });

      if (inspectResult.status === null) {
        results.push({ label: "NV→rootfs", status: "⚠️", detail: "Mount inspection timed out (30s) — pod may be cold-starting. Re-run run_preflight." });
        hasWarn = true;
      } else if (inspectResult.status !== 0) {
        results.push({ label: "NV→rootfs", status: "⚠️", detail: `Mount inspection failed (exit ${inspectResult.status}) — skipping NV check. stderr: ${(inspectResult.stderr ?? "").substring(0, 150)}` });
        hasWarn = true;
      } else {
        // Parse: R=<mount>|W=<mount>|T=<mount>|FREE=<gb>|TOTAL=<gb>|DATA=<gb>
        const lastLine = (inspectResult.stdout ?? "").trim().split("\n").pop() ?? "";
        const parts: Record<string, string> = {};
        for (const kv of lastLine.split("|")) {
          const eq = kv.indexOf("=");
          if (eq > 0) parts[kv.substring(0, eq)] = kv.substring(eq + 1);
        }
        const rootMount = parts.R || "/";
        const workspaceMount = parts.W || "MISSING";
        const trainMount = parts.T || "";
        const rootfsFreeGb = parseInt(parts.FREE || "0", 10) || 0;
        const rootfsTotalGb = parseInt(parts.TOTAL || "0", 10) || 0;
        const trainDataGb = parseInt(parts.DATA || "0", 10) || 0;
        const nvAttached = workspaceMount !== "MISSING" && workspaceMount !== rootMount;

        // ── Decision tree ──────────────────────────────────────────
        if (!nvAttached) {
          results.push({ label: "NV→rootfs", status: "✅", detail: `No NV attached — rootfs only (${rootfsFreeGb}/${rootfsTotalGb}GB free)` });
        } else if (!trainDataPath && !allowNvStreaming) {
          // HALT: Pod has NV, no path specified, no opt-out → require user input
          results.push({
            label: "NV→rootfs",
            status: "❌",
            detail:
              `HALT — NV attached at ${workspaceMount} but trainDataPath not specified. ` +
              `NV random read ~18x slower than rootfs (43 vs 775 files/sec). ` +
              `ASK USER: (1) which path will the training script READ data from? ` +
              `(2) dataset size in GB needing random access? ` +
              `Then re-run with trainDataPath="<path>" and expectedRandomAccessGb=<N>. ` +
              `Opt-out (rare, sequential reads only): allowNvStreaming:true. ` +
              `Note: rootfs has ${rootfsFreeGb}/${rootfsTotalGb}GB free — cannot grow after pod creation.`,
          });
          hasFail = true;
        } else if (!trainDataPath && allowNvStreaming) {
          results.push({
            label: "NV→rootfs",
            status: "⚠️",
            detail: `NV streaming opted in (allowNvStreaming:true). Random reads will be ~18x slower than rootfs. Verify training pattern is sequential.`,
          });
          hasWarn = true;
        } else if (trainMount === "MISSING") {
          results.push({
            label: "NV→rootfs",
            status: "❌",
            detail: `trainDataPath does not exist on pod: ${trainDataPath}. Check the path or upload data first.`,
          });
          hasFail = true;
        } else if (trainMount !== rootMount) {
          // Data on NV — fail with copy recommendation
          const dataInfo = trainDataGb > 0 ? `${trainDataGb}GB` : "unknown size";
          const targetPath = `/root/data`;
          const recCmd = `mkdir -p ${targetPath} && cp -r --reflink=auto ${trainDataPath} ${targetPath}/`;
          const sufficient = trainDataGb === 0 || rootfsFreeGb >= trainDataGb + 5;
          if (!sufficient) {
            const recDisk = Math.ceil(trainDataGb * 1.3 + 30);
            results.push({
              label: "NV→rootfs",
              status: "❌",
              detail:
                `Data on NV (mount=${trainMount}, ${dataInfo}) AND rootfs too small (${rootfsFreeGb}GB free, need ≥${trainDataGb + 5}GB). ` +
                `rootfs CANNOT be grown on running pod — recreate pod with containerDiskInGb≥${recDisk}.`,
            });
          } else {
            results.push({
              label: "NV→rootfs",
              status: "❌",
              detail:
                `Data on NV (mount=${trainMount}, ${dataInfo}) — must copy to rootfs first. ` +
                `NV random read ~18x slower than rootfs. ` +
                `Run on pod: ${recCmd}. ` +
                `Then re-point training script to ${targetPath}/<basename> and re-run preflight.`,
            });
          }
          hasFail = true;
        } else if (expectedRandomAccessGb != null && rootfsFreeGb < expectedRandomAccessGb + 10) {
          // Data on rootfs but rootfs too small for the planned data
          const recDisk = Math.ceil(expectedRandomAccessGb * 1.3 + 30);
          results.push({
            label: "NV→rootfs",
            status: "❌",
            detail:
              `Data on rootfs ✓ but rootfs has only ${rootfsFreeGb}GB free, need ≥${expectedRandomAccessGb + 10}GB ` +
              `(expectedRandomAccessGb=${expectedRandomAccessGb} + 10GB headroom). ` +
              `rootfs CANNOT be grown on running pod — recreate with containerDiskInGb≥${recDisk}.`,
          });
          hasFail = true;
        } else {
          const sizeNote = expectedRandomAccessGb != null
            ? ` (rootfs ${rootfsFreeGb}/${rootfsTotalGb}GB free, expected ${expectedRandomAccessGb}GB)`
            : ` (rootfs ${rootfsFreeGb}/${rootfsTotalGb}GB free)`;
          results.push({
            label: "NV→rootfs",
            status: "✅",
            detail: `Data on rootfs (mount=${trainMount})${sizeNote}`,
          });
        }
      }
    }

    // 1. Disk check
    const dfResult = await spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", `df -BG /workspace | awk 'NR==2{print $4}' | tr -d G`], { timeout: 15_000 });
    const freeGb = dfResult.status === 0 ? parseInt(dfResult.stdout.trim(), 10) : -1;
    if (freeGb < 0) {
      results.push({ label: "Disk free", status: "❌", detail: "Could not query disk space" });
      hasFail = true;
    } else if (freeGb < minDiskFreeGb) {
      results.push({ label: "Disk free", status: "❌", detail: `Only ${freeGb}GB free — need ≥${minDiskFreeGb}GB` });
      hasFail = true;
    } else {
      results.push({ label: "Disk free", status: "✅", detail: `${freeGb}GB free (min: ${minDiskFreeGb}GB)` });
    }

    // 2. requirements.txt pinning check (local file)
    if (requirementsPath) {
      try {
        const content = await readFile(requirementsPath, "utf-8");
        const lines = content.split("\n").filter(l => l.trim() && !l.trim().startsWith("#"));
        const issues: string[] = [];
        for (const line of lines) {
          const match = line.match(/^([a-zA-Z0-9_-]+)(.*)$/);
          if (!match) continue;
          const pkg = match[1].toLowerCase();
          if (!CRITICAL_ML.includes(pkg)) continue;
          const spec = match[2].trim();
          if (!spec) {
            issues.push(`${pkg} (no version spec)`);
          } else if (/^>=/.test(spec) && !spec.includes(",<")) {
            issues.push(`${pkg}${spec} (unbounded — no upper bound)`);
          }
        }
        if (issues.length > 0) {
          results.push({ label: "requirements.txt", status: "⚠️", detail: `Unbounded: ${issues.join(", ")}` });
          hasWarn = true;
        } else {
          results.push({ label: "requirements.txt", status: "✅", detail: "All critical ML packages pinned" });
        }
      } catch {
        results.push({ label: "requirements.txt", status: "❌", detail: `File not found: ${requirementsPath}` });
        hasFail = true;
      }
    }

    // 3. File existence on pod
    if (requiredFiles && requiredFiles.length > 0) {
      const checkCmd = requiredFiles.map(f => `ls /workspace/${f} 2>/dev/null && echo "OK:${f}" || echo "MISSING:${f}"`).join("; ");
      const fileResult = await spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", checkCmd], { timeout: 20_000 });
      const out = fileResult.stdout;
      const missing = requiredFiles.filter(f => out.includes(`MISSING:${f}`));
      if (missing.length > 0) {
        results.push({ label: "Required files", status: "❌", detail: `Missing: ${missing.join(", ")}` });
        hasFail = true;
      } else {
        results.push({ label: "Required files", status: "✅", detail: `${requiredFiles.length} file(s) present` });
      }
    }

    // 4. System tools
    if (requiredTools && requiredTools.length > 0) {
      const toolCmd = requiredTools.map(t => `which ${t} 2>/dev/null && echo "FOUND:${t}" || echo "MISSING:${t}"`).join("; ");
      const toolResult = await spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", toolCmd], { timeout: 15_000 });
      const out = toolResult.stdout;
      const missing = requiredTools.filter(t => out.includes(`MISSING:${t}`));
      if (missing.length > 0) {
        results.push({ label: "System tools", status: "❌", detail: `Not in PATH: ${missing.join(", ")}` });
        hasFail = true;
      } else {
        results.push({ label: "System tools", status: "✅", detail: requiredTools.join(", ") });
      }
    }

    // 5. Python import smoke tests
    if (importSmokes && importSmokes.length > 0) {
      for (const imp of importSmokes) {
        const smokeCmd = `python3 -c "${imp.replace(/"/g, '\\"')}" 2>&1 && echo "__IMPORT_OK__" || echo "__IMPORT_FAIL__"`;
        const smokeResult = await spawnAsync(sshArgs[0], [...sshArgs.slice(1), "--", smokeCmd], { timeout: 30_000 });
        if (smokeResult.stdout.includes("__IMPORT_OK__")) {
          results.push({ label: `Import: ${imp.split(" ")[1]}`, status: "✅", detail: "OK" });
        } else {
          const errLine = smokeResult.stdout.split("\n").find(l => l.includes("Error") || l.includes("error")) ?? "import failed";
          results.push({ label: `Import: ${imp.split(" ")[1]}`, status: "❌", detail: errLine.trim() });
          hasFail = true;
        }
      }
    }

    const lines: string[] = [`## Pre-flight Results — ${podId}`];
    for (const r of results) {
      lines.push(`${r.status} ${r.label}: ${r.detail}`);
    }
    lines.push("─".repeat(50));

    const totalChecks = results.length;
    const passed = results.filter(r => r.status === "✅").length;
    const failed = results.filter(r => r.status === "❌").length;
    const warned = results.filter(r => r.status === "⚠️").length;

    const overallFail = hasFail || (strict && hasWarn);
    lines.push(`RESULT: ${overallFail ? "❌ FAIL" : "✅ PASS"} (${passed}/${totalChecks} checks passed${warned > 0 ? `, ${warned} warning(s)` : ""}, ${failed} failure(s))`);
    return text(lines.join("\n"));
  })
);

// ── watch_running_pods ──
server.tool(
  "watch_running_pods",
  "Launch a background bash watcher (scripts/pod_watcher.sh) that polls specific RUNNING pods every N minutes via SSH. " +
  "Auto-stops pods with GPU compute utilization < idleThresholdPct for consecutive checks. " +
  "Writes events to .omc/gpu-exec/events.jsonl. PID saved to .omc/gpu-exec/watcher.pid. " +
  "Call stop_watching_pods() to stop. Call get_pipeline_events() to read status.",
  {
    podIds: z.array(z.string()).min(1).describe("Pod IDs to watch (required — at least 1)"),
    intervalMinutes: z.number().default(5).describe("Poll interval in minutes"),
    idleThresholdPct: z.number().default(20).describe("GPU compute utilization % below which pod is considered idle"),
    idleConsecutiveChecks: z.number().default(2).describe("Consecutive idle checks before auto-stop"),
    mode: z.enum(["full", "error-only"]).default("full").describe(
      "full: log HEALTH_CHECK event every interval. " +
      "error-only: only log IDLE_WARNING/AUTO_STOPPED/ERROR/WATCHER_EXITED — reduces noise for long runs (23+ hours)."
    ),
    expectedCompletionAt: z.string().optional().describe(
      "ISO8601 timestamp of expected training completion. " +
      "Watcher switches to 1-minute check interval in the 30 minutes before this time."
    ),
  },
  safeTool(async ({ podIds, intervalMinutes, idleThresholdPct, idleConsecutiveChecks, mode, expectedCompletionAt }) => {
    const pidFile = `${NV_READY_DIR}/watcher.pid`;

    // Check for existing watcher
    try {
      const existingPid = (await readFile(pidFile, "utf-8")).trim();
      const checkResult = await spawnAsync("kill", ["-0", existingPid], { timeout: 5_000 });
      if (checkResult.status === 0) {
        return text(`❌ Watcher already running (PID ${existingPid}). Call stop_watching_pods() first.`);
      }
    } catch {
      // No pid file or process not found — OK to proceed
    }

    await mkdir(NV_READY_DIR, { recursive: true });

    const scriptPath = `${process.cwd()}/scripts/pod_watcher.sh`;
    const args = [
      "--pods", podIds.join(","),
      "--interval", String(intervalMinutes),
      "--idle-pct", String(idleThresholdPct),
      "--idle-checks", String(idleConsecutiveChecks),
      "--mode", mode,
    ];
    if (expectedCompletionAt) args.push("--expected-completion", expectedCompletionAt);

    const eventsFile = `${NV_READY_DIR}/events.jsonl`;
    // Spawn detached (nohup-style)
    const spawn = await spawnAsync(
      "bash",
      ["-c", `nohup bash ${scriptPath} ${args.join(" ")} >> ${eventsFile} 2>&1 & echo $!`],
      { timeout: 5_000 }
    );

    if (spawn.status !== 0) {
      return text(`❌ Failed to start watcher: ${spawn.stderr}`);
    }

    const pid = spawn.stdout.trim();
    await writeFile(pidFile, pid, "utf-8");

    const completionNote = expectedCompletionAt ? ` Expected completion: ${expectedCompletionAt}.` : "";
    return text(
      `✅ Watcher started (PID ${pid}) for pods: [${podIds.join(", ")}]. Mode: ${mode}.${completionNote}\n` +
      `Events → ${eventsFile}\n` +
      `Call get_pipeline_events() for updates, stop_watching_pods() to stop.`
    );
  })
);

// ── stop_watching_pods ──
server.tool(
  "stop_watching_pods",
  "Stop the background pod watcher launched by watch_running_pods.",
  {},
  safeTool(async () => {
    const pidFile = `${NV_READY_DIR}/watcher.pid`;
    let pid: string;
    try {
      pid = (await readFile(pidFile, "utf-8")).trim();
    } catch {
      return text("No watcher running (PID file not found).");
    }

    // kill + waitpid(5s) + SIGKILL fallback
    await spawnAsync("kill", [pid], { timeout: 5_000 });

    // Wait up to 5s for process to exit
    let exited = false;
    for (let i = 0; i < 10; i++) {
      await new Promise(r => setTimeout(r, 500));
      const check = await spawnAsync("kill", ["-0", pid], { timeout: 2_000 });
      if (check.status !== 0) { exited = true; break; }
    }

    if (!exited) {
      await spawnAsync("kill", ["-9", pid], { timeout: 5_000 });
    }

    try { await writeFile(pidFile, "", "utf-8"); } catch { /* ignore */ }
    // Remove pid file
    await spawnAsync("rm", ["-f", pidFile], { timeout: 5_000 });

    return text(`Watcher stopped (PID ${pid}).`);
  })
);

// ── get_pipeline_events ──
server.tool(
  "get_pipeline_events",
  "Read and summarize events from the background pod watcher. Returns per-pod GPU utilization trend, " +
  "auto-stop events, and cost warnings. Highlights WATCHER_EXITED with manual stop instructions. " +
  "If WATCHER_EXITED is detected, repeats warning on every call (sticky) until watcher is restarted.",
  {
    podId: z.string().optional().describe("Filter events to a specific pod ID (omit for all pods)"),
    tail: z.number().default(50).describe("Last N events to return"),
  },
  safeTool(async ({ podId, tail }) => {
    const eventsFile = `${NV_READY_DIR}/events.jsonl`;
    let raw: string;
    try {
      raw = await readFile(eventsFile, "utf-8");
    } catch {
      return text("No events file found. Start a watcher with watch_running_pods() first.");
    }

    const allLines = raw.split("\n").filter(l => l.trim());
    const filtered = podId
      ? allLines.filter(l => {
          try { return JSON.parse(l).podId === podId; } catch { return false; }
        })
      : allLines;

    const recent = filtered.slice(-tail);
    const parsed = recent.map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);

    // Check for WATCHER_EXITED (sticky warning)
    const exitedEvents = parsed.filter((e: any) => e.event === "WATCHER_EXITED");
    const lastExited = exitedEvents[exitedEvents.length - 1] as any;

    const lines: string[] = [`## Pipeline Events${podId ? ` — ${podId}` : ""} (last ${tail})`];

    if (lastExited) {
      lines.push(`\n🚨 **WATCHER EXITED** — Pod ${lastExited.podId ?? "unknown"} may still be running and accruing cost.`);
      lines.push(`   Reason: ${lastExited.reason ?? "GraphQL stop failure"}`);
      lines.push(`   → Manual action required: call delete_pod("${lastExited.podId ?? "<podId>"}") or stop_pod() immediately.`);
      lines.push(``);
    }

    // Format events
    for (const e of parsed as any[]) {
      const ts = e.ts ?? "";
      const event = e.event ?? "UNKNOWN";
      const pod = e.podId ?? "";
      const gpu = e.gpuPct != null ? ` GPU:${e.gpuPct}%` : "";
      const idle = e.idleCheck != null ? ` idle:${e.idleCheck}` : "";
      const detail = e.detail ? ` — ${e.detail}` : "";
      lines.push(`${ts} [${event}]${pod ? ` pod:${pod}` : ""}${gpu}${idle}${detail}`);
    }

    if (parsed.length === 0) {
      lines.push("(no events)");
    }

    return text(lines.join("\n"));
  })
);

// ══════════════════════════════════════════
//  START
// ══════════════════════════════════════════

const transport = new StdioServerTransport();
await server.connect(transport);
