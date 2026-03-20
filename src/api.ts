import { createConnection } from "node:net";
import type { Pod, GpuType, CreatePodOptions, RunPodApiConfig, NetworkVolume } from "./types.js";

/** Normalize portMappings keys: "22/tcp" → "22", "22" → "22" */
function normalizePorts(pm: Record<string, number> | undefined): Record<string, number> | undefined {
  if (!pm) return pm;
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(pm)) {
    out[k.split("/")[0]] = v;
  }
  return out;
}

export class RunPodClient {
  private config: RunPodApiConfig;

  constructor(config: RunPodApiConfig) {
    this.config = config;
  }

  private async restRequest<T>(method: string, path: string, body?: unknown): Promise<T> {
    const res = await fetch(`${this.config.restBaseUrl}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${this.config.apiKey}`,
        "Content-Type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`RunPod REST API ${method} ${path}: ${res.status} ${text}`);
    }
    return res.json() as Promise<T>;
  }

  private async graphqlRequest<T>(query: string, variables?: Record<string, unknown>): Promise<T> {
    const res = await fetch(this.config.graphqlUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({ query, variables }),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`RunPod GraphQL API: ${res.status} ${text}`);
    }
    const json = (await res.json()) as { data?: T; errors?: Array<{ message: string }> };
    if (json.errors?.length) {
      if (json.data) {
        // Partial success: data returned with some field errors (e.g. lowestPrice)
        // Log warning but return available data
        const uniqueMessages = [...new Set(json.errors.map((e) => e.message))];
        process.stderr.write(`RunPod GraphQL partial error (data still returned): ${uniqueMessages.join("; ")}\n`);
      } else {
        throw new Error(`RunPod GraphQL: ${json.errors.map((e) => e.message).join(", ")}`);
      }
    }
    return json.data as T;
  }

  /** Normalize pod portMappings on read */
  private normalizePod(pod: Pod): Pod {
    return { ...pod, portMappings: normalizePorts(pod.portMappings) };
  }

  // ── Pod CRUD ──

  async listPods(): Promise<Pod[]> {
    const pods = await this.restRequest<Pod[]>("GET", "/pods");
    return pods.map((p) => this.normalizePod(p));
  }

  async getPod(podId: string): Promise<Pod> {
    const pod = await this.restRequest<Pod>("GET", `/pods/${podId}`);
    return this.normalizePod(pod);
  }

  async createPod(opts: CreatePodOptions): Promise<Pod> {
    const body: Record<string, unknown> = {
      name: opts.name,
      imageName: opts.imageName,
      gpuTypeIds: opts.gpuTypeIds,
      gpuCount: opts.gpuCount ?? 1,
      interruptible: opts.interruptible ?? false,
      containerDiskInGb: opts.containerDiskInGb ?? 50,
      volumeInGb: opts.volumeInGb ?? 20,
      volumeMountPath: opts.volumeMountPath ?? "/workspace",
      ports: opts.ports ?? ["22/tcp"],
      supportPublicIp: opts.supportPublicIp ?? true,
      env: { ...opts.env },
    };

    if (opts.sshPublicKey) {
      (body.env as Record<string, string>).SSH_PUBLIC_KEY = opts.sshPublicKey;
    }
    if (opts.networkVolumeId) body.networkVolumeId = opts.networkVolumeId;
    if (opts.dockerArgs) body.dockerArgs = opts.dockerArgs;
    if (opts.dockerStartCmd) body.dockerStartCmd = opts.dockerStartCmd;
    if (opts.dataCenterIds) body.dataCenterIds = opts.dataCenterIds;

    const pod = await this.restRequest<Pod>("POST", "/pods", body);
    return this.normalizePod(pod);
  }

  async deletePod(podId: string): Promise<void> {
    await this.restRequest<unknown>("DELETE", `/pods/${podId}`);
  }

  // ── Pod Lifecycle ──

  async stopPod(podId: string): Promise<unknown> {
    return this.restRequest<unknown>("POST", `/pods/${podId}/stop`);
  }

  async startPod(podId: string): Promise<unknown> {
    return this.restRequest<unknown>("POST", `/pods/${podId}/start`);
  }

  async restartPod(podId: string): Promise<unknown> {
    return this.restRequest<unknown>("POST", `/pods/${podId}/restart`);
  }

  // ── Spot Instances (GraphQL with variables) ──

  async createSpotPod(opts: CreatePodOptions & { bidPerGpu: number }): Promise<{ id: string }> {
    const envArray = Object.entries(opts.env ?? {}).map(([key, value]) => ({ key, value }));
    if (opts.sshPublicKey) {
      envArray.push({ key: "SSH_PUBLIC_KEY", value: opts.sshPublicKey });
    }

    const query = `
      mutation CreateSpotPod($input: PodRentInterruptableInput!) {
        podRentInterruptable(input: $input) {
          id
          imageName
          machineId
        }
      }
    `;

    const input: Record<string, unknown> = {
      name: opts.name,
      gpuCount: opts.gpuCount ?? 1,
      gpuTypeId: opts.gpuTypeIds[0],
      imageName: opts.imageName,
      bidPerGpu: opts.bidPerGpu,
      containerDiskInGb: opts.containerDiskInGb ?? 50,
      volumeInGb: opts.volumeInGb ?? 20,
      volumeMountPath: opts.volumeMountPath ?? "/workspace",
      ports: (opts.ports ?? ["22/tcp"]).join(","),
      env: envArray,
      cloudType: "SECURE",
      supportPublicIp: opts.supportPublicIp ?? true,
    };
    if (opts.networkVolumeId) input.networkVolumeId = opts.networkVolumeId;
    if (opts.dockerArgs) input.dockerArgs = opts.dockerArgs;
    if (opts.dataCenterIds?.length) input.dataCenterIds = opts.dataCenterIds;

    const data = await this.graphqlRequest<{ podRentInterruptable: { id: string } }>(query, { input });
    return data.podRentInterruptable;
  }

  // ── GPU Types ──

  async listGpuTypes(): Promise<GpuType[]> {
    const query = `
      query {
        gpuTypes {
          id
          displayName
          memoryInGb
          communityCloud
          secureCloud
          communityPrice
          communitySpotPrice
          securePrice
          secureSpotPrice
          lowestPrice {
            minimumBidPrice
            uninterruptablePrice
            stockStatus
          }
        }
      }
    `;
    const data = await this.graphqlRequest<{ gpuTypes: GpuType[] }>(query);
    return data.gpuTypes;
  }

  // ── Network Volumes ──

  async listNetworkVolumes(): Promise<NetworkVolume[]> {
    const query = `
      query {
        myself {
          networkVolumes {
            id
            name
            size
            dataCenterId
          }
        }
      }
    `;
    const data = await this.graphqlRequest<{ myself: { networkVolumes: NetworkVolume[] } }>(query);
    return data.myself.networkVolumes;
  }

  async getNetworkVolume(volumeId: string): Promise<NetworkVolume | null> {
    const volumes = await this.listNetworkVolumes();
    return volumes.find((v) => v.id === volumeId) ?? null;
  }

  async createNetworkVolume(name: string, size: number, dataCenterId: string): Promise<NetworkVolume> {
    const query = `
      mutation CreateNetworkVolume($input: CreateNetworkVolumeInput!) {
        createNetworkVolume(input: $input) {
          id
          name
          size
          dataCenterId
        }
      }
    `;
    const data = await this.graphqlRequest<{ createNetworkVolume: NetworkVolume }>(query, {
      input: { name, size, dataCenterId },
    });
    return data.createNetworkVolume;
  }

  async deleteNetworkVolume(volumeId: string): Promise<void> {
    await this.restRequest<unknown>("DELETE", `/networkvolumes/${volumeId}`);
  }

  // ── SSH Helpers (return args arrays for spawn, not shell strings) ──

  getSshArgs(pod: Pod): string[] | null {
    if (!pod.publicIp || !pod.portMappings?.["22"]) return null;
    const args = ["ssh", "-o", "StrictHostKeyChecking=no", "-p", String(pod.portMappings["22"])];
    if (this.config.sshKeyPath) args.push("-i", this.config.sshKeyPath);
    args.push(`root@${pod.publicIp}`);
    return args;
  }

  getSshCommandString(pod: Pod): string | null {
    const args = this.getSshArgs(pod);
    return args ? args.join(" ") : null;
  }

  getRsyncArgs(pod: Pod, localPath: string, remotePath: string, direction: "upload" | "download"): string[] | null {
    if (!pod.publicIp || !pod.portMappings?.["22"]) return null;
    const sshCmd = ["-o", "StrictHostKeyChecking=no", "-p", String(pod.portMappings["22"])];
    if (this.config.sshKeyPath) sshCmd.push("-i", this.config.sshKeyPath);
    const sshArg = `ssh ${sshCmd.join(" ")}`;

    const remote = `root@${pod.publicIp}:${remotePath}`;
    if (direction === "upload") {
      return ["rsync", "-avzP", "-e", sshArg, localPath, remote];
    }
    return ["rsync", "-avzP", "-e", sshArg, remote, localPath];
  }

  // ── Wait for Pod (with TCP probe) ──

  private tcpProbe(host: string, port: number, timeoutMs: number = 5000): Promise<boolean> {
    return new Promise((resolve) => {
      const sock = createConnection({ host, port, timeout: timeoutMs });
      sock.on("connect", () => {
        sock.destroy();
        resolve(true);
      });
      sock.on("error", () => {
        sock.destroy();
        resolve(false);
      });
      sock.on("timeout", () => {
        sock.destroy();
        resolve(false);
      });
    });
  }

  async waitForPod(podId: string, timeoutMs: number = 300_000, intervalMs: number = 10_000): Promise<Pod> {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const pod = await this.getPod(podId);
      if (pod.desiredStatus === "RUNNING" && pod.publicIp && pod.portMappings?.["22"]) {
        const sshReady = await this.tcpProbe(pod.publicIp, pod.portMappings["22"]);
        if (sshReady) return pod;
      }
      if (pod.desiredStatus === "EXITED" || pod.desiredStatus === "ERROR") {
        throw new Error(`Pod ${podId} entered terminal state: ${pod.desiredStatus}`);
      }
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    throw new Error(`Pod ${podId} did not become ready within ${timeoutMs / 1000}s`);
  }
}
