#!/usr/bin/env node
/**
 * RunPod GPU Watchdog — standalone cron script for 24/7 idle detection.
 *
 * Architecture decisions (T4 spec):
 * - Auth: reads RUNPOD_API_KEY from environment (same as MCP server)
 * - Build: compiled to dist/watchdog.js, registered as bin entry
 * - Alert: stdout/stderr phase 1, alerter interface for future webhook extension
 * - Skip: --skip-pattern regex CLI arg, default "keep|persist"
 *
 * Usage:
 *   npx runpod-watchdog                    # alert-only (default)
 *   npx runpod-watchdog --auto-stop        # stop pods after 3 consecutive idle checks
 *   npx runpod-watchdog --skip-pattern "my-keep|important"
 *   npx runpod-watchdog --idle-threshold 3  # consecutive idle checks before action
 *
 * Cron example (every 5 min):
 *   *​/5 * * * * RUNPOD_API_KEY=rp_xxx node /path/to/dist/watchdog.js >> /var/log/runpod-watchdog.log 2>&1
 */
import { RunPodClient, spawnAsync } from "./api.js";
import { parseNvidiaSmiOutput, labelVramUsage } from "./gpu-utils.js";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

// ── Types ──

export interface WatchdogOptions {
  autoStop: boolean;
  skipPattern: RegExp;
  idleThreshold: number;
  stateFile: string;
}

export interface PodIdleState {
  [podId: string]: {
    consecutiveIdle: number;
    lastChecked: string;
    lastStatus: string;
  };
}

export interface Alerter {
  warn(podName: string, podId: string, idleCount: number, message: string): void;
  action(podName: string, podId: string, action: string): void;
}

// ── Default alerter (stdout) ──

export const consoleAlerter: Alerter = {
  warn(podName, podId, idleCount, message) {
    console.warn(`[WARN] ${podName} (${podId}): idle ${idleCount} consecutive checks — ${message}`);
  },
  action(podName, podId, action) {
    console.log(`[ACTION] ${podName} (${podId}): ${action}`);
  },
};

// ── State management ──

export function loadState(stateFile: string): PodIdleState {
  try {
    return JSON.parse(readFileSync(stateFile, "utf-8"));
  } catch {
    return {};
  }
}

export function saveState(stateFile: string, state: PodIdleState): void {
  writeFileSync(stateFile, JSON.stringify(state, null, 2));
}

// ── Core watchdog logic ──

export async function checkPod(
  client: RunPodClient,
  podId: string,
  podName: string,
  sshKeyPath: string | undefined
): Promise<{ status: string; vramPercent: number | null }> {
  const pod = await client.getPod(podId);
  if (!pod.publicIp || !pod.portMappings?.["22"]) {
    return { status: "NO_SSH", vramPercent: null };
  }

  const sshArgs = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
    ...(sshKeyPath ? ["-i", sshKeyPath] : []),
    "-p", String(pod.portMappings["22"]),
    `root@${pod.publicIp}`,
    "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits",
  ];

  const result = await spawnAsync("ssh", sshArgs, { timeout: 30_000 });
  if (result.error || result.status !== 0) {
    return { status: "SSH_FAILED", vramPercent: null };
  }

  const output = result.stdout.trim();
  if (!output) {
    return { status: "UNKNOWN", vramPercent: null };
  }

  const gpus = parseNvidiaSmiOutput(output);
  if (!gpus.length) {
    return { status: "UNKNOWN", vramPercent: null };
  }

  // Use first GPU metrics
  const gpu = gpus[0];
  const vramPercent = gpu.totalMb > 0
    ? Math.round((gpu.usedMb / gpu.totalMb) * 100)
    : 0;
  const label = labelVramUsage(gpu.usedMb, gpu.totalMb);

  return { status: label, vramPercent };
}

export async function runWatchdog(
  client: RunPodClient,
  options: WatchdogOptions,
  alerter: Alerter = consoleAlerter
): Promise<{ checked: number; idle: number; stopped: number }> {
  const pods = await client.listPods();
  const running = pods.filter((p) => p.desiredStatus === "RUNNING");
  const state = loadState(options.stateFile);
  let checked = 0;
  let idle = 0;
  let stopped = 0;

  for (const pod of running) {
    if (options.skipPattern.test(pod.name)) continue;

    checked++;
    try {
      const { status, vramPercent } = await checkPod(
        client,
        pod.id,
        pod.name,
        client.config.sshKeyPath
      );

      if (status === "SSH_FAILED" || status === "NO_SSH") {
        // Can't check — skip, don't count as idle
        state[pod.id] = {
          consecutiveIdle: state[pod.id]?.consecutiveIdle ?? 0,
          lastChecked: new Date().toISOString(),
          lastStatus: status,
        };
        continue;
      }

      if (status === "UNKNOWN") {
        // Empty nvidia-smi — treat as unknown, not idle (false positive prevention)
        state[pod.id] = {
          consecutiveIdle: 0,
          lastChecked: new Date().toISOString(),
          lastStatus: "UNKNOWN",
        };
        continue;
      }

      const prevIdle = state[pod.id]?.consecutiveIdle ?? 0;

      if (status === "IDLE") {
        const newIdle = prevIdle + 1;
        idle++;
        state[pod.id] = {
          consecutiveIdle: newIdle,
          lastChecked: new Date().toISOString(),
          lastStatus: `IDLE (${vramPercent}% VRAM)`,
        };

        alerter.warn(pod.name, pod.id, newIdle,
          `VRAM ${vramPercent}% — GPU is wasted ($${pod.costPerHr ?? "?"}/hr)`);

        if (options.autoStop && newIdle >= options.idleThreshold) {
          try {
            await client.stopPod(pod.id);
            stopped++;
            alerter.action(pod.name, pod.id, `Auto-stopped after ${newIdle} consecutive idle checks`);
          } catch (e) {
            alerter.warn(pod.name, pod.id, newIdle, `Auto-stop failed: ${(e as Error).message}`);
          }
        }
      } else {
        // Not idle — reset counter
        state[pod.id] = {
          consecutiveIdle: 0,
          lastChecked: new Date().toISOString(),
          lastStatus: `${status} (${vramPercent}% VRAM)`,
        };
      }
    } catch (e) {
      // Pod check failed entirely — skip
      state[pod.id] = {
        consecutiveIdle: state[pod.id]?.consecutiveIdle ?? 0,
        lastChecked: new Date().toISOString(),
        lastStatus: `ERROR: ${(e as Error).message}`,
      };
    }
  }

  // Clean up state for pods no longer running
  for (const podId of Object.keys(state)) {
    if (!running.some((p) => p.id === podId)) {
      delete state[podId];
    }
  }

  saveState(options.stateFile, state);
  return { checked, idle, stopped };
}

// ── CLI ──

function parseArgs(argv: string[]): WatchdogOptions {
  let autoStop = false;
  let skipPattern = "keep|persist";
  let idleThreshold = 3;
  let stateFile = resolve(process.env.HOME ?? "/tmp", ".runpod-watchdog-state.json");

  for (let i = 2; i < argv.length; i++) {
    switch (argv[i]) {
      case "--auto-stop":
        autoStop = true;
        break;
      case "--skip-pattern":
        skipPattern = argv[++i] ?? skipPattern;
        break;
      case "--idle-threshold":
        idleThreshold = parseInt(argv[++i] ?? "3", 10);
        break;
      case "--state-file":
        stateFile = argv[++i] ?? stateFile;
        break;
      case "--help":
        console.log(`Usage: runpod-watchdog [options]
  --auto-stop          Stop pods after consecutive idle checks (default: alert only)
  --skip-pattern RE    Regex for pod names to skip (default: "keep|persist")
  --idle-threshold N   Consecutive idle checks before auto-stop (default: 3)
  --state-file PATH    Path to state file (default: ~/.runpod-watchdog-state.json)
  --help               Show this help`);
        process.exit(0);
    }
  }

  return {
    autoStop,
    skipPattern: new RegExp(skipPattern, "i"),
    idleThreshold,
    stateFile,
  };
}

async function main() {
  const apiKey = process.env.RUNPOD_API_KEY;
  if (!apiKey) {
    console.error("RUNPOD_API_KEY is required. Set it in your environment.");
    process.exit(1);
  }

  const client = new RunPodClient({
    apiKey,
    restBaseUrl: "https://rest.runpod.io/v1",
    graphqlUrl: "https://api.runpod.io/graphql",
    sshKeyPath: process.env.SSH_KEY_PATH,
  });

  const options = parseArgs(process.argv);
  const { checked, idle, stopped } = await runWatchdog(client, options);

  console.log(`[${new Date().toISOString()}] Watchdog complete: ${checked} pods checked, ${idle} idle, ${stopped} stopped`);
}

main().catch((e) => {
  console.error("Watchdog error:", e);
  process.exit(1);
});
