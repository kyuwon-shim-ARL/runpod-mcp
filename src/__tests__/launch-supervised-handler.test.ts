import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Pod } from "../types.js";

const spawnAsync = vi.fn();
const getPod = vi.fn();

vi.mock("../api.js", async () => {
  const actual = await vi.importActual<typeof import("../api.js")>("../api.js");
  class FakeRunPodClient {
    listPods = vi.fn().mockResolvedValue([]);
    getPod = getPod;
    getSshArgs(pod: Pod) {
      if (!pod.publicIp || !pod.portMappings?.["22"]) return null;
      return ["ssh", "-p", String(pod.portMappings["22"]), `root@${pod.publicIp}`];
    }
    getSshCommandString() {
      return null;
    }
  }
  return { ...actual, spawnAsync, RunPodClient: FakeRunPodClient };
});

process.env.RUNPOD_API_KEY = "rp_test_key";

const { server } = await import("../index.js");

const POD: Pod = {
  id: "pod-1",
  name: "trainer",
  desiredStatus: "RUNNING",
  publicIp: "1.2.3.4",
  portMappings: { "22": 40001 },
  costPerHr: 0.22,
};

const ok = (stdout: string) => ({ status: 0, stdout, stderr: "", error: undefined });

async function callLaunch(args: Record<string, unknown> = {}) {
  const tool = (server as any)._registeredTools["launch_supervised_training"];
  const result = await tool.handler(
    {
      podId: "pod-1",
      command: "python3 train.py",
      label: "run",
      statusPath: "/root/outputs/STATUS",
      logPath: "/root/outputs/train.log",
      workingDir: "/workspace",
      idleAlertMinutes: 25,
      progressPattern: "Epoch [0-9]+",
      ...args,
    },
    {}
  );
  return result.content[0].text as string;
}

/** The single SSH command the tool sent, with its base64 payload decoded back to the script. */
function sentScript(): string {
  const command = spawnAsync.mock.calls[0][1].at(-1) as string;
  const b64 = command.match(/echo ([A-Za-z0-9+/=]+) \| base64 -d/)?.[1] ?? "";
  return Buffer.from(b64, "base64").toString("utf8");
}

describe("launch_supervised_training", () => {
  beforeEach(() => {
    getPod.mockReset().mockResolvedValue(POD);
    spawnAsync.mockReset().mockResolvedValue(ok("LAUNCHED 4242\n"));
  });

  it("installs the generated script and reports where the status lives", async () => {
    const output = await callLaunch({ totalSteps: 30 });

    expect(output).toContain("✅ Launched run");
    expect(output).toContain("/root/outputs/STATUS");
    expect(output).toContain("cat /root/outputs/STATUS");

    const script = sentScript();
    expect(script).toContain("python3 train.py");
    expect(script).toContain('wait "$TRAIN_PID"');
    expect(script).toContain("IDLE_ALERT_MIN=25");
  });

  it("launches detached, so the MCP server is not blocked by the training run", async () => {
    await callLaunch();
    const command = spawnAsync.mock.calls[0][1].at(-1) as string;
    expect(command).toContain("nohup");
    expect(command).toMatch(/&\s*echo LAUNCHED/);
  });

  it("refuses a command with a single quote instead of shipping mangled shell", async () => {
    const output = await callLaunch({ command: "python3 -c 'import x'" });

    expect(output).toContain("single quote");
    expect(spawnAsync).not.toHaveBeenCalled();
  });

  it("reports failure when the launch command does not confirm", async () => {
    spawnAsync.mockResolvedValue({ status: 1, stdout: "", stderr: "bash: no such file", error: undefined });

    const output = await callLaunch();

    expect(output).toContain("❌ Launch failed");
    expect(output).toContain("no such file");
  });

  it("reports an SSH error rather than claiming a launch", async () => {
    spawnAsync.mockResolvedValue({ status: null, stdout: "", stderr: "", error: new Error("connection refused") });

    const output = await callLaunch();

    expect(output).toContain("❌ SSH error");
    expect(output).not.toContain("✅");
  });

  it("refuses to relaunch over a job whose STATUS still reads RUNNING", async () => {
    // Relaunching with a live label would overwrite the running job's script and its status,
    // leaving the job that is actually burning GPU time unattributable.
    spawnAsync.mockResolvedValue(ok("ALREADY_RUNNING\n"));

    const output = await callLaunch();

    expect(output).toContain("already reads RUNNING");
    expect(output).toContain("cat /root/outputs/STATUS");
    expect(output).not.toContain("✅");
  });

  it("guards the relaunch with a STATUS check before writing anything", async () => {
    await callLaunch();
    const command = spawnAsync.mock.calls[0][1].at(-1) as string;
    // The guard must come before the script is written, or the overwrite has already happened.
    expect(command.indexOf("ALREADY_RUNNING")).toBeLessThan(command.indexOf("base64 -d"));
  });

  it("refuses a pod that is not SSH-ready", async () => {
    getPod.mockResolvedValue({ ...POD, publicIp: undefined, portMappings: undefined });

    const output = await callLaunch();

    expect(output).toContain("wait_for_pod");
    expect(spawnAsync).not.toHaveBeenCalled();
  });
});
