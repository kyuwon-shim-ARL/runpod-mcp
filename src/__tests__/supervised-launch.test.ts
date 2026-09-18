import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm, readFile, writeFile, chmod } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnAsync } from "../api.js";
import { buildSupervisedScript, parseStatusLine } from "../supervised-launch.js";

describe("buildSupervisedScript", () => {
  const base = { command: "python3 train.py", statusPath: "/out/STATUS", logPath: "/out/train.log" };

  it("embeds the training command, status path and log path", () => {
    const s = buildSupervisedScript(base);
    expect(s).toContain("python3 train.py");
    expect(s).toContain("/out/STATUS");
    expect(s).toContain("/out/train.log");
  });

  it("waits on the training pid instead of polling kill -0 from the main shell", () => {
    // The zombie trap: `kill -0` succeeds on a reaped-but-not-waited child, so a main shell
    // that polls it spins forever while STATUS never leaves "starting" — the watchdog fails
    // in exactly the case it exists for. `wait` reaps, so the shell learns the truth.
    const s = buildSupervisedScript(base);
    expect(s).toContain('wait "$TRAIN_PID"');
    expect(s).not.toMatch(/kill\s+-0\s+"?\$TRAIN_PID/);
  });

  it("runs the watchdog as its own process, not inline in the main shell", () => {
    const s = buildSupervisedScript(base);
    expect(s).toContain("WATCHDOG_PID=$!");
  });

  it("uses the given idle alert threshold", () => {
    expect(buildSupervisedScript({ ...base, idleAlertMinutes: 42 })).toContain("IDLE_ALERT_MIN=42");
  });

  it("defaults the idle threshold to 25 minutes", () => {
    expect(buildSupervisedScript(base)).toContain("IDLE_ALERT_MIN=25");
  });

  it("skips work that is already complete when skipIfExists is given", () => {
    const s = buildSupervisedScript({ ...base, skipIfExists: "/out/predictions.json" });
    expect(s).toContain("/out/predictions.json");
    expect(s).toContain("already complete");
  });

  it("omits the skip branch when skipIfExists is not given", () => {
    expect(buildSupervisedScript(base)).not.toContain("already complete");
  });

  it("rejects a command containing a single quote, which would break the quoting", () => {
    expect(() => buildSupervisedScript({ ...base, command: "python3 -c 'x'" })).toThrow(/single quote/i);
  });

  it("rejects a multi-line command — $! would capture the wrong process", () => {
    // `cmd\nother > "$LOG" &` backgrounds only the last line, so TRAIN_PID points at
    // something that is not the training run and the watchdog supervises the wrong thing.
    expect(() => buildSupervisedScript({ ...base, command: "python3 a.py\npython3 b.py" })).toThrow(
      /single line/i
    );
    expect(() => buildSupervisedScript({ ...base, command: "python3 a.py\r\npython3 b.py" })).toThrow(
      /single line/i
    );
  });

  it("rejects a newline in a path, which would split the generated script", () => {
    expect(() => buildSupervisedScript({ ...base, statusPath: "/out/S\nid" })).toThrow(/single line/i);
    expect(() => buildSupervisedScript({ ...base, logPath: "/out/L\nid" })).toThrow(/single line/i);
    expect(() => buildSupervisedScript({ ...base, workingDir: "/w\nid" })).toThrow(/single line/i);
    expect(() => buildSupervisedScript({ ...base, skipIfExists: "/d\nid" })).toThrow(/single line/i);
  });

  it("rejects a label that would escape the script directory", () => {
    // label lands in /root/.runpod-mcp/<label>.sh on the pod, as root.
    expect(() => buildSupervisedScript({ ...base, label: "../../etc/cron.d/x" })).toThrow(/label must match/i);
    expect(() => buildSupervisedScript({ ...base, label: "a/b" })).toThrow(/label must match/i);
    expect(() => buildSupervisedScript({ ...base, label: ".." })).toThrow(/label must match/i);
    expect(() => buildSupervisedScript({ ...base, label: "a b" })).toThrow(/label must match/i);
    expect(() => buildSupervisedScript({ ...base, label: "" })).toThrow(/label must match/i);
  });

  it("accepts ordinary labels", () => {
    for (const label of ["run1", "clean-lopo_s123", "e219.arm-A"]) {
      expect(() => buildSupervisedScript({ ...base, label })).not.toThrow();
    }
  });
});

describe("parseStatusLine", () => {
  it("parses RUNNING with progress fields", () => {
    const r = parseStatusLine("RUNNING run1 epoch=7/30 gpu=100% idle=0min 2026-09-18T10:00:00+09:00");
    expect(r).toMatchObject({ state: "RUNNING", idleMinutes: 0, gpuUtil: 100 });
  });

  it("parses ALERT and its idle minutes", () => {
    const r = parseStatusLine("ALERT idle 47min - log has not moved | run1 epoch=7/30 gpu=0% 2026-09-18T10:00:00+09:00");
    expect(r).toMatchObject({ state: "ALERT", idleMinutes: 47 });
  });

  it("parses DONE with the exit code", () => {
    expect(parseStatusLine("DONE run1 rc=0 elapsed=29449s 2026-09-18T10:00:00+09:00")).toMatchObject({
      state: "DONE",
      exitCode: 0,
    });
  });

  it("parses FAILED with the exit code", () => {
    expect(parseStatusLine("FAILED run1 rc=1 elapsed=12s - tail /out/train.log")).toMatchObject({
      state: "FAILED",
      exitCode: 1,
    });
  });

  it("returns null for an empty or unrecognized line", () => {
    expect(parseStatusLine("")).toBeNull();
    expect(parseStatusLine("something else entirely")).toBeNull();
  });
});

/**
 * The zombie-trap regression, run against a real shell. A generated script whose training
 * command dies must end at FAILED — the original bug left STATUS stuck at "starting" forever.
 */
describe("generated script against a real shell", () => {
  let dir: string;
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "supervised-"));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  async function runScript(overrides: Record<string, unknown> = {}) {
    const statusPath = join(dir, "STATUS");
    const logPath = join(dir, "train.log");
    const script = buildSupervisedScript({
      command: "true",
      statusPath,
      logPath,
      label: "testrun",
      workingDir: dir,
      ...overrides,
    } as Parameters<typeof buildSupervisedScript>[0]);
    const scriptPath = join(dir, "run.sh");
    await writeFile(scriptPath, script, "utf8");
    await chmod(scriptPath, 0o755);
    const result = await spawnAsync("bash", [scriptPath], { timeout: 30_000 });
    const status = await readFile(statusPath, "utf8").catch(() => "");
    return { result, status };
  }

  it("reports FAILED when the training command exits non-zero (the zombie trap)", async () => {
    const { result, status } = await runScript({ command: "exit 3" });

    expect(status).toContain("FAILED");
    expect(status).toContain("rc=3");
    expect(status).not.toContain("starting");
    expect(result.status).toBe(3);
  });

  it("reports DONE when the training command succeeds", async () => {
    const { result, status } = await runScript({ command: "true" });

    expect(status).toContain("DONE");
    expect(status).toContain("rc=0");
    expect(result.status).toBe(0);
  });

  it("leaves no watchdog process behind after the run", async () => {
    await runScript({ command: "true" });

    const ps = await spawnAsync("bash", ["-c", "ps -eo args | grep -c '[I]DLE_ALERT_MIN' || true"], {
      timeout: 10_000,
    });
    expect((ps.stdout ?? "").trim()).toBe("0");
  });

  it("skips and reports DONE when skipIfExists already exists", async () => {
    const done = join(dir, "predictions.json");
    await writeFile(done, "{}", "utf8");

    const { status } = await runScript({ command: "exit 1", skipIfExists: done });

    expect(status).toContain("DONE");
    expect(status).toContain("already complete");
  });
});
