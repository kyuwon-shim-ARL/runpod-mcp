import { describe, it, expect, vi, beforeEach } from "vitest";

const spawnAsync = vi.fn();
vi.mock("../api.js", async () => {
  const actual = await vi.importActual<typeof import("../api.js")>("../api.js");
  return { ...actual, spawnAsync };
});

const { ensureRemoteRsync } = await import("../index.js");

const SSH = ["ssh", "-p", "40000", "root@1.2.3.4"];
const ok = (stdout: string) => ({ status: 0, stdout, stderr: "", error: undefined });
const fail = (stderr = "") => ({ status: 1, stdout: "", stderr, error: undefined });

describe("ensureRemoteRsync", () => {
  beforeEach(() => spawnAsync.mockReset());

  it("does nothing when rsync is already on the pod", async () => {
    spawnAsync.mockResolvedValueOnce(ok("/usr/bin/rsync\n"));
    expect(await ensureRemoteRsync(SSH)).toBeNull();
    expect(spawnAsync).toHaveBeenCalledTimes(1);
  });

  it("installs rsync when the probe finds nothing, then reports success", async () => {
    spawnAsync.mockResolvedValueOnce(fail()).mockResolvedValueOnce(ok("/usr/bin/rsync\n"));
    expect(await ensureRemoteRsync(SSH)).toBeNull();
    const installCmd = spawnAsync.mock.calls[1][1].at(-1) as string;
    // the default image is Debian-based, but the fallbacks must not be dropped silently
    expect(installCmd).toContain("apt-get install -y -qq rsync");
    expect(installCmd).toContain("yum install");
    expect(installCmd).toContain("apk add");
  });

  it("treats an exit-0 probe with empty stdout as missing", async () => {
    // `command -v rsync` can exit 0 with no output under some shells; empty means absent
    spawnAsync.mockResolvedValueOnce(ok("   \n")).mockResolvedValueOnce(ok("/usr/bin/rsync\n"));
    expect(await ensureRemoteRsync(SSH)).toBeNull();
    expect(spawnAsync).toHaveBeenCalledTimes(2);
  });

  it("returns an actionable message when install fails", async () => {
    spawnAsync.mockResolvedValueOnce(fail()).mockResolvedValueOnce(fail("E: Unable to locate package"));
    const msg = await ensureRemoteRsync(SSH);
    expect(msg).toContain("could not be installed");
    expect(msg).toContain("apt-get install -y rsync");
    expect(msg).toContain("Unable to locate package");
  });

  it("probes with `command -v`, not by running rsync", async () => {
    spawnAsync.mockResolvedValueOnce(ok("/usr/bin/rsync\n"));
    await ensureRemoteRsync(SSH);
    expect(spawnAsync.mock.calls[0][1].at(-1)).toBe("command -v rsync");
  });
});
