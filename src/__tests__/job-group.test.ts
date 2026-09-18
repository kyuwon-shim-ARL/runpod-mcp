import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { toYaml, buildPodMetadataStub } from "../pod-ops.js";
import { readPodGroups, renderGroupLine, groupSiblings } from "../job-group.js";

describe("buildPodMetadataStub job_group", () => {
  it("emits job_group when given", () => {
    const stub = JSON.parse(
      buildPodMetadataStub({
        pod_id: "p1",
        name: "n",
        created_at: "2026-09-18T00:00:00.000Z",
        job_group: "lopo-s123",
      })
    );
    expect(stub.job_group).toBe("lopo-s123");
  });

  it("emits job_group: null when omitted, so the field is visible in the stub", () => {
    const stub = JSON.parse(
      buildPodMetadataStub({ pod_id: "p1", name: "n", created_at: "2026-09-18T00:00:00.000Z" })
    );
    expect(stub.job_group).toBeNull();
  });
});

describe("readPodGroups", () => {
  let dir: string;
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "job-group-"));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  const write = (file: string, obj: unknown) => writeFile(join(dir, file), toYaml(obj), "utf8");

  it("maps pod ids to their job_group", async () => {
    await write("2026-09-18_a.yaml", { pod_id: "p1", name: "a", job_group: "lopo-s123" });
    await write("2026-09-18_b.yaml", { pod_id: "p2", name: "b", job_group: "lopo-s123" });

    expect(await readPodGroups(dir)).toEqual({ p1: "lopo-s123", p2: "lopo-s123" });
  });

  it("skips records with no job_group", async () => {
    await write("a.yaml", { pod_id: "p1", name: "a", job_group: null });
    await write("b.yaml", { pod_id: "p2", name: "b" });

    expect(await readPodGroups(dir)).toEqual({});
  });

  it("returns empty when the directory does not exist (fail-soft)", async () => {
    expect(await readPodGroups(join(dir, "nope"))).toEqual({});
  });

  it("ignores unparseable files instead of throwing", async () => {
    await writeFile(join(dir, "broken.yaml"), "  not yaml at all", "utf8");
    await write("good.yaml", { pod_id: "p1", name: "a", job_group: "g" });

    expect(await readPodGroups(dir)).toEqual({ p1: "g" });
  });

  it("ignores non-yaml files", async () => {
    await writeFile(join(dir, "notes.md"), "pod_id: p9\njob_group: bogus\n", "utf8");
    await write("good.yaml", { pod_id: "p1", name: "a", job_group: "g" });

    expect(await readPodGroups(dir)).toEqual({ p1: "g" });
  });

  it("does not pick up job_group nested under another key", async () => {
    await writeFile(
      join(dir, "nested.yaml"),
      "pod_id: p1\nname: a\nincidents:\n  - job_group: not-a-real-group\n",
      "utf8"
    );

    expect(await readPodGroups(dir)).toEqual({});
  });
});

describe("groupSiblings", () => {
  it("lists the other live pods in the same group", () => {
    const groups = { p1: "g", p2: "g", p3: "other" };
    expect(groupSiblings("p1", groups, ["p1", "p2", "p3"])).toEqual(["p2"]);
  });

  it("returns empty for a pod with no group", () => {
    expect(groupSiblings("p9", { p1: "g" }, ["p1", "p9"])).toEqual([]);
  });

  it("excludes siblings that are no longer live", () => {
    expect(groupSiblings("p1", { p1: "g", p2: "g" }, ["p1"])).toEqual([]);
  });
});

describe("renderGroupLine", () => {
  it("names the group and its live siblings", () => {
    expect(renderGroupLine("lopo-s123", ["p2", "p3"])).toBe("Group: lopo-s123 (siblings live: p2, p3)");
  });

  it("marks a group with no live siblings", () => {
    expect(renderGroupLine("lopo-s123", [])).toBe("Group: lopo-s123 (no live siblings)");
  });

  it("returns empty string for an ungrouped pod", () => {
    expect(renderGroupLine(undefined, [])).toBe("");
  });
});
