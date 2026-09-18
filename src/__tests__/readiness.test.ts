import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm, writeFile, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  pendingGates,
  recordGate,
  markVerified,
  pruneGates,
  renderGateBlock,
  renderPendingRefusal,
  loadReadinessState,
  saveReadinessState,
  importStatementLabel,
  unmetImports,
  type ReadinessState,
} from "../readiness.js";

describe("pendingGates", () => {
  it("returns gates that were never verified", () => {
    const state: ReadinessState = {
      p1: { imports: ["torch"], createdAt: "2026-09-18T00:00:00.000Z", verifiedAt: null },
    };
    expect(pendingGates(state).map((g) => g.podId)).toEqual(["p1"]);
  });

  it("excludes verified gates", () => {
    const state: ReadinessState = {
      p1: { imports: ["torch"], createdAt: "2026-09-18T00:00:00.000Z", verifiedAt: "2026-09-18T01:00:00.000Z" },
    };
    expect(pendingGates(state)).toEqual([]);
  });

  it("returns empty for empty state", () => {
    expect(pendingGates({})).toEqual([]);
  });
});

describe("recordGate", () => {
  it("adds an unverified gate", () => {
    const now = new Date("2026-09-18T00:00:00.000Z");
    const next = recordGate({}, "p1", ["torch", "pandas"], now);
    expect(next.p1).toEqual({
      imports: ["torch", "pandas"],
      createdAt: "2026-09-18T00:00:00.000Z",
      verifiedAt: null,
    });
  });

  it("does not mutate the input state", () => {
    const state: ReadinessState = {};
    recordGate(state, "p1", ["torch"], new Date());
    expect(state).toEqual({});
  });
});

describe("markVerified", () => {
  it("stamps verifiedAt on a known pod", () => {
    const state = recordGate({}, "p1", ["torch"], new Date("2026-09-18T00:00:00.000Z"));
    const next = markVerified(state, "p1", new Date("2026-09-18T02:00:00.000Z"));
    expect(next.p1.verifiedAt).toBe("2026-09-18T02:00:00.000Z");
    expect(pendingGates(next)).toEqual([]);
  });

  it("is a no-op for a pod with no gate", () => {
    const next = markVerified({}, "ghost", new Date());
    expect(next).toEqual({});
  });
});

describe("pruneGates", () => {
  it("drops gates for pods that no longer exist", () => {
    const state = recordGate(recordGate({}, "p1", ["torch"], new Date()), "p2", ["numpy"], new Date());
    expect(Object.keys(pruneGates(state, ["p2"]))).toEqual(["p2"]);
  });

  it("keeps everything when all pods are live", () => {
    const state = recordGate({}, "p1", ["torch"], new Date());
    expect(pruneGates(state, ["p1"])).toEqual(state);
  });
});

describe("renderPendingRefusal", () => {
  it("names every blocking pod and its run_preflight call", () => {
    const state = recordGate({}, "abc123", ["torch", "kornia"], new Date("2026-09-18T00:00:00.000Z"));
    const out = renderPendingRefusal(pendingGates(state));
    expect(out).toContain("abc123");
    expect(out).toContain("run_preflight");
    expect(out).toContain("kornia");
    expect(out).toContain("skipReadinessGate");
  });
});

describe("renderGateBlock", () => {
  it("marks the pod as not ready and shows the exact verification call", () => {
    const out = renderGateBlock("abc123", ["torch", "pandas"]);
    expect(out).toContain("abc123");
    expect(out).toContain("run_preflight");
    expect(out).toContain("torch");
    expect(out).toMatch(/NOT READY|미검증/);
  });
});

describe("state persistence", () => {
  let dir: string;
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "readiness-"));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it("round-trips through disk", async () => {
    const path = join(dir, "nested", "readiness.json");
    const state = recordGate({}, "p1", ["torch"], new Date("2026-09-18T00:00:00.000Z"));
    await saveReadinessState(state, path);
    expect(await loadReadinessState(path)).toEqual(state);
  });

  it("returns empty state when the file is missing (fail-soft)", async () => {
    expect(await loadReadinessState(join(dir, "nope.json"))).toEqual({});
  });

  it("returns empty state when the file is corrupt (fail-soft)", async () => {
    const path = join(dir, "bad.json");
    await writeFile(path, "{not json", "utf8");
    expect(await loadReadinessState(path)).toEqual({});
  });

  it("returns empty state when the file holds a non-object (fail-soft)", async () => {
    const path = join(dir, "arr.json");
    await writeFile(path, "[1,2,3]", "utf8");
    expect(await loadReadinessState(path)).toEqual({});
  });

  it("writes atomically, leaving no temp file behind", async () => {
    const { readdir } = await import("node:fs/promises");
    const path = join(dir, "readiness.json");
    await saveReadinessState(recordGate({}, "p1", ["torch"], new Date()), path);

    expect((await readdir(dir)).filter((f) => f.includes(".tmp-"))).toEqual([]);
    const raw = await readFile(path, "utf8");
    expect(JSON.parse(raw).p1.verifiedAt).toBeNull();
  });
});

describe("importStatementLabel", () => {
  it("labels a bare module name (the old split(\" \")[1] gave undefined)", () => {
    expect(importStatementLabel("torch")).toBe("torch");
  });

  it("labels a plain import", () => {
    expect(importStatementLabel("import pandas")).toBe("pandas");
  });

  it("labels a from-import by its module", () => {
    expect(importStatementLabel("from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer")).toBe(
      "chemprop.featurizers"
    );
  });

  it("falls back to the whole statement for anything else", () => {
    expect(importStatementLabel("x = 1")).toBe("x = 1");
  });

  it("tolerates surrounding whitespace", () => {
    expect(importStatementLabel("  import kornia  ")).toBe("kornia");
  });
});

describe("unmetImports", () => {
  const gate = (imports: string[]) => recordGate({}, "p", imports, new Date()).p;

  it("reports the gated imports that were not among the passing set", () => {
    expect(unmetImports(gate(["torch", "kornia"]), ["os"])).toEqual(["torch", "kornia"]);
  });

  it("is empty when the passing set covers the gate", () => {
    expect(unmetImports(gate(["torch"]), ["torch", "pandas"])).toEqual([]);
  });

  it("compares by module, so statement and bare forms match", () => {
    expect(unmetImports(gate(["import torch"]), ["torch"])).toEqual([]);
    expect(unmetImports(gate(["torch"]), ["import torch"])).toEqual([]);
  });

  it("reports only the missing subset", () => {
    expect(unmetImports(gate(["torch", "kornia"]), ["torch"])).toEqual(["kornia"]);
  });
});
