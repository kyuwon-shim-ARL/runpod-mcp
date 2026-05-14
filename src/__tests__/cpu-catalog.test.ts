import { describe, it, expect } from "vitest";
import { CPU_FLAVORS, CPU_FLAVOR_IDS, getCpuFlavor, flavorsByFamily, defaultFlavorOrder } from "../cpu-catalog.js";

describe("cpu-catalog", () => {
  it("contains all six documented RunPod CPU flavors", () => {
    expect(CPU_FLAVOR_IDS).toEqual(
      expect.arrayContaining(["cpu3c", "cpu3g", "cpu3m", "cpu5c", "cpu5g", "cpu5m"])
    );
    expect(CPU_FLAVORS).toHaveLength(6);
  });

  it("encodes RAM-per-vCPU by family (c=2, g=4, m=8)", () => {
    for (const f of CPU_FLAVORS) {
      const expected = { compute: 2, general: 4, highmem: 8 }[f.family];
      expect(f.ramGbPerVcpu).toBe(expected);
    }
  });

  it("leaves hourlyPriceUsd null until verified (RunPod doesn't expose CPU pricing via API)", () => {
    for (const f of CPU_FLAVORS) {
      expect(f.hourlyPriceUsd).toBeNull();
    }
  });

  it("getCpuFlavor returns the right entry by id", () => {
    expect(getCpuFlavor("cpu5c")?.family).toBe("compute");
    expect(getCpuFlavor("cpu3m")?.family).toBe("highmem");
    expect(getCpuFlavor("nope")).toBeUndefined();
  });

  it("flavorsByFamily filters correctly", () => {
    expect(flavorsByFamily("compute").map((f) => f.id).sort()).toEqual(["cpu3c", "cpu5c"]);
    expect(flavorsByFamily("highmem").map((f) => f.id).sort()).toEqual(["cpu3m", "cpu5m"]);
  });

  it("defaultFlavorOrder puts cpu5 (newer) before cpu3", () => {
    const order = defaultFlavorOrder();
    const idx5 = order.findIndex((id) => id.startsWith("cpu5"));
    const idx3 = order.findIndex((id) => id.startsWith("cpu3"));
    expect(idx5).toBeLessThan(idx3);
  });

  it("defaultFlavorOrder with family filter returns only that family", () => {
    expect(defaultFlavorOrder("general")).toEqual(["cpu5g", "cpu3g"]);
  });
});
