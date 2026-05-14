import type { CpuFlavor } from "./types.js";

// RunPod CPU flavor catalog.
//
// Pricing source: RunPod Console (https://console.runpod.io/pods → CPU tab).
// RunPod does not expose CPU pod pricing via GraphQL or REST as of this writing,
// so hourlyPriceUsd stays null until verified by hand. Run `list_cpu_types` and
// pass the verified figures via PR rather than committing untrusted numbers.
//
// Last metadata review: 2026-05 (flavor IDs, suffix conventions, vendor).
//
// Naming convention:
//   cpu{N}{suffix}
//     N      = generation. cpu3 = AMD EPYC Milan-class. cpu5 = Intel Xeon (newer).
//     suffix = family. c = compute-optimized. g = general-purpose. m = high-memory.
// RAM-per-vCPU ratios mirror the standard cloud tiers (2/4/8 GB per vCPU).

export const CPU_FLAVORS: CpuFlavor[] = [
  {
    id: "cpu3c",
    displayName: "CPU3 Compute (AMD EPYC)",
    generation: "cpu3",
    family: "compute",
    cpuVendor: "AMD EPYC Milan",
    ramGbPerVcpu: 2,
    hourlyPriceUsd: null,
  },
  {
    id: "cpu3g",
    displayName: "CPU3 General (AMD EPYC)",
    generation: "cpu3",
    family: "general",
    cpuVendor: "AMD EPYC Milan",
    ramGbPerVcpu: 4,
    hourlyPriceUsd: null,
  },
  {
    id: "cpu3m",
    displayName: "CPU3 High-Memory (AMD EPYC)",
    generation: "cpu3",
    family: "highmem",
    cpuVendor: "AMD EPYC Milan",
    ramGbPerVcpu: 8,
    hourlyPriceUsd: null,
  },
  {
    id: "cpu5c",
    displayName: "CPU5 Compute (Intel Xeon)",
    generation: "cpu5",
    family: "compute",
    cpuVendor: "Intel Xeon",
    ramGbPerVcpu: 2,
    hourlyPriceUsd: null,
  },
  {
    id: "cpu5g",
    displayName: "CPU5 General (Intel Xeon)",
    generation: "cpu5",
    family: "general",
    cpuVendor: "Intel Xeon",
    ramGbPerVcpu: 4,
    hourlyPriceUsd: null,
  },
  {
    id: "cpu5m",
    displayName: "CPU5 High-Memory (Intel Xeon)",
    generation: "cpu5",
    family: "highmem",
    cpuVendor: "Intel Xeon",
    ramGbPerVcpu: 8,
    hourlyPriceUsd: null,
  },
];

export const CPU_FLAVOR_IDS = CPU_FLAVORS.map((f) => f.id);

export function getCpuFlavor(id: string): CpuFlavor | undefined {
  return CPU_FLAVORS.find((f) => f.id === id);
}

export function flavorsByFamily(family: CpuFlavor["family"]): CpuFlavor[] {
  return CPU_FLAVORS.filter((f) => f.family === family);
}

/**
 * Pick a default ordered list for a workload family. Falls back to all flavors
 * when family is unspecified. Order: cpu5 before cpu3 (newer CPU usually wins).
 */
export function defaultFlavorOrder(family?: CpuFlavor["family"]): string[] {
  const pool = family ? flavorsByFamily(family) : CPU_FLAVORS;
  return [...pool]
    .sort((a, b) => (a.generation < b.generation ? 1 : a.generation > b.generation ? -1 : 0))
    .map((f) => f.id);
}
