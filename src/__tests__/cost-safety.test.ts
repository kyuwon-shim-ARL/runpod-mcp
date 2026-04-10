/**
 * Unit tests for the create_pod_auto costSafetyConfirmed gate.
 * Tests the guard logic: gpuCount >= COST_GATE_GPU_COUNT + !dryRun → gate triggers.
 * With elicitation: action must be 'accept' AND content.confirmed === true.
 * Without elicitation: boolean gate (costSafetyConfirmed must be true).
 */
import { describe, it, expect } from "vitest";

const COST_GATE_GPU_COUNT = 2;
const COST_GATE_HOURLY_USD = 1.0;

// Pure guard logic extracted for unit testing — mirrors the check in src/index.ts
function costSafetyGate(args: {
  gpuCount: number;
  costSafetyConfirmed?: boolean;
  dryRun: boolean;
}): "BLOCK" | "PASS" {
  if (args.gpuCount >= COST_GATE_GPU_COUNT && !args.dryRun) {
    // Simulate no-elicitation path (boolean gate fallback)
    if (!args.costSafetyConfirmed) {
      return "BLOCK";
    }
  }
  return "PASS";
}

// Elicitation result gate — mirrors the strict check in src/index.ts
function elicitationGate(result: { action?: string; content?: { confirmed?: boolean } } | null): "BLOCK" | "PASS" {
  const approved = result?.action === 'accept' && result?.content?.confirmed === true;
  return approved ? "PASS" : "BLOCK";
}

// Price-based gate — mirrors the COST_GATE_HOURLY_USD threshold
function priceGate(args: { ondemandPrice: number; gpuCount: number; dryRun: boolean }): "TRIGGERED" | "SKIP" {
  const estimatedHourlyCost = args.ondemandPrice * args.gpuCount;
  if (estimatedHourlyCost >= COST_GATE_HOURLY_USD && !args.dryRun) {
    return "TRIGGERED";
  }
  return "SKIP";
}

describe("create_pod_auto: costSafetyConfirmed gate (boolean fallback)", () => {
  it("gpuCount=1, dryRun=false → no block (single GPU is fine)", () => {
    expect(costSafetyGate({ gpuCount: 1, costSafetyConfirmed: undefined, dryRun: false })).toBe("PASS");
  });

  it("gpuCount=2, dryRun=false, costSafetyConfirmed=undefined → block", () => {
    expect(costSafetyGate({ gpuCount: 2, costSafetyConfirmed: undefined, dryRun: false })).toBe("BLOCK");
  });

  it("gpuCount=2, dryRun=false, costSafetyConfirmed=false → block", () => {
    expect(costSafetyGate({ gpuCount: 2, costSafetyConfirmed: false, dryRun: false })).toBe("BLOCK");
  });

  it("gpuCount=2, dryRun=false, costSafetyConfirmed=true → pass (user confirmed)", () => {
    expect(costSafetyGate({ gpuCount: 2, costSafetyConfirmed: true, dryRun: false })).toBe("PASS");
  });

  it("gpuCount=2, dryRun=true, costSafetyConfirmed=undefined → pass (dry run: warn but don't block)", () => {
    expect(costSafetyGate({ gpuCount: 2, costSafetyConfirmed: undefined, dryRun: true })).toBe("PASS");
  });
});

describe("create_pod_auto: elicitation gate", () => {
  it("elicitation accept + confirmed=true → pass", () => {
    expect(elicitationGate({ action: 'accept', content: { confirmed: true } })).toBe("PASS");
  });

  it("elicitation cancel → block", () => {
    expect(elicitationGate({ action: 'cancel' })).toBe("BLOCK");
  });

  it("elicitation decline → block", () => {
    expect(elicitationGate({ action: 'decline' })).toBe("BLOCK");
  });

  it("elicitation returns null → block", () => {
    expect(elicitationGate(null)).toBe("BLOCK");
  });

  it("elicitation accept + confirmed=false → block (unchecked checkbox)", () => {
    expect(elicitationGate({ action: 'accept', content: { confirmed: false } })).toBe("BLOCK");
  });

  it("elicitation accept + content undefined → block", () => {
    expect(elicitationGate({ action: 'accept' })).toBe("BLOCK");
  });
});

describe("create_pod_auto: price-based gate (COST_GATE_HOURLY_USD)", () => {
  it("gpuCount=1, ondemandPrice=0.5 → $0.5/hr, below threshold → skip", () => {
    expect(priceGate({ ondemandPrice: 0.5, gpuCount: 1, dryRun: false })).toBe("SKIP");
  });

  it("gpuCount=1, ondemandPrice=1.2 → $1.2/hr, above threshold → triggered", () => {
    expect(priceGate({ ondemandPrice: 1.2, gpuCount: 1, dryRun: false })).toBe("TRIGGERED");
  });

  it("gpuCount=2, ondemandPrice=0.6 → $1.2/hr, above threshold → triggered", () => {
    expect(priceGate({ ondemandPrice: 0.6, gpuCount: 2, dryRun: false })).toBe("TRIGGERED");
  });

  it("gpuCount=2, ondemandPrice=0.4 → $0.8/hr, below threshold → skip", () => {
    expect(priceGate({ ondemandPrice: 0.4, gpuCount: 2, dryRun: false })).toBe("SKIP");
  });

  it("gpuCount=2, ondemandPrice=1.0 → $2.0/hr, dryRun=true → skip (dry run bypass)", () => {
    expect(priceGate({ ondemandPrice: 1.0, gpuCount: 2, dryRun: true })).toBe("SKIP");
  });
});
