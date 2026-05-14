// Ad-hoc CPU pod smoke test against real RunPod API.
// Creates smallest CPU pod (2 vCPU compute-optimized), inspects response,
// then deletes immediately. Cost ≈ $0.005 for ~1 minute.
//
// Run: RUNPOD_API_KEY=... node scripts/e2e-cpu-smoke.mjs

import { RunPodClient } from "../dist/api.js";

if (!process.env.RUNPOD_API_KEY) {
  console.error("RUNPOD_API_KEY not set");
  process.exit(1);
}

const c = new RunPodClient({
  apiKey: process.env.RUNPOD_API_KEY,
  restBaseUrl: "https://rest.runpod.io/v1",
  graphqlUrl: "https://api.runpod.io/graphql",
});

let podId = null;
const t0 = Date.now();

try {
  console.log("[1/4] createPod (CPU mode, 2 vCPU, cpu5c→cpu3c)...");
  const pod = await c.createPod({
    name: `cpu-qa-smoke-${Date.now()}`,
    imageName: "ubuntu:22.04",
    computeType: "CPU",
    vcpuCount: 2,
    cpuFlavorIds: ["cpu5c", "cpu3c"],
    cpuFlavorPriority: "availability",
    containerDiskInGb: 10,
    volumeInGb: 0,
    ports: ["22/tcp"],
    cloudType: "COMMUNITY",
  });
  podId = pod.id;
  console.log("    pod.id:", pod.id);
  console.log("    pod.desiredStatus:", pod.desiredStatus);
  console.log("    pod.gpu:", JSON.stringify(pod.gpu));
  console.log("    pod.vcpuCount:", pod.vcpuCount);
  console.log("    pod.memoryInGb:", pod.memoryInGb);
  console.log("    pod.costPerHr:", pod.costPerHr);

  console.log("\n[2/4] Wait 4s, then getPod to see fully-populated record...");
  await new Promise((r) => setTimeout(r, 4000));
  const fresh = await c.getPod(pod.id);
  console.log("    fresh.desiredStatus:", fresh.desiredStatus);
  console.log("    fresh.gpu:", JSON.stringify(fresh.gpu));
  console.log("    fresh.vcpuCount:", fresh.vcpuCount);
  console.log("    fresh.memoryInGb:", fresh.memoryInGb);
  console.log("    fresh.costPerHr:", fresh.costPerHr);
  console.log("    fresh.imageName:", fresh.imageName);

  console.log("\n[3/4] Full pod object dump:");
  console.log(JSON.stringify(fresh, null, 2));
} catch (e) {
  console.error("\nERROR:", e?.message ?? e);
  console.error(e?.stack ?? "");
} finally {
  if (podId) {
    console.log(`\n[4/4] deletePod(${podId})...`);
    try {
      await c.deletePod(podId);
      console.log("    deleted OK");
    } catch (e) {
      console.error("    DELETE FAILED:", e?.message);
      console.error("    MANUAL CLEANUP REQUIRED:", podId);
    }
  }
  const dur = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\nTotal duration: ${dur}s (cost ~ $${(0.2 * (dur / 3600)).toFixed(4)} at $0.20/hr)`);
}
