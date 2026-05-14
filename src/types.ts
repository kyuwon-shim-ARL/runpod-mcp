export interface Pod {
  id: string;
  name: string;
  desiredStatus: string;
  lastStartedAt?: string;
  lastStatusChange?: string;
  publicIp?: string;
  portMappings?: Record<string, number>;
  ports?: string[];
  gpu?: {
    id: string;
    displayName: string;
    count: number;
  };
  vcpuCount?: number;
  memoryInGb?: number;
  /** CPU pod only — actual flavor RunPod assigned (single, even when request sent an array). */
  cpuFlavorId?: string;
  containerDiskInGb?: number;
  volumeInGb?: number;
  volumeMountPath?: string;
  costPerHr?: number;
  adjustedCostPerHr?: number;
  imageName?: string;
  env?: Record<string, string>;
  networkVolumeId?: string;
}

export interface GpuType {
  id: string;
  displayName: string;
  memoryInGb: number;
  communityCloud?: boolean;
  secureCloud?: boolean;
  communityPrice?: number;
  communitySpotPrice?: number;
  securePrice?: number;
  secureSpotPrice?: number;
  lowestPrice?: {
    minimumBidPrice: number;
    uninterruptablePrice: number;
    stockStatus: string; // "High" | "Medium" | "Low" | "Out of Stock"
  } | null;
}

export interface CreatePodOptions {
  name: string;
  imageName: string;
  /** GPU pod: required. CPU pod (computeType: "CPU"): omit. */
  gpuTypeIds?: string[];
  gpuCount?: number;
  /** "GPU" (default) or "CPU". When "CPU", gpuTypeIds is ignored by RunPod. */
  computeType?: "GPU" | "CPU";
  /** CPU pod only. Priority-ordered list. Valid: cpu3c, cpu3g, cpu3m, cpu5c, cpu5g, cpu5m. */
  cpuFlavorIds?: string[];
  /** CPU pod only. "availability" picks any available; "custom" honors cpuFlavorIds order. */
  cpuFlavorPriority?: "availability" | "custom";
  /** CPU pod only. Default 2. */
  vcpuCount?: number;
  interruptible?: boolean;
  containerDiskInGb?: number;
  volumeInGb?: number;
  volumeMountPath?: string;
  networkVolumeId?: string;
  ports?: string[];
  env?: Record<string, string>;
  sshPublicKey?: string;
  dockerArgs?: string;
  dockerStartCmd?: string[];
  dataCenterIds?: string[];
  supportPublicIp?: boolean;
  bidPerGpu?: number;
  cloudType?: "ALL" | "SECURE" | "COMMUNITY";
}

export interface CpuFlavor {
  id: string;
  displayName: string;
  generation: "cpu3" | "cpu5";
  family: "compute" | "general" | "highmem";
  cpuVendor: string;
  ramGbPerVcpu: number;
  /** Hourly price in USD. null when not catalogued — verify on RunPod Console. */
  hourlyPriceUsd: number | null;
}

export interface NetworkVolume {
  id: string;
  name: string;
  size: number;
  dataCenterId: string;
}

export interface RunPodApiConfig {
  apiKey: string;
  restBaseUrl: string;
  graphqlUrl: string;
  sshKeyPath?: string;
}
