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
  lowestPrice: {
    minimumBidPrice: number;
    uninterruptablePrice: number;
    stockStatus: string; // "High" | "Medium" | "Low" | "Out of Stock"
  };
}

export interface CreatePodOptions {
  name: string;
  imageName: string;
  gpuTypeIds: string[];
  gpuCount?: number;
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
}

export interface RunPodApiConfig {
  apiKey: string;
  restBaseUrl: string;
  graphqlUrl: string;
  sshKeyPath?: string;
}
