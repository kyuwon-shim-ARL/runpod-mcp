/**
 * Sibling-pod grouping (issue #14).
 *
 * `.omc/pods/*.yaml` describes one pod each. When three pods are created for the same run and
 * only one is recorded, the record no longer says which pods belonged to the job — "delete the
 * pods for this job" becomes guesswork. `job_group` ties the records together, and `list_pods`
 * surfaces it so siblings are visible without opening the files.
 *
 * The reader is deliberately narrow: it pulls the two top-level scalars it needs out of the YAML
 * this repo's own `toYaml` writes, rather than adding a YAML parser dependency for two fields.
 */

import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

export const POD_METADATA_DIR = ".omc/pods";

/** podId → job_group, for pods that declare one. */
export type PodGroups = Record<string, string>;

/** Value of a top-level `key: value` scalar, or undefined. Indented lines are not top-level. */
function topLevelScalar(content: string, key: string): string | undefined {
  for (const line of content.split("\n")) {
    if (line.startsWith(" ") || line.startsWith("-") || line.startsWith("\t")) continue;
    const match = line.match(/^([A-Za-z_][\w-]*):\s*(.*)$/);
    if (!match || match[1] !== key) continue;
    const raw = match[2].trim();
    if (raw === "" || raw === "null" || raw === "~") return undefined;
    return raw.replace(/^["'](.*)["']$/, "$1");
  }
  return undefined;
}

/**
 * Scan pod metadata files for job_group assignments. Fail-soft: a missing directory, an
 * unreadable file, or a malformed record yields no entry rather than an error — grouping is
 * a convenience, and losing it must never break `list_pods`.
 */
export async function readPodGroups(dir: string = POD_METADATA_DIR): Promise<PodGroups> {
  let files: string[];
  try {
    files = await readdir(dir);
  } catch {
    return {};
  }

  const groups: PodGroups = {};
  for (const file of files) {
    if (!file.endsWith(".yaml") && !file.endsWith(".yml")) continue;
    try {
      const content = await readFile(join(dir, file), "utf8");
      const podId = topLevelScalar(content, "pod_id");
      const group = topLevelScalar(content, "job_group");
      if (podId && group) groups[podId] = group;
    } catch {
      continue;
    }
  }
  return groups;
}

/** Other pods in this pod's group that are still live. */
export function groupSiblings(podId: string, groups: PodGroups, livePodIds: string[]): string[] {
  const group = groups[podId];
  if (!group) return [];
  const live = new Set(livePodIds);
  return Object.entries(groups)
    .filter(([id, g]) => g === group && id !== podId && live.has(id))
    .map(([id]) => id);
}

/** The "Group: ..." line for a pod in `list_pods` output. Empty when the pod has no group. */
export function renderGroupLine(group: string | undefined, siblings: string[]): string {
  if (!group) return "";
  return siblings.length > 0
    ? `Group: ${group} (siblings live: ${siblings.join(", ")})`
    : `Group: ${group} (no live siblings)`;
}
