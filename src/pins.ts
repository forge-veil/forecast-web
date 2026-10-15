import PINS from "src/pins.json";

/** A published weights repo pinned to a commit, with the size and SHA-256 of each file there. */
export interface Pin {
  revision: string;
  files: Record<string, { bytes: number; sha256: string }>;
}

export function pinOf(repo: string): Pin {
  const pin = (PINS as Record<string, Pin>)[repo];
  if (!pin) throw new Error(`no pin for ${repo}; run scripts/pin-weights.mjs`);
  return pin;
}

/**
 * Revision and expected hashes for a load. The published weights at their pinned commit are
 * verified against hashes built into the package; any other source falls back to its config.json.
 */
export function resolvePin(published: string, weights?: string, revision?: string): { revision: string; hashes?: Pin["files"] } {
  if (weights !== undefined && weights !== published) return { revision: revision ?? "main" };
  const pin = pinOf(published);
  const rev = revision ?? pin.revision;
  return rev === pin.revision ? { revision: rev, hashes: pin.files } : { revision: rev };
}
