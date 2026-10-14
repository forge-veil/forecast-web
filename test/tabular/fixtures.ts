import { readFileSync } from "node:fs";

/** Reads a JSON fixture from test/fixtures. */
export function fixture<T = any>(path: string): T {
  return JSON.parse(readFileSync(new URL(`../fixtures/${path}`, import.meta.url), "utf8"));
}
