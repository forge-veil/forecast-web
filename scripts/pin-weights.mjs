// Pins published weights: records a Hugging Face commit and the size and SHA-256 of every
// weight file at that commit in src/pins.json, which the package verifies downloads against.
//
//   node scripts/pin-weights.mjs <owner/repo>[@revision] ...   pin these repos (default: their latest commit)
//   node scripts/pin-weights.mjs                               re-pin every repo already in pins.json to its latest commit
import { readFileSync, writeFileSync } from "node:fs";

const PATH = new URL("../src/pins.json", import.meta.url);
const pins = JSON.parse(readFileSync(PATH, "utf8"));
const get = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} returned HTTP ${res.status}`);
  return res.json();
};

const targets = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(pins);
for (const target of targets) {
  const [repo, wanted] = target.split("@");
  const { sha } = await get(`https://huggingface.co/api/models/${repo}/revision/${wanted ?? "main"}`);
  const config = await get(`https://huggingface.co/${repo}/resolve/${sha}/config.json`);
  if (!config.files || !Object.keys(config.files).length) throw new Error(`${repo}@${sha}: config.json lists no files`);
  const files = Object.fromEntries(Object.entries(config.files).sort(([a], [b]) => a.localeCompare(b)).map(([f, m]) => [f, { bytes: m.bytes, sha256: m.sha256 }]));
  pins[repo] = { revision: sha, files };
  console.log(`${repo}: ${sha} (${Object.keys(files).length} files)`);
}
writeFileSync(PATH, JSON.stringify(Object.fromEntries(Object.entries(pins).sort(([a], [b]) => a.localeCompare(b))), null, 2) + "\n");
