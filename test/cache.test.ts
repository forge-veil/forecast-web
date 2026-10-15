import { clearCache } from "src/cache";
import { CACHE_NAME } from "src/core/assets";
import { pinOf } from "src/pins";

const tiny = pinOf("bakshinghampalace/chronos-bolt-tiny-onnx");
const base = pinOf("bakshinghampalace/chronos-bolt-base-onnx");
const key = (sha: string) => `https://forecast-web.invalid/sha256/${sha}`;

function fakeCaches(urls: string[]) {
  const store = new Set(urls);
  const cache = {
    keys: async () => [...store].map((u) => new Request(u)),
    delete: async (r: Request | string) => store.delete(typeof r === "string" ? r : r.url),
  };
  const deleted: string[] = [];
  vi.stubGlobal("caches", {
    open: async (name: string) => (name === CACHE_NAME ? cache : null),
    delete: async (name: string) => (deleted.push(name), true),
  });
  return { store, deleted };
}

describe("clearCache", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("deletes everything without a model", async () => {
    const { deleted } = fakeCaches([]);
    await expect(clearCache()).resolves.toBe(true);
    expect(deleted).toEqual([CACHE_NAME]);
  });

  it("deletes one model's pinned files and its older repo entries, nothing else", async () => {
    const tinyFile = key(Object.values(tiny.files)[0].sha256);
    const baseFile = key(Object.values(base.files)[0].sha256);
    const old = "https://huggingface.co/bakshinghampalace/chronos-bolt-tiny-onnx/resolve/main/onnx/model_fp16w.f16";
    const config = `https://huggingface.co/bakshinghampalace/chronos-bolt-tiny-onnx/resolve/${tiny.revision}/config.json`;
    const custom = "https://cdn.example/w/onnx/model.onnx";
    const { store } = fakeCaches([tinyFile, baseFile, old, config, custom]);
    await expect(clearCache("chronos-bolt-tiny")).resolves.toBe(true);
    expect([...store].sort()).toEqual([baseFile, custom].sort());
    await expect(clearCache("chronos-bolt-tiny")).resolves.toBe(false);
  });

  it("works for tabular models and rejects unknown ids", async () => {
    const cls = pinOf("bakshinghampalace/tabicl-classifier-v2-onnx");
    const { store } = fakeCaches([key(Object.values(cls.files)[0].sha256)]);
    await expect(clearCache("tabicl-classifier-v2")).resolves.toBe(true);
    expect(store.size).toBe(0);
    await expect(clearCache("nope" as never)).rejects.toMatchObject({ code: "UNKNOWN_MODEL" });
  });

  it("resolves false without the Cache API", async () => {
    vi.stubGlobal("caches", undefined);
    await expect(clearCache("chronos-bolt-tiny")).resolves.toBe(false);
    await expect(clearCache()).resolves.toBe(false);
  });
});
