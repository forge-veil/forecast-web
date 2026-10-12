import { CACHE_NAME, DEFAULT_MODEL, fetchBytes, resolveBase, resolveFiles } from "src/assets";

describe("resolveBase", () => {
  it("maps a repo id to the Hugging Face resolve endpoint", () => {
    expect(resolveBase("org/name")).toBe("https://huggingface.co/org/name/resolve/main/");
    expect(resolveBase("org/name", "v1.0")).toBe("https://huggingface.co/org/name/resolve/v1.0/");
  });

  it("uses the default repo when no model is given", () => {
    expect(resolveBase()).toBe(`https://huggingface.co/${DEFAULT_MODEL}/resolve/main/`);
  });

  it("passes absolute URLs through and adds a trailing slash", () => {
    expect(resolveBase("https://cdn.example.com/tfm")).toBe("https://cdn.example.com/tfm/");
  });

  it("resolves relative URLs against the page so workers get absolute ones", () => {
    expect(resolveBase("/models/tfm/", undefined, "https://site.dev/blog/post")).toBe("https://site.dev/models/tfm/");
    expect(resolveBase("./tfm", undefined, "https://site.dev/blog/post")).toBe("https://site.dev/blog/tfm/");
  });

  it("rejects strings that are neither", () => {
    expect(() => resolveBase("not a repo")).toThrow(/neither/);
  });
});

describe("resolveFiles", () => {
  it("uses Transformers.js file names per dtype", () => {
    const b = "https://x/";
    expect(resolveFiles(b).model).toBe("https://x/onnx/model_quantized.onnx");
    expect(resolveFiles(b, "fp16")).toEqual({
      config: "https://x/config.json",
      model: "https://x/onnx/model_fp16w.onnx",
      halfData: { url: "https://x/onnx/model_fp16w.f16", location: "model_fp16w.onnx_data" },
    });
    expect(resolveFiles(b).halfData).toBeUndefined();
    expect(resolveFiles(b, "fp32").model).toBe("https://x/onnx/model.onnx");
    expect(resolveFiles(b).config).toBe("https://x/config.json");
  });
});

describe("fetchBytes", () => {
  const body = (chunks: number[][]) =>
    new ReadableStream<Uint8Array>({
      start(c) {
        for (const ch of chunks) c.enqueue(new Uint8Array(ch));
        c.close();
      },
    });

  class FakeCache {
    store = new Map<string, Response>();
    putError: Error | null = null;
    async match(url: string) {
      return this.store.get(url)?.clone();
    }
    async put(url: string, res: Response) {
      if (this.putError) throw this.putError;
      this.store.set(url, res);
    }
  }

  let cache: FakeCache;
  beforeEach(() => {
    cache = new FakeCache();
    vi.stubGlobal("caches", { open: vi.fn(async (name: string) => (name === CACHE_NAME ? cache : null)) });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("streams the body and reports progress against content-length", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(body([[1, 2], [3, 4, 5]]), { headers: { "content-length": "5" } })),
    );
    const seen: [number, number, boolean][] = [];
    const bytes = await fetchBytes("https://x/m", { onProgress: (l, t, c) => seen.push([l, t, c]) });
    expect(Array.from(bytes)).toEqual([1, 2, 3, 4, 5]);
    expect(seen).toEqual([
      [2, 5, false],
      [5, 5, false],
    ]);
  });

  it("serves a cache hit without touching the network", async () => {
    cache.store.set("https://x/m", new Response(new Uint8Array([9, 9])));
    const f = vi.fn();
    vi.stubGlobal("fetch", f);
    const seen: boolean[] = [];
    const bytes = await fetchBytes("https://x/m", { onProgress: (_l, _t, c) => seen.push(c) });
    expect(Array.from(bytes)).toEqual([9, 9]);
    expect(f).not.toHaveBeenCalled();
    expect(seen).toEqual([true]);
  });

  it("stores downloads in the cache", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(body([[7]]))));
    await fetchBytes("https://x/m");
    expect(cache.store.has("https://x/m")).toBe(true);
  });

  it("still returns the bytes when the cache is full", async () => {
    cache.putError = new DOMException("full", "QuotaExceededError");
    vi.stubGlobal("fetch", vi.fn(async () => new Response(body([[1, 2, 3]]))));
    expect(Array.from(await fetchBytes("https://x/m"))).toEqual([1, 2, 3]);
  });

  it("skips the cache entirely when cache is false", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(body([[1]]))));
    await fetchBytes("https://x/m", { cache: false });
    expect(cache.store.size).toBe(0);
  });

  it("never reports more than 100% when content-length is the compressed size", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(body([[1, 2, 3, 4]]), { headers: { "content-length": "2" } })),
    );
    const seen: [number, number][] = [];
    await fetchBytes("https://x/m", { onProgress: (l, t) => seen.push([l, t]) });
    expect(seen.at(-1)).toEqual([4, 4]);
  });

  it("maps HTTP errors to LOAD_FAILED", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("nope", { status: 404 })));
    await expect(fetchBytes("https://x/m")).rejects.toMatchObject({ code: "LOAD_FAILED" });
  });

  it("maps an abort to ABORTED", async () => {
    const ac = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_u: string, init: RequestInit) => {
        ac.abort();
        throw init.signal?.reason ?? new Error("aborted");
      }),
    );
    await expect(fetchBytes("https://x/m", { signal: ac.signal })).rejects.toMatchObject({ code: "ABORTED" });
  });

  it("rejects immediately when already aborted", async () => {
    const ac = new AbortController();
    ac.abort();
    await expect(fetchBytes("https://x/m", { signal: ac.signal })).rejects.toMatchObject({ code: "ABORTED" });
  });
});
