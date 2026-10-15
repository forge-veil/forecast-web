import { CACHE_NAME, fetchBytes, resolveBase, resolveFiles } from "src/core/assets";
import { timesfm } from "src/models/timesfm/index";

describe("resolveBase", () => {
  it("maps a repo id to the Hugging Face resolve endpoint", () => {
    expect(resolveBase("org/name")).toBe("https://huggingface.co/org/name/resolve/main/");
    expect(resolveBase("org/name", "v1.0")).toBe("https://huggingface.co/org/name/resolve/v1.0/");
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
  it("builds URLs from the adapter's file spec", () => {
    const b = "https://x/";
    expect(resolveFiles(b, timesfm.files("q8")).model).toBe("https://x/onnx/model_quantized.onnx");
    expect(resolveFiles(b, timesfm.files("fp16"))).toEqual({
      config: "https://x/config.json",
      model: "https://x/onnx/model_fp16w.onnx",
      halfData: { url: "https://x/onnx/model_fp16w.f16", location: "model_fp16w.onnx_data" },
    });
    expect(resolveFiles(b, timesfm.files("q8")).halfData).toBeUndefined();
    expect(resolveFiles(b, timesfm.files("fp32")).model).toBe("https://x/onnx/model.onnx");
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

  // sha256("abc")
  const ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

  it("accepts a download whose SHA-256 matches", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(body([[97, 98], [99]]))));
    const bytes = await fetchBytes("https://x/m", { sha256: ABC });
    expect([...bytes]).toEqual([97, 98, 99]);
    expect(cache.store.has("https://x/m")).toBe(true);
  });

  it("rejects and never caches a download whose SHA-256 differs", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(body([[97, 98, 100]]))));
    await expect(fetchBytes("https://x/m", { sha256: ABC })).rejects.toMatchObject({ code: "LOAD_FAILED", message: expect.stringMatching(/SHA-256/) });
    expect(cache.store.size).toBe(0);
  });

  it("reports a hashing failure as LOAD_FAILED", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(body([[97, 98, 99]]))));
    vi.stubGlobal("crypto", { subtle: { digest: async () => { throw new RangeError("out of memory"); } } });
    await expect(fetchBytes("https://x/m", { sha256: ABC })).rejects.toMatchObject({ code: "LOAD_FAILED" });
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

  it("stores and finds entries under a custom cache key while fetching the real URL", async () => {
    const f = vi.fn(async () => new Response(body([[4, 2]])));
    vi.stubGlobal("fetch", f);
    await fetchBytes("https://x/m", { cacheKey: "https://x/m?sha256=abc" });
    expect(f).toHaveBeenCalledWith("https://x/m", expect.anything());
    expect(cache.store.has("https://x/m?sha256=abc")).toBe(true);
    expect(cache.store.has("https://x/m")).toBe(false);
    f.mockClear();
    expect(Array.from(await fetchBytes("https://x/m", { cacheKey: "https://x/m?sha256=abc" }))).toEqual([4, 2]);
    expect(f).not.toHaveBeenCalled();
  });

  it("networkFirst prefers fresh bytes and refreshes the cache", async () => {
    cache.store.set("https://x/c", new Response(new Uint8Array([1])));
    vi.stubGlobal("fetch", vi.fn(async () => new Response(body([[2]]))));
    expect(Array.from(await fetchBytes("https://x/c", { networkFirst: true }))).toEqual([2]);
    expect(Array.from(new Uint8Array(await (await cache.match("https://x/c"))!.arrayBuffer()))).toEqual([2]);
  });

  it("networkFirst falls back to the cache when offline", async () => {
    cache.store.set("https://x/c", new Response(new Uint8Array([1])));
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("offline"); }));
    expect(Array.from(await fetchBytes("https://x/c", { networkFirst: true }))).toEqual([1]);
  });

  it("networkFirst fails with LOAD_FAILED when offline with nothing cached", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("offline"); }));
    await expect(fetchBytes("https://x/c", { networkFirst: true })).rejects.toMatchObject({ code: "LOAD_FAILED" });
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
