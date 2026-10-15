import { createHash } from "node:crypto";
import { loadCore, planLoad, type CoreOptions, type SessionFactory } from "src/core/loader";
import type { SessionOptions } from "src/core/session";
import { MODELS } from "src/models/index";
import type { ModelInfo, Progress } from "src/types";

const info = MODELS["timesfm-2.5"];

describe("planLoad", () => {
  it.each([
    // dtype, device, gpu available -> dtype, device, fallback
    ["auto", "auto", true, "fp16", "webgpu", true],
    ["auto", "auto", false, "q8", "wasm", false],
    ["auto", "wasm", true, "q8", "wasm", false],
    ["auto", "webgpu", false, "fp16", "webgpu", false],
    ["q8", "auto", true, "q8", "wasm", false],
    ["fp32", "auto", true, "fp32", "webgpu", true],
    ["fp16", "wasm", true, "fp16", "wasm", false],
  ] as const)("%s on %s (gpu=%s) -> %s on %s", (dtype, device, gpu, d, dev, fb) => {
    expect(planLoad(info, dtype, device, gpu)).toEqual({ dtype: d, device: dev, fallback: fb });
  });

  it("refuses q8 on explicit WebGPU with a clear message", () => {
    expect(() => planLoad(info, "q8", "webgpu", true)).toThrow(/WASM only/);
  });

  it("rejects a dtype the model does not ship, naming the options", () => {
    const only: ModelInfo = { ...info, dtypes: ["q8"] };
    expect(() => planLoad(only, "fp32", "wasm", false)).toThrow(/TimesFM 2.5 has no "fp32" weights; choose one of q8/);
  });

  it("auto follows the model's per-backend defaults", () => {
    const custom: ModelInfo = { ...info, defaults: { webgpu: "fp32", wasm: "fp16" } };
    expect(planLoad(custom, "auto", "auto", true).dtype).toBe("fp32");
    expect(planLoad(custom, "auto", "auto", false).dtype).toBe("fp16");
  });
});

/** Hex SHA-256, as config.json lists it. */
const sha = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");

describe("loadCore", () => {
  const config = { model_type: "timesfm", context: 512, horizon: 128, patch: 32, quantiles: [] };
  const files: Record<string, Uint8Array> = {
    "https://m/config.json": new TextEncoder().encode(JSON.stringify(config)),
    "https://m/onnx/model_quantized.onnx": new Uint8Array(10),
    "https://m/onnx/model_fp16w.onnx": new Uint8Array(4),
    "https://m/onnx/model_fp16w.f16": new Uint8Array([0x00, 0x3c, 0x00, 0x40]), // 1.0, 2.0
  };

  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        files[url]
          ? new Response(files[url].slice().buffer, { headers: { "content-length": String(files[url].length) } })
          : new Response("", { status: 404 }),
      ),
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  const opts = (o: Partial<CoreOptions>): CoreOptions => ({
    model: "timesfm-2.5",
    base: "https://m/",
    dtype: "auto",
    device: "auto",
    cache: false,
    ...o,
  });

  function fakeSession() {
    const calls: { bytes: Uint8Array; opts: SessionOptions }[] = [];
    const factory: SessionFactory = async (bytes, o) => {
      calls.push({ bytes, opts: o });
      return { backend: o.device, run: vi.fn(), release: vi.fn(async () => {}) };
    };
    return { calls, factory };
  }

  it("auto on a GPU machine loads the fp16 pair and expands the weights", async () => {
    const { calls, factory } = fakeSession();
    const seen: Progress[] = [];
    const core = await loadCore(opts({}), factory, (p) => seen.push(p), undefined, async () => true);
    expect(core.dtype).toBe("fp16");
    expect(core.backend).toBe("webgpu");
    const ext = calls[0].opts.externalData!;
    expect(ext[0].path).toBe("model_fp16w.onnx_data");
    expect(Array.from(new Float32Array(ext[0].data.buffer))).toEqual([1, 2]);
    expect(calls[0].opts.fallback).toBe(true);
    // progress sums both files: 4-byte graph + 4-byte weights
    expect(seen.at(-1)).toEqual({ loaded: 8, total: 8, cached: false });
  });

  it("auto without a GPU loads q8 on WASM with no external data", async () => {
    const { calls, factory } = fakeSession();
    const core = await loadCore(opts({}), factory, undefined, undefined, async () => false);
    expect(core.dtype).toBe("q8");
    expect(calls[0].opts).toMatchObject({ device: "wasm", externalData: undefined });
    expect(calls[0].bytes.length).toBe(10);
  });

  it("skips the GPU probe when the device is explicit", async () => {
    const probe = vi.fn(async () => true);
    const { factory } = fakeSession();
    const core = await loadCore(opts({ device: "wasm" }), factory, undefined, undefined, probe);
    expect(probe).not.toHaveBeenCalled();
    expect(core.dtype).toBe("q8");
  });

  it("reports monotonic progress against the sizes listed in config.json", async () => {
    const sized = {
      ...config,
      files: {
        "onnx/model_fp16w.onnx": { bytes: 4, sha256: sha(files["https://m/onnx/model_fp16w.onnx"]) },
        "onnx/model_fp16w.f16": { bytes: 4, sha256: sha(files["https://m/onnx/model_fp16w.f16"]) },
      },
    };
    files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify(sized));
    const { factory } = fakeSession();
    const seen: Progress[] = [];
    await loadCore(opts({}), factory, (p) => seen.push(p), undefined, async () => true);
    expect(seen.length).toBeGreaterThan(0);
    for (const p of seen) expect(p.total).toBe(8);
    for (let i = 1; i < seen.length; i++) expect(seen[i].loaded).toBeGreaterThanOrEqual(seen[i - 1].loaded);
    files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify(config));
  });

  it("fails with LOAD_FAILED when a weight file does not match its listed SHA-256", async () => {
    const listed = { ...config, files: { "onnx/model_quantized.onnx": { bytes: 10, sha256: sha(new Uint8Array(10).fill(1)) } } };
    files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify(listed));
    const { factory } = fakeSession();
    try {
      await expect(loadCore(opts({ device: "wasm" }), factory)).rejects.toMatchObject({ code: "LOAD_FAILED", message: expect.stringMatching(/SHA-256/) });
    } finally {
      files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify(config));
    }
  });

    it("verifies against built-in hashes instead of config.json when given", async () => {
    const good = sha(files["https://m/onnx/model_quantized.onnx"]);
    const bad = sha(new Uint8Array(10).fill(1));
    const { factory } = fakeSession();
    try {
      // config.json lies, the built-in hash is right: loads.
      files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify({ ...config, files: { "onnx/model_quantized.onnx": { bytes: 10, sha256: bad } } }));
      await expect(loadCore({ ...opts({ device: "wasm" }), hashes: { "onnx/model_quantized.onnx": { bytes: 10, sha256: good } } }, factory)).resolves.toBeTruthy();
      // config.json agrees with a tampered file, the built-in hash does not: fails.
      files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify({ ...config, files: { "onnx/model_quantized.onnx": { bytes: 10, sha256: good } } }));
      await expect(loadCore({ ...opts({ device: "wasm" }), hashes: { "onnx/model_quantized.onnx": { bytes: 10, sha256: bad } } }, factory)).rejects.toMatchObject({ code: "LOAD_FAILED" });
      // A built-in list without the file fails closed.
      await expect(loadCore({ ...opts({ device: "wasm" }), hashes: {} }, factory)).rejects.toMatchObject({ code: "LOAD_FAILED", message: expect.stringMatching(/no pinned/) });
    } finally {
      files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify(config));
    }
  });

  it("does not report progress until every file's size is known", async () => {
    const { factory } = fakeSession();
    const seen: Progress[] = [];
    await loadCore(opts({}), factory, (p) => seen.push(p), undefined, async () => true);
    for (const p of seen) expect(p.total).toBe(8);
  });

  it("keys cached weights by the content hash from config.json", async () => {
    const store = new Map<string, Response>();
    vi.stubGlobal("caches", {
      open: async () => ({
        match: async (k: string) => store.get(k)?.clone(),
        put: async (k: string, r: Response) => void store.set(k, r),
      }),
    });
    const hash = sha(files["https://m/onnx/model_quantized.onnx"]);
    const sized = { ...config, files: { "onnx/model_quantized.onnx": { bytes: 10, sha256: hash } } };
    files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify(sized));
    const { factory } = fakeSession();
    await loadCore(opts({ device: "wasm", cache: true }), factory);
    // Keyed by content alone, so a new revision with the same bytes reuses the cached copy.
    expect([...store.keys()]).toContain(`https://forecast-web.invalid/sha256/${hash}`);
    files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify(config));
  });

  it("always fetches config.json from the network so new hashes are seen", async () => {
    const store = new Map<string, Response>([["https://m/config.json", new Response('{"stale":true}')]]);
    vi.stubGlobal("caches", {
      open: async () => ({
        match: async (k: string) => store.get(k)?.clone(),
        put: async (k: string, r: Response) => void store.set(k, r),
      }),
    });
    const { factory } = fakeSession();
    const core = await loadCore(opts({ device: "wasm", cache: true }), factory);
    expect(core.config.context).toBe(512);
  });

  it("cancels the other download when one file fails", async () => {
    let graphSignal: AbortSignal | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        if (url.endsWith("config.json")) return new Response(files["https://m/config.json"].slice().buffer);
        if (url.endsWith(".f16")) return new Response("", { status: 404 });
        graphSignal = init?.signal ?? undefined;
        return new Promise<Response>((_r, reject) =>
          init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))),
        );
      }),
    );
    const { factory } = fakeSession();
    await expect(loadCore(opts({}), factory, undefined, undefined, async () => true)).rejects.toMatchObject({
      code: "LOAD_FAILED",
    });
    expect(graphSignal?.aborted).toBe(true);
  });

  it("releases the session and rejects when aborted during session creation", async () => {
    const ac = new AbortController();
    const release = vi.fn(async () => {});
    const factory: SessionFactory = async (_b, o) => {
      ac.abort();
      return { backend: o.device, run: vi.fn(), release };
    };
    await expect(loadCore(opts({ device: "wasm" }), factory, undefined, ac.signal)).rejects.toMatchObject({
      code: "ABORTED",
    });
    expect(release).toHaveBeenCalledTimes(1);
  });

  it("rejects a config with an unexpected shape", async () => {
    files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify({ ...config, context: 1024 }));
    const { factory } = fakeSession();
    await expect(loadCore(opts({ device: "wasm" }), factory)).rejects.toThrow(/unsupported model shape/);
    files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify(config));
  });

  it("rejects an unknown model before fetching anything", async () => {
    const { factory } = fakeSession();
    await expect(loadCore(opts({ model: "nope" as never }), factory)).rejects.toMatchObject({ code: "UNKNOWN_MODEL" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("reports a missing weights file as LOAD_FAILED", async () => {
    const { factory } = fakeSession();
    await expect(loadCore(opts({ base: "https://nope/", device: "wasm" }), factory)).rejects.toMatchObject({
      code: "LOAD_FAILED",
    });
  });
});

describe("loadCore with a tabular model", () => {
  const cfg = { model_type: "tabicl", task: "classification", max_classes: 10 };
  const files: Record<string, Uint8Array> = {
    "https://t/config.json": new TextEncoder().encode(JSON.stringify(cfg)),
    "https://t/onnx/model.onnx": new Uint8Array(6),
  };
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => (files[url] ? new Response(files[url].slice().buffer) : new Response("", { status: 404 }))),
    );
  });
  afterEach(() => vi.unstubAllGlobals());
  const opts = (o: Partial<CoreOptions> = {}): CoreOptions => ({ model: "tabicl-classifier-v2", base: "https://t/", dtype: "fp32", device: "wasm", cache: false, ...o });
  const factory: SessionFactory = async (_b, o) => ({ backend: o.device, run: vi.fn(async () => ({ logits: new Float32Array(10) })), release: vi.fn(async () => {}) });

  it("loads the graph, exposes run, and rejects forecast", async () => {
    const core = await loadCore(opts(), factory);
    expect(Object.keys(await core.run({}))).toEqual(["logits"]);
    await expect(core.forecast([[1, 2, 3]])).rejects.toThrow(/not a forecasting model/);
  });

  it("rejects a config for the other task", async () => {
    files["https://t/config.json"] = new TextEncoder().encode(JSON.stringify({ ...cfg, task: "regression" }));
    await expect(loadCore(opts(), factory)).rejects.toThrow(/expected a TabICL classification graph/);
    files["https://t/config.json"] = new TextEncoder().encode(JSON.stringify(cfg));
  });
});

