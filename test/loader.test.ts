import { loadCore, planLoad, type CoreOptions, type SessionFactory } from "src/loader";
import type { SessionOptions } from "src/session";
import type { Progress } from "src/types";

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
    expect(planLoad(dtype, device, gpu)).toEqual({ dtype: d, device: dev, fallback: fb });
  });

  it("refuses q8 on explicit WebGPU with a clear message", () => {
    expect(() => planLoad("q8", "webgpu", true)).toThrow(/WASM only/);
  });
});

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

  it("rejects a config with an unexpected shape", async () => {
    files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify({ ...config, context: 1024 }));
    const { factory } = fakeSession();
    await expect(loadCore(opts({ device: "wasm" }), factory)).rejects.toThrow(/unsupported model shape/);
    files["https://m/config.json"] = new TextEncoder().encode(JSON.stringify(config));
  });

  it("reports a missing weights file as LOAD_FAILED", async () => {
    const { factory } = fakeSession();
    await expect(loadCore(opts({ base: "https://nope/", device: "wasm" }), factory)).rejects.toMatchObject({
      code: "LOAD_FAILED",
    });
  });
});
