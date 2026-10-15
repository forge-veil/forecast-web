export {};

vi.mock("src/core/session", () => ({ createSession: vi.fn() }));
const { loadForecaster, MODELS } = await import("src/index");

describe("loadForecaster", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("rejects an unknown model before fetching", async () => {
    const f = vi.fn();
    vi.stubGlobal("fetch", f);
    await expect(loadForecaster({ model: "nope" as never, worker: false })).rejects.toMatchObject({
      code: "UNKNOWN_MODEL",
    });
    expect(f).not.toHaveBeenCalled();
  });

  it("loads from the weights option when given", async () => {
    const f = vi.fn(async (_url: string) => new Response("", { status: 404 }));
    vi.stubGlobal("fetch", f);
    await expect(
      loadForecaster({ weights: "https://w.example/tfm", worker: false, device: "wasm", cache: false }),
    ).rejects.toMatchObject({ code: "LOAD_FAILED" });
    expect(f.mock.calls[0][0]).toBe("https://w.example/tfm/config.json");
  });

  it("defaults to the model's published weights", async () => {
    const f = vi.fn(async (_url: string) => new Response("", { status: 404 }));
    vi.stubGlobal("fetch", f);
    await loadForecaster({ worker: false, device: "wasm", cache: false }).catch(() => {});
    expect(f.mock.calls[0][0]).toBe(
      "https://huggingface.co/bakshinghampalace/timesfm-2.5-200m-onnx/resolve/0ee24a65f124d76ce21af21c0fc228f94b71140a/config.json",
    );
  });

  it("uses main for custom weights repos and an explicit revision when given", async () => {
    const f = vi.fn(async (_url: string) => new Response("", { status: 404 }));
    vi.stubGlobal("fetch", f);
    await loadForecaster({ weights: "me/my-weights", worker: false, device: "wasm", cache: false }).catch(() => {});
    await loadForecaster({ model: "chronos-bolt-tiny", revision: "v2", worker: false, device: "wasm", cache: false }).catch(() => {});
    expect(f.mock.calls[0][0]).toBe("https://huggingface.co/me/my-weights/resolve/main/config.json");
    expect(f.mock.calls[1][0]).toBe("https://huggingface.co/bakshinghampalace/chronos-bolt-tiny-onnx/resolve/v2/config.json");
  });

  it("keeps the pinned revision when weights names the published repo", async () => {
    const f = vi.fn(async (_url: string) => new Response("", { status: 404 }));
    vi.stubGlobal("fetch", f);
    await loadForecaster({ weights: "bakshinghampalace/timesfm-2.5-200m-onnx", worker: false, device: "wasm", cache: false }).catch(() => {});
    expect(f.mock.calls[0][0]).toContain("/resolve/0ee24a65f124d76ce21af21c0fc228f94b71140a/");
  });

  it("exports exactly the public runtime API", async () => {
    const mod = await import("src/index");
    expect(Object.keys(mod).sort()).toEqual(["DEFAULT_MODEL_ID", "ForecastError", "MODELS", "clearCache", "loadForecaster"]);
  });

  it("exports MODELS", () => {
    expect(Object.keys(MODELS)).toEqual([
      "timesfm-2.5",
      "chronos-bolt-tiny",
      "chronos-bolt-mini",
      "chronos-bolt-small",
      "chronos-bolt-base",
    ]);
  });
});
